from qiskit import QuantumRegister
import os
import torch
from torch import cat, no_grad, manual_seed
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from torch.nn import Module
from torch.nn import Conv2d, Dropout2d, Linear
from torch.nn import NLLLoss
from torch.nn import functional as F
from qiskit_machine_learning.optimizers import COBYLA

# Initialize the Estimator
estimator = Estimator()

# Create directory if it doesn't exist
if not os.path.exists('tutorial1'):
    os.makedirs('tutorial1')

# Train Dataset
# -------------

# Set train shuffle seed (for reproducibility)
manual_seed(42)

batch_size = 1
n_samples = 100  # We will concentrate on the first 100 samples

# Use pre-defined torchvision function to load MNIST train data
X_train = datasets.MNIST(
    root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
)

# Filter out labels (originally 0-9), leaving only labels 0 and 1
idx = np.append(
    np.where(X_train.targets == 0)[0][:n_samples], np.where(X_train.targets == 1)[0][:n_samples]
)
X_train.data = X_train.data[idx]
X_train.targets = X_train.targets[idx]

# Define torch dataloader with filtered data
train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)

# Visualize some training samples
n_samples_show = 6
data_iter = iter(train_loader)
fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

while n_samples_show > 0:
    images, targets = data_iter.__next__()

    axes[n_samples_show - 1].imshow(images[0, 0].numpy().squeeze(), cmap="gray")
    axes[n_samples_show - 1].set_xticks([])
    axes[n_samples_show - 1].set_yticks([])
    axes[n_samples_show - 1].set_title("Labeled: {}".format(targets[0].item()))

    n_samples_show -= 1

fig.savefig('tutorial1/train_samples.png')

# Test Dataset
# -------------

# Set test shuffle seed (for reproducibility)
# manual_seed(5)

n_samples = 50

# Use pre-defined torchvision function to load MNIST test data
X_test = datasets.MNIST(
    root="./data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])
)

# Filter out labels (originally 0-9), leaving only labels 0 and 1
idx = np.append(
    np.where(X_test.targets == 0)[0][:n_samples], np.where(X_test.targets == 1)[0][:n_samples]
)
X_test.data = X_test.data[idx]
X_test.targets = X_test.targets[idx]

# Define torch dataloader with filtered data
test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=True)

# Function to encode an image in FRQI format
def frqi_encoding(image):
    n = int(np.log2(len(image)))  # Number of qubits needed
    qreg = QuantumRegister(n + 1)  # Extra qubit for grayscale
    qc = QuantumCircuit(qreg)
    
    # Initialize with Hadamard transform
    for i in range(n):
        qc.h(qreg[i])
    
    # Encode pixel intensities
    for i, pixel in enumerate(image):
        theta = 2 * np.arccos(np.sqrt(pixel))
        bin_index = format(i, f'0{n}b')
        for j, bit in enumerate(bin_index):
            if bit == '1':
                qc.x(qreg[j])
        qc.cry(theta, qreg[n-1], qreg[n])
        for j, bit in enumerate(bin_index):
            if bit == '1':
                qc.x(qreg[j])
    
    # Store the number of qubits as a custom attribute
    qc.custom_num_qubits = n + 1
    return qc

# Function to encode an image in NEQR format
def neqr_encoding(image):
    n = int(np.log2(len(image)))
    qreg = QuantumRegister(n + 8)  # Extra 8 qubits for grayscale (256 levels)
    qc = QuantumCircuit(qreg)
    
    # Initialize position qubits with Hadamard
    for i in range(n):
        qc.h(qreg[i])
    
    # Encode pixel values
    for i, pixel in enumerate(image):
        bin_pixel = format(int(pixel * 255), '08b')
        bin_index = format(i, f'0{n}b')
        
        for j, bit in enumerate(bin_index):
            if bit == '1':
                qc.x(qreg[j])
        
        for j, bit in enumerate(bin_pixel):
            if bit == '1':
                qc.mcx(qreg[:n], qreg[n + j])
        
        for j, bit in enumerate(bin_index):
            if bit == '1':
                qc.x(qreg[j])
    
    return qc

# Function to create a Quantum Neural Network (QNN) with different feature maps
def create_qnn(encoding_method='ZZFeatureMap', image=None):
    if encoding_method == 'ZZFeatureMap':
        feature_map = ZZFeatureMap(2)
        num_qubits = 2
        input_params = feature_map.parameters
    elif encoding_method == 'FRQI':
        feature_map = frqi_encoding(image)  # Pass the image data
        num_qubits = feature_map.custom_num_qubits  # Use the custom attribute
        input_params = []  # FRQI encoding does not use input parameters
    elif encoding_method == 'NEQR':
        feature_map = neqr_encoding(image)  # Pass the image data
        num_qubits = feature_map.num_qubits  # Assuming NEQR sets num_qubits correctly
        input_params = []  # NEQR encoding does not use input parameters
    else:
        raise ValueError("Unknown encoding method. Choose from ['ZZFeatureMap', 'FRQI', 'NEQR']")
    
    ansatz = RealAmplitudes(num_qubits, reps=1)
    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    # Create the QNN
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=input_params,
        weight_params=ansatz.parameters,
        input_gradients=True,
        estimator=estimator,
    )
    return qnn

# Define a neural network model with a QNN layer
class Net(Module):
    def __init__(self, qnn):
        super().__init__()
        self.conv1 = Conv2d(1, 2, kernel_size=5)
        self.conv2 = Conv2d(2, 16, kernel_size=5)
        self.dropout = Dropout2d()
        self.fc1 = Linear(256, 64)
        self.fc2 = Linear(64, qnn.circuit.num_qubits)  # Dynamically set the output size
        self.qnn = TorchConnector(qnn)  # Apply torch connector, weights chosen
        # uniformly at random from interval [-1,1].
        self.fc3 = Linear(1, 1)  # 1-dimensional output from QNN

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Access the underlying QNN to determine input size
        qnn = self.qnn.neural_network  # Access the EstimatorQNN object
        num_qubits = qnn.circuit.num_qubits
        
        # Ensure the input tensor matches the expected size
        if x.shape[1] != num_qubits:
            x = x[:, :num_qubits]  # Slice to the appropriate number of qubits
        
        x = self.qnn(x)  # Apply QNN
        x = self.fc3(x)
        return cat((x, 1 - x), -1)


# Function to train and evaluate a model with a given encoding method
def train_and_evaluate(encoding_method):
    # Get a sample image for encoding
    sample_image, _ = next(iter(train_loader))
    sample_image = sample_image[0, 0].numpy().flatten()  # Flatten the image for encoding

    # Create a QNN with the specified encoding method
    qnn = create_qnn(encoding_method, sample_image)

    # Define the neural network model
    model = Net(qnn)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = NLLLoss()

    
    optimizer = COBYLA(maxiter=100, tol=1e-5)
    loss_func = NLLLoss()


    # Train the model
    epochs = 10
    loss_list = []
    model.train()

    for epoch in range(epochs):
        total_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            #optimizer.zero_grad(set_to_none=True) //so com o adam, meter no relatorio
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            #optimizer.step() //so com o adam, meter no relatorio
            total_loss.append(loss.item())
        loss_list.append(sum(total_loss) / len(total_loss))
        print(f"Encoding: {encoding_method}, Epoch [{epoch+1}/{epochs}] Loss: {loss_list[-1]:.4f}")

    # Evaluate the model
    model.eval()
    correct = 0
    total_loss = []
    with no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss = loss_func(output, target)
            total_loss.append(loss.item())

    accuracy = correct / len(test_loader) * 100
    avg_loss = sum(total_loss) / len(total_loss)
    print(f"Encoding: {encoding_method}, Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    
    return loss_list, accuracy

encoding_methods = ['ZZFeatureMap']
for method in encoding_methods:
    train_and_evaluate(method)
