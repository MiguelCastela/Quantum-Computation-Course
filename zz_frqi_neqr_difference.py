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
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from torch.nn import Module
from torch.nn import Conv2d, Dropout2d, Linear
from torch.nn import NLLLoss
from torch.nn import functional as F
from torch import optim
from qiskit.circuit import QuantumRegister
from qiskit.circuit import Parameter





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


def create_frqi_qnn(image_size=28):
    # Calculate the number of qubits needed
    num_qubits = int(np.log2(image_size * image_size)) + 1  # +1 for grayscale qubit

    # Create a quantum circuit
    qreg = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qreg)

    # Add Hadamard gates to initialize position qubits
    for i in range(num_qubits - 1):  # Exclude the grayscale qubit
        qc.h(qreg[i])

    # Create parameterized input features
    input_params = [Parameter(f"x_{i}") for i in range(image_size * image_size)]
    for i, param in enumerate(input_params):
        bin_index = format(i, f'0{num_qubits - 1}b')  # Binary index for pixel position
        for j, bit in enumerate(bin_index):
            if bit == '1':
                qc.x(qreg[j])
        qc.cry(param, qreg[num_qubits - 2], qreg[num_qubits - 1])  # Grayscale rotation
        for j, bit in enumerate(bin_index):
            if bit == '1':
                qc.x(qreg[j])

    # Add an ansatz for trainable weights
    ansatz = RealAmplitudes(num_qubits, reps=1)
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

def create_feqr_qnn(image_size=28):
    """
    Create a QNN using the Flexible Encoding Quantum Representation (FEQR).
    This encoding method allows for flexible encoding of grayscale pixel values and positions.

    Args:
        image_size (int): The size of the image (e.g., 28 for a 28x28 image).

    Returns:
        EstimatorQNN: A quantum neural network with FEQR encoding.
    """
    # Calculate the number of qubits needed
    num_qubits = int(np.log2(image_size * image_size)) + 8  # +8 for 8-bit grayscale encoding

    # Create a quantum circuit
    qreg = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qreg)

    # Add Hadamard gates to initialize position qubits
    for i in range(num_qubits - 8):  # Exclude the 8 grayscale qubits
        qc.h(qreg[i])

    # Create parameterized input features
    input_params = [Parameter(f"x_{i}") for i in range(image_size * image_size)]
    for i, param in enumerate(input_params):
        bin_index = format(i, f'0{num_qubits - 8}b')  # Binary index for pixel position
        for j, bit in enumerate(bin_index):
            if bit == '1':
                qc.x(qreg[j])

        # Encode grayscale value using 8 qubits
        for k in range(8):  # 8-bit grayscale encoding
            qc.cry(param / 255.0 * (2 * np.pi), qreg[num_qubits - 8 + k], qreg[k])

        for j, bit in enumerate(bin_index):
            if bit == '1':
                qc.x(qreg[j])

    # Add an ansatz for trainable weights
    ansatz = RealAmplitudes(num_qubits, reps=1)
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

# Function to create a Quantum Neural Network (QNN)
def create_qnn():
    feature_map = ZZFeatureMap(2)
    ansatz = RealAmplitudes(2, reps=1)
    qc = QuantumCircuit(2)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
        estimator=estimator,
    )
    return qnn

# Create a QNN
qnn4 = create_feqr_qnn(image_size=28)


# Define a neural network model with a QNN layer
class Net(Module):
    def __init__(self, qnn, input_size):
        super().__init__()
        self.conv1 = Conv2d(1, 2, kernel_size=5)
        self.conv2 = Conv2d(2, 16, kernel_size=5)
        self.dropout = Dropout2d()
        self.fc1 = Linear(256, 64)
        self.fc2 = Linear(64, input_size)  # 2-dimensional input to QNN
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
        x = self.qnn(x)  # apply QNN
        x = self.fc3(x)
        return cat((x, 1 - x), -1)

# Instantiate the model
model4 = Net(qnn4, input_size=784) # 2 qubits for 28x28 image

# Define model, optimizer, and loss function
optimizer = optim.Adam(model4.parameters(), lr=0.001)
loss_func = NLLLoss()

# Start training
epochs = 10  # Set number of epochs
loss_list = []  # Store loss history
model4.train()  # Set model to training mode

for epoch in range(epochs):
    total_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)  # Initialize gradient
        output = model4(data)  # Forward pass
        loss = loss_func(output, target)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize weights
        total_loss.append(loss.item())  # Store loss
    loss_list.append(sum(total_loss) / len(total_loss))
    print("Training [{:.0f}%]\tLoss: {:.4f}".format(100.0 * (epoch + 1) / epochs, loss_list[-1]))

# Plot loss convergence
plt.plot(loss_list)
plt.title("Hybrid NN Training Convergence")
plt.xlabel("Training Iterations")
plt.ylabel("Neg. Log Likelihood Loss")
plt.savefig('tutorial1/loss_convergence.png')

# Save the trained model
torch.save(model4.state_dict(), "model4.pt")

# Create a new QNN and load the trained model's state
qnn5 = create_qnn()
model5 = Net(qnn5)
model5.load_state_dict(torch.load("model4.pt"))

# Evaluate the model on the test dataset
model5.eval()  # set model to evaluation mode
with no_grad():
    correct = 0
    total_loss = []
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model5(data)
        if len(output.shape) == 1:
            output = output.reshape(1, *output.shape)

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss = loss_func(output, target)
        total_loss.append(loss.item())

    print(
        "Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%".format(
            sum(total_loss) / len(total_loss), correct / len(test_loader) / batch_size * 100
        )
    )

# Plot predicted labels
n_samples_show = 6
count = 0
fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

model5.eval()
with no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        if count == n_samples_show:
            break
        output = model5(data[0:1])
        if len(output.shape) == 1:
            output = output.reshape(1, *output.shape)

        pred = output.argmax(dim=1, keepdim=True)

        axes[count].imshow(data[0].numpy().squeeze(), cmap="gray")

        axes[count].set_xticks([])
        axes[count].set_yticks([])
        axes[count].set_title("Predicted {}".format(pred.item()))

        count += 1

fig.savefig('tutorial1/predicted_labels.png')
