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

# Initialize the Estimator
estimator = Estimator()

# Create directory if it doesn't exist
if not os.path.exists('results/tutorial1'):
    os.makedirs('results/tutorial1')

# Train Dataset
# -------------
def load_filtered_fashion_mnist(train=True, n_samples=100):
    dataset = datasets.FashionMNIST(
        root="./data", train=train, download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    # Choose classes 0 and 2 for binary classification
    selected_classes = [0, 2]
    idx = np.where(np.isin(dataset.targets, selected_classes))[0]
    
    # Take first n_samples from each class
    idx0 = idx[dataset.targets[idx] == 0][:n_samples]
    idx2 = idx[dataset.targets[idx] == 2][:n_samples]
    idx = np.append(idx0, idx2)
    
    dataset.data = dataset.data[idx]
    dataset.targets = dataset.targets[idx]
    
    # Remap targets: 0 -> 0, 2 -> 1 for binary classification
    dataset.targets = torch.where(dataset.targets == 2, torch.tensor(1), dataset.targets)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader



# Set train shuffle seed (for reproducibility)
manual_seed(1111)

n_train_samples = 100
n_test_samples = 50
batch_size = 1

train_loader = load_filtered_fashion_mnist(train=True, n_samples=n_train_samples)
test_loader = load_filtered_fashion_mnist(train=False, n_samples=n_test_samples)





# Define torch dataloader with filtered data

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

fig.savefig('results/tutorial1/train_samples.png')

# Test Dataset
# -------------

# Set test shuffle seed (for reproducibility)
# manual_seed(5)

n_samples = 50

# Use pre-defined torchvision function to load MNIST test data



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
qnn4 = create_qnn()

# Define a neural network model with a QNN layer
class Net(Module):
    def __init__(self, qnn):
        super().__init__()
        self.conv1 = Conv2d(1, 2, kernel_size=5)
        self.conv2 = Conv2d(2, 16, kernel_size=5)
        self.dropout = Dropout2d()
        self.fc1 = Linear(256, 64)
        self.fc2 = Linear(64, 2)  # 2-dimensional input to QNN
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
model4 = Net(qnn4)

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
plt.savefig('results/tutorial1/loss_convergence.png')

# Save the trained model
torch.save(model4.state_dict(), "models/model4.pt")

# Create a new QNN and load the trained model's state
qnn5 = create_qnn()
model5 = Net(qnn5)
model5.load_state_dict(torch.load("models/model4.pt"))

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

fig.savefig('results/tutorial1/predicted_labels.png')
