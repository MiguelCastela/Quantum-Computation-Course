import os
import torch
from torch import cat, no_grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from torch.nn import Module, Conv2d, Linear
from torch.nn import functional as F
from torch.nn import NLLLoss

# Initialize Estimator
estimator = Estimator()

# Create data directory
if not os.path.exists('tutorial1'):
    os.makedirs('tutorial1')

# Fix seeds
np.random.seed(121)

batch_size = 1
n_train_samples = 100
n_test_samples = 500
# Load MNIST and filter only digits 0 and 1
def load_filtered_fashion_mnist(train=True, n_samples=100):
    dataset = datasets.FashionMNIST(
        root="./data", train=train, download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    selected_classes = [0, 2]
    idx = np.where(np.isin(dataset.targets, selected_classes))[0]
    idx0 = idx[dataset.targets[idx] == 0][:n_samples]
    idx2 = idx[dataset.targets[idx] == 2][:n_samples]
    idx = np.append(idx0, idx2)
    dataset.data = dataset.data[idx]
    dataset.targets = dataset.targets[idx]
    dataset.targets = torch.where(dataset.targets == 2, torch.tensor(1), dataset.targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

train_loader = load_filtered_fashion_mnist(train=True, n_samples=n_train_samples)
test_loader = load_filtered_fashion_mnist(train=False, n_samples=n_test_samples)

# Create QNN with TwoLocal ansatz
def create_qnn():
    feature_map = ZZFeatureMap(2)
    ansatz = TwoLocal(num_qubits=2, rotation_blocks='ry', entanglement_blocks='cz', reps=1)
    qc = QuantumCircuit(2)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
        estimator=estimator,
    )
    return qnn

qnn = create_qnn()

# Define hybrid model with 1 conv layer + 1 pooling + QNN layer
class HybridNet(Module):
    def __init__(self, qnn):
        super().__init__()
        self.conv1 = Conv2d(1, 4, kernel_size=5)  # output: (4, 24, 24)
        self.pool = torch.nn.MaxPool2d(2)         # output: (4, 12, 12)
        self.fc1 = Linear(4*12*12, 2)
        self.qnn = TorchConnector(qnn)
        self.fc2 = Linear(1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.qnn(x)
        x = self.fc2(x)
        return cat((x, 1 - x), dim=-1)

# Instantiate model, loss, optimizer
model = HybridNet(qnn)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = NLLLoss()

# Train loop
epochs = 10
loss_list = []

model.train()
for epoch in range(epochs):
    total_loss = []
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
    loss_list.append(sum(total_loss) / len(total_loss))
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss_list[-1]:.4f}")

# Plot loss convergence
plt.plot(loss_list)
plt.title("Hybrid NN Training without Evolutionary Operators")
plt.xlabel("Epoch")
plt.ylabel("Neg. Log Likelihood Loss")
plt.savefig('tutorial1/loss_convergence_no_evolution.png')

# Save model
torch.save(model.state_dict(), "model_no_evolution.pt")

# Load and evaluate
qnn_test = create_qnn()
model_test = HybridNet(qnn_test)
model_test.load_state_dict(torch.load("model_no_evolution.pt"))
model_test.eval()

correct = 0
total_loss = []
with no_grad():
    for data, target in test_loader:
        output = model_test(data)
        if len(output.shape) == 1:
            output = output.unsqueeze(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss = loss_func(output, target)
        total_loss.append(loss.item())

print(
    f"Test set performance:\n\tLoss: {sum(total_loss)/len(total_loss):.4f}\n\tAccuracy: {correct/len(test_loader)/batch_size*100:.1f}%"
)
