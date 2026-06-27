import os
import torch
from torch import cat, no_grad, manual_seed
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import ZFeatureMap, TwoLocal
from qiskit import QuantumCircuit
from qiskit.primitives import BackendEstimatorV2
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from torch.nn import Module, Conv2d, Linear
from torch.nn import functional as F
from torch.nn import NLLLoss
from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import EstimatorV2
from qiskit.compiler import transpile


from qiskit.transpiler import generate_preset_pass_manager



#load_dotenv()  # Load variables from .env file
qiskit_token = os.getenv("QISKIT_TOKEN")

service = QiskitRuntimeService()
print(qiskit_token)
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=2)
print(f"Using backend: {backend.name}")
estimator = EstimatorV2(backend)



# Initialize Qiskit Runtime Service and Estimator
#backend = service.least_busy(operational=True, simulator=True)
#estimator = Estimator()  # Use a simulator for testing

# Create data directory
if not os.path.exists('results/tutorial1'):
    os.makedirs('results/tutorial1')

# Fix seeds
manual_seed(12)
np.random.seed(12)

batch_size = 1
n_train_samples = 20
n_test_samples = 10

# Load MNIST and filter only digits 0 and 1
def load_filtered_mnist(train=True, n_samples=100):
    dataset = datasets.MNIST(
        root="./data", train=train, download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    idx0 = np.where(dataset.targets == 0)[0][:n_samples]
    idx1 = np.where(dataset.targets == 1)[0][:n_samples]
    idx = np.append(idx0, idx1)
    dataset.data = dataset.data[idx]
    dataset.targets = dataset.targets[idx]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

train_loader = load_filtered_mnist(train=True, n_samples=n_train_samples)
test_loader = load_filtered_mnist(train=False, n_samples=n_test_samples)
pass_manager = generate_preset_pass_manager(optimization_level=2, backend=backend)
# Create QNN with TwoLocal ansatz
def create_qnn():
    feature_map = ZFeatureMap(2)
    ansatz = TwoLocal(num_qubits=2, rotation_blocks='ry', entanglement_blocks='cz', reps=1)
    qc = QuantumCircuit(2)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    qc = transpile(qc, backend)




    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
        estimator=estimator,
        pass_manager=pass_manager
    )
    return qnn

qnn = create_qnn()

# Define hybrid model with 1 conv layer + 1 pooling + QNN layer
class HybridNet(Module):
    def __init__(self, qnn):
        super().__init__()
        self.conv1 = Conv2d(1, 2, kernel_size=3)  # output size: (4, 24, 24)
        self.pool = torch.nn.MaxPool2d(2)         # output size: (4, 12, 12)
        self.fc1 = Linear(2 * 13 * 13, 2)              # Flatten to 4*12*12=576, then Linear to 2 features for QNN
        self.qnn = TorchConnector(qnn)              # QNN layer
                  # QNN output to single output

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.qnn(x)
        # Output 2 classes as prob distribution
        return cat((x, 1 - x), dim=-1)



# Instantiate model, loss, optimizer
model = HybridNet(qnn)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = NLLLoss()

# Evolutionary operators for parameters (crossover and mutation)
def crossover(param1, param2):
    flat1 = param1.flatten()
    flat2 = param2.flatten()
    length = flat1.size(0)
    point = torch.randint(1, length, (1,)).item()  # crossover point
    new1 = torch.cat([flat1[:point], flat2[point:]])
    new2 = torch.cat([flat2[:point], flat1[point:]])
    # Both new1 and new2 have length == length
    return new1.view(param1.shape), new2.view(param2.shape)


def mutation(param, mutation_rate=0.05, mutation_strength=0.1):
    # Randomly mutate some elements of the tensor
    mask = (torch.rand_like(param) < mutation_rate).float()
    noise = torch.randn_like(param) * mutation_strength
    param.data.add_(mask * noise)
    
def apply_evolution(model):
    params = list(model.parameters())
    n = len(params)
    for i in range(0, n - 1, 2):
        if params[i].shape == params[i + 1].shape:
            new1, new2 = crossover(params[i], params[i + 1])
            params[i].data.copy_(new1)
            params[i + 1].data.copy_(new2)
        else:
            # Skip crossover if shapes don't match
            pass
    for i in range(n):
        mutation(params[i])

# Train loop
epochs = 3
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

    # Apply evolutionary step (crossover + mutation) on model parameters
    apply_evolution(model)

# Plot loss convergence
plt.plot(loss_list)
plt.title("Hybrid NN Training with Evolutionary Operators")
plt.xlabel("Epoch")
plt.ylabel("Neg. Log Likelihood Loss")
plt.savefig('results/tutorial1/loss_convergence_evolution.png')

# Save model
torch.save(model.state_dict(), "models/model_evolution.pt")

# Load and evaluate
qnn_test = create_qnn()
model_test = HybridNet(qnn_test)
model_test.load_state_dict(torch.load("models/model_evolution.pt"))
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

plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), loss_list, marker='o')
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Negative Log Likelihood Loss")
plt.grid(True)
plt.savefig("results/tutorial1/training_loss_per_epoch.png")
plt.show()
