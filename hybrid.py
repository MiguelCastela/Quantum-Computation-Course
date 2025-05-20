import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transform: Normalize to [0, 1] and keep size 28x28
transform = transforms.Compose([
    transforms.ToTensor()
])

transform = transforms.Compose([transforms.ToTensor()])
test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_data, batch_size=4, shuffle=True)

# Optional: for reproducibility
algorithm_globals.random_seed = 42


class ImprovedHybridQNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Feature extractor with convolution and pooling
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),   # -> (batch, 8, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                     # -> (batch, 8, 14, 14)

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),  # -> (batch, 16, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                     # -> (batch, 16, 7, 7)

            nn.Flatten(),                                           # -> (batch, 16*7*7)
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),                                       # -> 4 values (input to QNN)
        )

        # Define quantum feature map and ansatz
        self.num_qubits = 4
        feature_map = ZZFeatureMap(self.num_qubits, reps=1)
        var_form = TwoLocal(self.num_qubits, reps=1, rotation_blocks="ry", entanglement_blocks="cz")
        quantum_circuit = feature_map.compose(var_form)

        # Define EstimatorQNN and TorchConnector
        qnn = EstimatorQNN(
            circuit=quantum_circuit,
            input_params=feature_map.parameters,
            weight_params=var_form.parameters,
        )
        self.q_layer = TorchConnector(qnn)

        # Output layer for 10-class classification
        self.output_layer = nn.Sequential(
            nn.Linear(1, 10),             # QNN returns scalar output per example
            nn.LogSoftmax(dim=1)          # Log-softmax for NLLLoss
        )

    def forward(self, x):
        x = self.feature_extractor(x)      # shape: (batch, 4)
        x = self.q_layer(x)                # shape: (batch, 1)
        x = self.output_layer(x)           # shape: (batch, 10)
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedHybridQNN().to(device)
model.eval()  # inference mode

# Test one batch
with torch.no_grad():
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)

    print("Outputs:\n", outputs)
    print("Predicted classes:", torch.argmax(outputs, dim=1).tolist())
    print("True classes:", labels.tolist())







