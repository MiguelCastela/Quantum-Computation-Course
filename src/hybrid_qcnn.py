import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skimage.transform import resize

import torch
import torch.nn as nn
from torch.optim import Adam

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN

from torchvision import datasets, transforms

# Set random seed
torch.manual_seed(42)

# -------------------------
# 1. Load MNIST (0 and 1)
# -------------------------
def load_filtered_mnist(train=True, n_zeros=25, n_ones=25):
    dataset = datasets.MNIST(
        root="./data", train=train, download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    targets = dataset.targets.numpy()
    data = dataset.data.numpy()

    idx0 = np.where(targets == 0)[0][:n_zeros]
    idx1 = np.where(targets == 1)[0][:n_ones]
    idx = np.append(idx0, idx1)

    data = data[idx]
    labels = targets[idx]
    images = data / 255.0

    resized_images = np.array([resize(img, (2, 4), mode='constant').flatten() * np.pi for img in images])
    labels = np.where(labels == 0, -1, 1)

    return resized_images, labels

train_images, train_labels = load_filtered_mnist(train=True, n_zeros=25, n_ones=25)
test_images, test_labels = load_filtered_mnist(train=False, n_zeros=50, n_ones=50)

# -------------------------
# 2. Define Quantum Circuit
# -------------------------
num_qubits = 8
feature_map = ZZFeatureMap(num_qubits)
ansatz = TwoLocal(num_qubits, rotation_blocks="ry", entanglement_blocks="cz", reps=1)

qc = QuantumCircuit(num_qubits)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)

# -------------------------
# 3. Define Estimator QNN
# -------------------------
qnn = EstimatorQNN(
    circuit=qc,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
)

model = TorchConnector(qnn)

# -------------------------
# 4. Build Hybrid Model
# -------------------------
class HybridClassifier(nn.Module):
    def __init__(self, qnn):
        super().__init__()
        self.qnn = TorchConnector(qnn)
    
    def forward(self, x):
        return self.qnn(x)

hybrid_model = HybridClassifier(qnn)

# -------------------------
# 5. Train the Model
# -------------------------
def train(model, X, y, epochs=10, lr=0.1):
    optimizer = Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = loss_func(output.flatten(), y_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1:02d} | Loss: {loss.item():.4f}")

train(hybrid_model, train_images, train_labels)

# -------------------------
# 6. Evaluate the Model
# -------------------------
def evaluate(model, X, y_true):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        predictions = model(X_tensor).flatten().numpy()
        y_pred = np.where(predictions >= 0, 1, -1)
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.2%}")
    return y_pred

y_pred = evaluate(hybrid_model, test_images, test_labels)

# -------------------------
# 7. Plot Results
# -------------------------
fig, ax = plt.subplots(2, 2, figsize=(6, 6))
for i in range(4):
    original_img = test_images[i].reshape((2, 4))
    ax[i // 2, i % 2].imshow(original_img, cmap='gray')
    label = "Digit 0" if y_pred[i] == -1 else "Digit 1"
    ax[i // 2, i % 2].set_title(f"QCNN predicts: {label}")
    ax[i // 2, i % 2].axis("off")
plt.tight_layout()
plt.show()
