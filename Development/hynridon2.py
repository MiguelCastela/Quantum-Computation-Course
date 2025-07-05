import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.connectors import TorchConnector

# --- Dataset generation ---
def generate_dataset(num_images):
    images = []
    labels = []
    hor_array = np.zeros((6, 8))
    ver_array = np.zeros((4, 8))

    j = 0
    for i in range(0, 7):
        if i != 3:
            hor_array[j][i] = np.pi / 2
            hor_array[j][i + 1] = np.pi / 2
            j += 1

    j = 0
    for i in range(0, 4):
        ver_array[j][i] = np.pi / 2
        ver_array[j][i + 4] = np.pi / 2
        j += 1

    for n in range(num_images):
        rng = np.random.randint(0, 2)
        if rng == 0:
            labels.append(-1)
            random_image = np.random.randint(0, 6)
            images.append(np.array(hor_array[random_image]))
        else:
            labels.append(1)
            random_image = np.random.randint(0, 4)
            images.append(np.array(ver_array[random_image]))

        # Add noise
        for i in range(8):
            if images[-1][i] == 0:
                images[-1][i] = np.random.uniform(0, np.pi / 16)
    return images, labels

# --- Dataset class ---
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        # Map labels from -1,1 to 0,1
        labels = [0 if l == -1 else 1 for l in labels]
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# --- Define QNN ---
def create_qnn():
    feature_map = ZZFeatureMap(2)
    ansatz = TwoLocal(2, ['ry', 'rz'], 'cz', reps=1)
    qc = QuantumCircuit(2)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
        estimator=Estimator(),
    )
    return qnn

# --- Define model ---
class Net(nn.Module):
    def __init__(self, qnn):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(16, 2)  # Output size matches QNN input size
        self.qnn = TorchConnector(qnn)
        self.fc3 = nn.Linear(1, 1)   # QNN output dimension is 1

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.qnn(x)
        x = self.fc3(x)
        # Output logits for 2 classes, shape (batch, 2)
        # Here we create two logits: x and 1-x for binary classification
        return torch.cat((x, 1 - x), dim=-1)

# --- Training and testing functions ---
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Train Epoch: {epoch} \tLoss: {avg_loss:.4f}')

def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    accuracy = correct / len(test_loader.dataset)
    print(f'Test set: Accuracy: {accuracy*100:.2f}%')

# --- Main ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate data
    train_images, train_labels = generate_dataset(100)
    test_images, test_labels = generate_dataset(50)

    train_dataset = CustomDataset(train_images, train_labels)
    test_dataset = CustomDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    qnn = create_qnn()
    model = Net(qnn).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 10
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

if __name__ == '__main__':
    main()
