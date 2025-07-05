import os
import torch
from torch import cat, no_grad, manual_seed
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from torch.nn import Module, Linear, Conv2d, Dropout2d, NLLLoss, functional as F

# Create directory if it doesn't exist
if not os.path.exists('tutorial1'):
    os.makedirs('tutorial1')

# Custom dataset generation
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

        for i in range(8):
            if images[-1][i] == 0:
                images[-1][i] = np.random.uniform(0, np.pi / 16)
    return images, labels

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        labels = [0 if l == -1 else 1 for l in labels]
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Generate datasets
manual_seed(1122)
X_train_data, y_train_data = generate_dataset(200)
X_test_data, y_test_data = generate_dataset(100)

train_dataset = CustomDataset(X_train_data, y_train_data)
test_dataset = CustomDataset(X_test_data, y_test_data)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Initialize the Estimator
estimator = Estimator()

# Create QNN
def create_qnn():
    feature_map = ZZFeatureMap(2)
    ansatz = RealAmplitudes(2, reps=1)
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

qnn4 = create_qnn()

# Define Neural Network Model with QNN
class Net(Module):
    def __init__(self, qnn):
        super().__init__()
        self.fc1 = Linear(8, 64)
        self.fc2 = Linear(64, 2)
        self.qnn = TorchConnector(qnn)
        self.fc3 = Linear(1, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.qnn(x)
        x = self.fc3(x)
        return cat((x, 1 - x), -1)

# Instantiate model and training components
model4 = Net(qnn4)
optimizer = optim.Adam(model4.parameters(), lr=0.001)
loss_func = NLLLoss()

# Training
epochs = 10
loss_list = []
model4.train()

for epoch in range(epochs):
    total_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        output = model4(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
    loss_list.append(sum(total_loss) / len(total_loss))
    print("Training [{:.0f}%]\tLoss: {:.4f}".format(100.0 * (epoch + 1) / epochs, loss_list[-1]))

# Plot training loss
plt.plot(loss_list)
plt.title("Hybrid NN Training Convergence")
plt.xlabel("Training Iterations")
plt.ylabel("Neg. Log Likelihood Loss")
plt.savefig('tutorial1/loss_convergence.png')

# Save model
torch.save(model4.state_dict(), "model4.pt")

# Reload model with new QNN
qnn5 = create_qnn()
model5 = Net(qnn5)
model5.load_state_dict(torch.load("model4.pt"))

# Evaluate model
model5.eval()
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
            sum(total_loss) / len(total_loss), correct / len(test_loader) * 100
        )
    )

# Visualize predictions
n_samples_show = 6
count = 0
fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

with no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        if count == n_samples_show:
            break
        output = model5(data[0:1])
        if len(output.shape) == 1:
            output = output.reshape(1, *output.shape)
        pred = output.argmax(dim=1, keepdim=True)
        axes[count].imshow(data[0].numpy().reshape(1, 8), cmap="gray", aspect='auto')
        axes[count].set_xticks([])
        axes[count].set_yticks([])
        axes[count].set_title(f"Pred {pred.item()}")
        count += 1

fig.savefig('tutorial1/predicted_labels.png')
