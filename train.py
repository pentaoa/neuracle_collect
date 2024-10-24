import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define the classes and their corresponding folder names
classes = ['bonnet', 'cash_machine', 'coal', 'diamond', 'flower', 'headphones', 'hovercraft', 'hummingbird', 'lego', 'moss']
# classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Set the path to the folder containing the class subfolders with .npy data
data_path = "imagine_data"

# Function to load .npy data and transform it to torch tensors
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        for class_name in classes:
            class_path = os.path.join(self.root_dir, class_name)
            for i in range(80):
                file_name = f"{i}.npy"
                file_path = os.path.join(class_path, file_name)
                self.samples.append((file_path, classes.index(class_name)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        data = np.load(file_path)
        if self.transform:
            data = self.transform(data)
        return data, label

# Data transformation
class ToTensor(object):
    def __call__(self, data):
        return torch.tensor(data, dtype=torch.float32)

# Load the dataset
dataset = CustomDataset(data_path, transform=transforms.Compose([ToTensor()]))

# Split the dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create DataLoader for training and testing
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            # nn.Linear(64 * 156, 512),
            nn.Linear(64 * 125, 512),
            # nn.Linear(64 * 31, 512),  # Adjusted input size based on data shape (16, 125)
            nn.ReLU(),
            nn.Dropout(0.4),  # Dropout layer with 40% probability
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Create the model
model = SimpleCNN(num_classes=len(classes))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Function to calculate accuracy
def get_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# Training loop
num_epochs = 20
train_loss_list = []
train_accuracy_list = []
best_valid_loss = float('inf')
patience = 5
early_stopping_counter = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)  # No need to transpose input for Conv1d
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += get_accuracy(outputs, labels)

    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = total_accuracy / len(train_loader)
    train_loss_list.append(epoch_loss)
    train_accuracy_list.append(epoch_accuracy)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoh_accuracy:.4f}")


    # # Validation
    # model.eval()
    # valid_loss = 0.0
    # with torch.no_grad():
    #     for inputs, labels in test_loader:
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         valid_loss += loss.item()
    #
    # valid_loss /= len(test_loader)
    #
    # if valid_loss < best_valid_loss:
    #     best_valid_loss = valid_loss
    #     early_stopping_counter = 0
    # else:
    #     early_stopping_counter += 1
    #
    # if early_stopping_counter >= patience:
    #     print(f"Early stopping at epoch {epoch + 1}")
    #     break

# Evaluate the model on the test set
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)  # No need to transpose input for Conv1d
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.tolist())
        all_labels.extend(labels.tolist())

# Calculate the confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

# Plot training loss and accuracy
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracy_list) + 1), train_accuracy_list, label="Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()


# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)
test_accuracy = sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
print(f"Test Accuracy: {test_accuracy:.4f}")
