# train_model.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from alexnet import build_alexnet
from inception import build_inception_v3

def load_data(filepath):
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    data = np.nan_to_num(data)  # Replace nan values with zero
    X = data[:, 1:].reshape(-1, 1, 28, 28).astype(np.float32)
    y = data[:, 0].astype(int)
    return X, y

def train_model(filepath, epochs, batch_size, validation_split, model_name, progress_window):
    X, y = load_data(filepath)
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long))
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if model_name == "AlexNet":
        model = build_alexnet(num_classes=36)
    elif model_name == "InceptionV3":
        model = build_inception_v3(num_classes=36)
    else:
        raise ValueError("Unknown model name")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        val_accuracy = 100 * correct / total

        if progress_window:
            progress_window.add_data(epoch + 1, train_loss, val_accuracy)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Val Loss: {val_loss/len(val_loader)}, Accuracy: {val_accuracy}%')

    if progress_window:
        progress_window.stop_timer()
    torch.save(model.state_dict(), f'{model_name.lower()}_model.pth')
