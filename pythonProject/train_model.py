import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from alexnet import build_alexnet

def load_data(filepath):
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    data = np.nan_to_num(data)  # Replace nan values with zero
    X = data[:, 1:].reshape(-1, 1, 28, 28).astype(np.float32)
    y = data[:, 0].astype(int)
    return X, y

def train_alexnet_model(filepath, epochs=10, batch_size=32, validation_split=0.2):
    X, y = load_data(filepath)
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long))
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = build_alexnet(num_classes=36)
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

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}, Accuracy: {100 * correct / total}%')

    torch.save(model.state_dict(), 'alexnet_model.pth')
