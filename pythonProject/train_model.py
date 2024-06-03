# train_model.py
import uuid
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os
from datetime import datetime
from alexnet import build_alexnet
from inception import build_inception_v3
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath):
    logging.debug(f"Loading data from: {filepath}")
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    data = np.nan_to_num(data)
    X = data[:, 1:].reshape(-1, 1, 28, 28).astype(np.float32)
    y = data[:, 0].astype(int)
    return X, y

def train_model(filepath, epochs, batch_size, validation_split, model_name, progress_window, stop_event):
    try:
        X, y = load_data(filepath)
        dataset = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long))
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

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
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        best_val_accuracy = 0

        for epoch in range(epochs):
            if stop_event.is_set():
                logging.info("Training is stopped")
                break

            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                if stop_event.is_set():
                    logging.info("Training is stopped")
                    break

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
                    if stop_event.is_set():
                        logging.info("Training is stopped")
                        break

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            val_accuracy = 100 * correct / total
            scheduler.step()

            if progress_window:
                progress_window.add_data(epoch + 1, train_loss, val_accuracy)

            logging.info(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Val Loss: {val_loss/len(val_loader)}, Accuracy: {val_accuracy}%')

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = model.state_dict()

        if progress_window:
            progress_window.stop_timer()

        model_id = str(uuid.uuid4())
        model_save_path = os.path.join(
            'C:\\Users\\lojan\\OneDrive\\Documents\\Uni\\COMPSYS305\\project-python-group-21\\pythonProject\\Models',
            f'{model_name.lower()}_model_{model_id}.pth')
        torch.save({
            'model_state_dict': best_model_state,
            'model_name': model_name,
            'epochs': epochs,
            'batch_size': batch_size,
            'validation_split': validation_split,
            'image_paths': filepath
        }, model_save_path)
        logging.info(f'Model saved as {model_save_path}')
    except Exception as e:
        logging.error(f"Error during training: {e}")
