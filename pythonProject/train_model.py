import uuid
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os
from alexnet import build_alexnet
from inception import build_inception_v3
from resnet import ResNet18
from vgg import VGG16
from signsysmodel import SignSysModel
import logging
from torch.cuda.amp import GradScaler, autocast

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath):
    logging.debug(f"Loading data from: {filepath}")
    # Load data from CSV file
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    data = np.nan_to_num(data)
    X = data[:, 1:].reshape(-1, 1, 28, 28).astype(np.float32)  # Reshape and convert to float32
    y = data[:, 0].astype(int)  # Extract labels and convert to int
    return X, y

# This is for training model and it takes the inputs from sliders
def train_model(filepath, epochs, batch_size, validation_split, model_name, progress_window, stop_event):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Determine if GPU is available
        X, y = load_data(filepath)  # Load the dataset
        dataset = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long))  # Create a TensorDataset
        val_size = int(len(dataset) * validation_split)  # Calculate validation set size
        train_size = len(dataset) - val_size  # Calculate training set size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])  # Split dataset

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Create DataLoader for training set
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # Create DataLoader for validation set

        # Initialize the model
        if model_name == "AlexNet":
            model = build_alexnet(num_classes=36)
        elif model_name == "InceptionV3":
            model = build_inception_v3(num_classes=36)
        elif model_name == "Sign-SYS Model":
            vgg = VGG16(num_classes=36)
            resnet = ResNet18(num_classes=36)
            model = SignSysModel(vgg, resnet, num_classes=36)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        model.to(device)  # Move model to device (GPU or CPU)
        criterion = nn.CrossEntropyLoss()  # Loss function
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Learning rate scheduler

        scaler = GradScaler()  # Gradient scaler for mixed precision training
        best_val_accuracy = 0  # Initialize best validation accuracy

        for epoch in range(epochs):
            if stop_event.is_set():
                logging.info("Training is stopped")
                break

            model.train()  # Set model to training mode
            running_loss = 0.0
            for inputs, labels in train_loader:
                if stop_event.is_set():
                    logging.info("Training is stopped")
                    break

                inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to device

                optimizer.zero_grad()  # Zero the parameter gradients

                # Mixed precision training for specific models
                if model_name in ["Sign-SYS Model"]:
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()  # Accumulate running loss

            model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    if stop_event.is_set():
                        logging.info("Training is stopped")
                        break

                    inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to device

                    # Mixed precision evaluation for specific models
                    if model_name in ["Sign-SYS Model"]:
                        with autocast():
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    val_loss += loss.item()  # Accumulate validation loss
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)  # Total number of labels
                    correct += (predicted == labels).sum().item()  # Count correct predictions

            train_loss = running_loss / len(train_loader)  # Calculate average training loss
            val_accuracy = 100 * correct / total  # Calculate validation accuracy
            scheduler.step()  # Step the scheduler

            if progress_window:
                progress_window.add_data(epoch + 1, train_loss, val_accuracy)

            logging.info(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Val Loss: {val_loss/len(val_loader)}, Accuracy: {val_accuracy}%')

            # Save the best model based on validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = model.state_dict()

        if progress_window:
            progress_window.stop_timer()

        model_save_dir = 'models'

        os.makedirs(model_save_dir, exist_ok=True)  # Create directory if it doesn't exist

        # Save the model state dictionary and metadata
        model_save_path = os.path.join(model_save_dir, f'{model_name.lower().replace(" ", "_")}_{best_val_accuracy:.2f}.pth')
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
