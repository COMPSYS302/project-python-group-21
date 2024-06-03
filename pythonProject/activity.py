import pandas as pd
import numpy as np
import time
import torch
import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QMutexLocker
import logging
import os

from train import TrainingWindow  # Ensure this import is correct
from styles import ActivityStyles
from alexnet import build_alexnet
from inception import build_inception_v3

activitystyles = ActivityStyles()

# Define a constant for the number of images to load per batch
IMAGES_BATCH_SIZE = 100

# Define a dictionary to remap the labels
label_remap = {
    26: 35,
    27: 26,
    28: 33,
    29: 32,
    30: 27,
    31: 34,
    32: 30,
    33: 29,
    34: 28,
    35: 31
}

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# Created 'DataLoaderThread' class which handles the data loading and
# image conversion in a background thread.
class DataLoaderThread(QThread):
    # Signal to indicate the progress of data loading
    progress = pyqtSignal(int)
    # Signal to indicate that the data loading is complete and to send the loaded images
    data_loaded = pyqtSignal(list)
    # Signal to indicate the estimated time remaining
    time_remaining = pyqtSignal(str)

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath  # Path to the CSV file to be loaded
        self.start_time = None
        self.mutex = QMutex()
        self.stopped = False

    def run(self):
        # Record the start time
        self.start_time = time.time()

        try:
            # Read the CSV file into a DataFrame
            data = pd.read_csv(self.filepath)
        except Exception as e:
            logging.error(f"Error reading CSV file: {e}")
            return

        images = []  # List to hold the QPixmap images along with their labels
        progress_num = len(data)  # Total number of rows in the DataFrame

        # Loop through each row in the DataFrame
        for i in range(progress_num):
            with QMutexLocker(self.mutex):
                if self.stopped:
                    break

            try:
                label = data.iloc[i, 0]  # Extract the label
                label = label_remap.get(label, label)  # Remap the label using the dictionary

                pixels = data.iloc[i, 1:].values  # Extract pixel values from the row
                # Convert pixel values to a NumPy array and reshape to a 28x28 image
                image_array = np.array(pixels, dtype=np.uint8).reshape((28, 28))
                # Create a QImage from the NumPy array
                qimage = QImage(image_array.data, 28, 28, QImage.Format_Grayscale8)
                # Convert QImage to QPixmap and add to the list
                images.append((label, QPixmap.fromImage(qimage), image_array))
                # Emit the progress signal with the percentage completed
                self.progress.emit(int((i + 1) / progress_num * 100))

                # Calculate elapsed time and estimate remaining time
                elapsed_time = time.time() - self.start_time
                if i > 0:  # Avoid division by zero
                    remaining_time = elapsed_time / (i + 1) * (progress_num - i - 1)
                    minutes, seconds = divmod(remaining_time, 60)
                    self.time_remaining.emit(f"{int(minutes)} min {int(seconds)} sec left")
            except Exception as e:
                logging.error(f"Error processing row {i}: {e}")
                continue

        # Emit the data_loaded signal with the list of QPixmap images if not stopped
        if not self.stopped:
            self.data_loaded.emit(images)

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True


class ImageDisplayWindow(qtw.QWidget):
    def __init__(self, images, model=None, model_name=None):
        super().__init__()
        self.setWindowTitle("Training Images")
        self.setGeometry(100, 100, 800, 600)

        self.images = images
        self.model = model
        self.model_name = model_name

        layout = qtw.QVBoxLayout()
        self.search_bar = qtw.QLineEdit(self)
        self.search_bar.setPlaceholderText("Search by label")
        self.search_bar.textChanged.connect(self.filterImages)
        self.search_bar.setStyleSheet(activitystyles.line_edit_style)

        layout.addWidget(self.search_bar)

        scroll_area = qtw.QScrollArea()
        scroll_widget = qtw.QWidget()
        self.image_layout = qtw.QGridLayout()

        scroll_widget.setLayout(self.image_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)

        for i, (label, pixmap, image_array) in enumerate(images):
            if self.model:
                label_widget = ClickableLabel(label, pixmap, image_array, self.model, self.model_name)
            else:
                label_widget = qtw.QLabel()
                label_widget.setPixmap(pixmap)
            self.image_layout.addWidget(label_widget, i // 10, i % 10)

        layout.addWidget(scroll_area)
        self.setLayout(layout)
        self.show()

    def filterImages(self):
        query = self.search_bar.text()
        filtered_images = []

        if query.isdigit():
            label = int(query)
            if 0 <= label <= 35:
                filtered_images = [img for img in self.images if img[0] == label]
        else:
            if query:
                filtered_images = []
                qtw.QMessageBox.warning(self, "Error", "Please enter a valid digit between 0 and 35.")
            else:
                filtered_images = self.images

        self.updateImageDisplay(filtered_images)

    def updateImageDisplay(self, images):
        for i in reversed(range(self.image_layout.count())):
            self.image_layout.itemAt(i).widget().setParent(None)

        for i, (label, pixmap, image_array) in enumerate(images):
            if self.model:
                label_widget = ClickableLabel(label, pixmap, image_array, self.model, self.model_name)
            else:
                label_widget = qtw.QLabel()
                label_widget.setPixmap(pixmap)
            self.image_layout.addWidget(label_widget, i // 10, i % 10)


class ClickableLabel(qtw.QLabel):
    def __init__(self, label, pixmap, image_array, model, model_name):
        super().__init__()
        self.label = label
        self.setPixmap(pixmap)
        self.image_array = image_array
        self.model = model
        self.model_name = model_name

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.predict_and_show()

    def predict_and_show(self):
        try:
            self.model.eval()
            with torch.no_grad():
                image_tensor = torch.tensor(self.image_array).unsqueeze(0).unsqueeze(0).float()
                output = self.model(image_tensor)
                _, predicted = torch.max(output, 1)
                predicted_label = predicted.item()

            qtw.QMessageBox.information(self, "Prediction", f"Predicted Label: {predicted_label}")

        except Exception as e:
            qtw.QMessageBox.critical(self, "Error", f"Failed to predict: {str(e)}")


class ActivityOptionsWindow(qtw.QWidget):

    def returnToHome(self):
        response = qtw.QMessageBox.question(self, 'Return to Home',
                                            'Returning will clear loaded data. Do you want to continue?',
                                            qtw.QMessageBox.Yes | qtw.QMessageBox.No, qtw.QMessageBox.No)
        if response == qtw.QMessageBox.Yes:
            self.prevWindow.show()
            self.close()

    def openTrainingWindow(self):
        if not self.images:
            qtw.QMessageBox.warning(self, "Warning", "No data loaded. Please load data first.")
        else:
            self.training_window = TrainingWindow(self, self.file_path)
            self.training_window.show()
            self.hide()

    def openTestModelWindow(self):
        fname, _ = qtw.QFileDialog.getOpenFileName(self, "Open Model File",
                                                   "C:\\Users\\Harsh\\OneDrive\\Documents\\Android studip\\project-python-group-21\\pythonProject\\Models",
                                                   "Model Files (*.pth);;All Files (*)")
        if fname:
            self.model_path = fname
            self.loadAndTestModel(self.model_path)

    def loadAndTestModel(self, model_path):
        try:
            model_info = torch.load(model_path)
            model_name = model_info['model_name']
            model_state_dict = model_info['model_state_dict']

            if model_name == "AlexNet":
                model = build_alexnet(num_classes=36)
            elif model_name == "InceptionV3":
                model = build_inception_v3(num_classes=36)
            else:
                raise ValueError("Unknown model name")

            model.load_state_dict(model_state_dict)
            model.eval()
            qtw.QMessageBox.information(self, "Success", f"Model {model_name} loaded successfully for testing.")

            # Use the CSV file path selected during the load data operation
            self.displayTrainingImages(self.file_path, model, model_name)

        except Exception as e:
            qtw.QMessageBox.critical(self, "Error", f"Failed to load the model: {str(e)}")

    def displayTrainingImages(self, filepath, model=None, model_name=None):
        try:
            data = pd.read_csv(filepath)
            images = []

            for i in range(len(data)):
                label = data.iloc[i, 0]
                label = label_remap.get(label, label)
                pixels = data.iloc[i, 1:].values
                image_array = np.array(pixels, dtype=np.uint8).reshape((28, 28))
                qimage = QImage(image_array.data, 28, 28, QImage.Format_Grayscale8)
                images.append((label, QPixmap.fromImage(qimage), image_array))

            self.image_display_window = ImageDisplayWindow(images, model, model_name)
            self.image_display_window.show()
        except Exception as e:
            qtw.QMessageBox.critical(self, "Error", f"Failed to load images: {str(e)}")

    # Method to load file
    def loadFile(self):
        fname, _ = qtw.QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv);;All Files (*)")
        if fname:
            self.file_path = fname
            self.loading_thread = DataLoaderThread(fname)
            self.loading_thread.progress.connect(self.updateProgressBar)
            self.loading_thread.data_loaded.connect(self.onDataLoaded)
            self.loading_thread.time_remaining.connect(self.updateTimeRemaining)
            self.loading_thread.start()
            self.loadingProgressBar.setRange(0, 100)
            self.loadingProgressBar.setAlignment(Qt.AlignCenter)  # Center align the text
            self.loadingProgressBar.show()
            self.stopButton.show()

    # Method to update the progress bar
    def updateProgressBar(self, value):
        self.loadingProgressBar.setValue(value)

    # Method to update the remaining time display
    def updateTimeRemaining(self, time_left):
        self.loadingProgressBar.setFormat(f"%p% - {time_left}")

    # Method to handle data once loaded
    def onDataLoaded(self, images):
        self.images = images
        self.filtered_images = images[:IMAGES_BATCH_SIZE]  # Limit the number of displayed images initially
        self.displayed_images_count = len(self.filtered_images)
        self.loadingProgressBar.hide()
        self.stopButton.hide()
        qtw.QMessageBox.information(self, "Success!", "Data loaded successfully.")

    # Method to stop the data loading
    def stopLoading(self):
        if hasattr(self, 'loading_thread'):
            self.loading_thread.stop()
            self.loadingProgressBar.hide()
            self.stopButton.hide()

    # Method to view the images after loading and converting the data into images
    def viewConvertedImages(self):
        if not self.images:
            qtw.QMessageBox.warning(self, "Warning", "No data loaded. Please load data first.")
            return

        self.displayTrainingImages(self.file_path)

    # Method to filter images based on the search query
    def filterImages(self):
        query = self.search_bar.text()
        if query.isdigit():
            label = int(query)
            if 0 <= label <= 35:
                self.filtered_images = [img for img in self.images if img[0] == label]
                self.displayed_images_count = len(self.filtered_images)
            else:
                self.filtered_images = []
                qtw.QMessageBox.warning(self, "Error", "Please enter a label between 0 and 35.")
        else:
            if query:
                self.filtered_images = []
                qtw.QMessageBox.warning(self, "Error", "Please enter a valid digit between 0 and 35.")
            else:
                self.filtered_images = self.images[:IMAGES_BATCH_SIZE]
                self.displayed_images_count = len(self.filtered_images)

        # Update the image display
        self.updateImageDisplay(self.filtered_images)

    # Method to load more images when scrolling
    def loadMoreImages(self):
        if self.scroll_area.verticalScrollBar().value() == self.scroll_area.verticalScrollBar().maximum():
            next_images = self.filtered_images[
                          self.displayed_images_count:self.displayed_images_count + IMAGES_BATCH_SIZE]
            self.filtered_images.extend(next_images)
            self.displayed_images_count += len(next_images)
            self.updateImageDisplay(self.filtered_images, append=True)

    # Method to update the display of images
    def updateImageDisplay(self, images, append=False):
        if not append:
            # Clear any existing image display layout if not appending
            if hasattr(self, 'image_display_layout'):
                for i in reversed(range(self.image_display_layout.count())):
                    self.image_display_layout.itemAt(i).widget().setParent(None)

        # Populating the image array with images that have been converted from the .csv file
        for i, (label, pixmap) in enumerate(images):
            label_widget = qtw.QLabel()
            label_widget.setPixmap(pixmap)
            self.image_display_layout.addWidget(label_widget, (self.image_display_layout.count() // 10),
                                                self.image_display_layout.count() % 10)

    # Class initialization method
    def __init__(self, previouswindow):
        super().__init__()

        self.prevWindow = previouswindow
        self.data = None
        self.images = []
        self.filtered_images = []
        self.displayed_images_count = 0
        self.file_path = None

        # Creating a title for the window
        self.window = None
        self.setWindowTitle("User Options")
        self.setWindowIcon(QIcon('signsysweblogoturq.png'))

        # Setting geometry of the window
        self.setGeometry(950, 450, 720, 600)

        # Changing bg color to #8C52FF
        self.setStyleSheet("background-color: #8C52FF;")

        # Primary layout
        parent_layout = qtw.QVBoxLayout()

        self.setLayout(parent_layout)

        # Back button for activity window
        self.back_button = qtw.QPushButton("Return")
        self.back_button.setStyleSheet(activitystyles.button_style)
        self.back_button.clicked.connect(self.returnToHome)

        # Creating a text to ask the user what they want to do
        self.question_text = qtw.QLabel("What do you want to do?", self)
        self.question_text.setStyleSheet(activitystyles.text_styles)
        parent_layout.addWidget(self.back_button, alignment=Qt.AlignLeft)
        parent_layout.addWidget(self.question_text, alignment=Qt.AlignTop)

        horizontal_grid = qtw.QGridLayout()

        # Buttons Initialization
        self.load_data_button = qtw.QPushButton("Load Data")
        self.view_data_button = qtw.QPushButton("View Data")
        self.train_button = qtw.QPushButton("Train")
        self.test_button = qtw.QPushButton("Test")

        # Buttons layout / style
        self.load_data_button.setStyleSheet(activitystyles.button_style)
        self.view_data_button.setStyleSheet(activitystyles.button_style)
        self.train_button.setStyleSheet(activitystyles.button_style)
        self.test_button.setStyleSheet(activitystyles.button_style)
        # Buttons clicked
        self.train_button.clicked.connect(self.openTrainingWindow)
        self.test_button.clicked.connect(self.openTestModelWindow)

        # Add the buttons to the inner layout
        horizontal_grid.addWidget(self.load_data_button, 0, 0)  # The arguments are (widget, row, column)
        horizontal_grid.addWidget(self.view_data_button, 0, 1)
        horizontal_grid.addWidget(self.train_button, 0, 2)
        horizontal_grid.addWidget(self.test_button, 0, 3)

        parent_layout.addLayout(horizontal_grid)

        self.progressNum = None
        self.loadingProgressBar = qtw.QProgressBar()
        self.loadingProgressBar.setStyleSheet(activitystyles.loading_bar_style)
        self.loadingProgressBar.hide()

        # Add stop button below the progress bar
        self.stopButton = qtw.QPushButton("Stop")
        self.stopButton.setStyleSheet(activitystyles.button_style)
        self.stopButton.clicked.connect(self.stopLoading)
        self.stopButton.hide()

        # Center the progress bar and stop button
        progress_layout = qtw.QVBoxLayout()
        progress_layout.addWidget(self.loadingProgressBar)
        progress_layout.addWidget(self.stopButton, alignment=Qt.AlignCenter)

        parent_layout.addLayout(progress_layout)
        parent_layout.addStretch()

        self.load_data_button.clicked.connect(self.loadFile)
        self.view_data_button.clicked.connect(self.viewConvertedImages)
        self.show()
