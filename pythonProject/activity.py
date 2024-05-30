import pandas as pd
import numpy as np
import time
import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QMutexLocker

from pythonProject.train import TrainingWindow
from styles import ActivityStyles

activitystyles = ActivityStyles()

# Define a constant for the number of images to load per batch
IMAGES_BATCH_SIZE = 100


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

        # Read the CSV file into a DataFrame
        data = pd.read_csv(self.filepath)
        images = []  # List to hold the QPixmap images along with their labels
        progress_num = len(data)  # Total number of rows in the DataFrame

        # Loop through each row in the DataFrame
        for i in range(progress_num):
            with QMutexLocker(self.mutex):
                if self.stopped:
                    break

            label = data.iloc[i, 0]  # Extract the label
            pixels = data.iloc[i, 1:].values  # Extract pixel values from the row
            # Convert pixel values to a NumPy array and reshape to a 28x28 image
            image_array = np.array(pixels, dtype=np.uint8).reshape((28, 28))
            # Create a QImage from the NumPy array
            qimage = QImage(image_array.data, 28, 28, QImage.Format_Grayscale8)
            # Convert QImage to QPixmap and add to the list
            images.append((label, QPixmap.fromImage(qimage)))
            # Emit the progress signal with the percentage completed
            self.progress.emit(int((i + 1) / progress_num * 100))

            # Calculate elapsed time and estimate remaining time
            elapsed_time = time.time() - self.start_time
            if i > 0:  # Avoid division by zero
                remaining_time = elapsed_time / (i + 1) * (progress_num - i - 1)
                minutes, seconds = divmod(remaining_time, 60)
                self.time_remaining.emit(f"{int(minutes)} min {int(seconds)} sec left")

        # Emit the data_loaded signal with the list of QPixmap images if not stopped
        if not self.stopped:
            self.data_loaded.emit(images)

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True


class ActivityOptionsWindow(qtw.QWidget):

    def returnToHome(self):
        self.prevWindow.show()
        self.close()

    def openTrainingWindow(self):
        self.training_window = TrainingWindow(self)
        self.training_window.show()
        self.hide()

    # Method to load file
    def loadFile(self):
        fname = qtw.QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv);;All Files (*)")
        if fname[0]:
            self.loading_thread = DataLoaderThread(fname[0])
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
        # Showing a warning message if the user has not loaded any data
        if not self.images:
            qtw.QMessageBox.warning(self, "Warning", "No data loaded. Please load data first.")
            return

        # Clear any existing image display layout
        if hasattr(self, 'image_display_layout'):
            for i in reversed(range(self.image_display_layout.count())):
                self.image_display_layout.itemAt(i).widget().setParent(None)

        # Creating a grid layout for the loaded images to be displayed
        self.image_display_layout = qtw.QGridLayout()

        # Creating a scroll area to contain the grid layout
        scroll_area = qtw.QScrollArea()
        scroll_images_widget = qtw.QWidget()
        scroll_images_widget.setLayout(self.image_display_layout)

        # Creating search bar
        self.search_bar = qtw.QLineEdit(self)
        self.search_bar.setPlaceholderText("Search by label")
        self.search_bar.textChanged.connect(self.filterImages)

        # Add the search bar to the layout
        self.layout().insertWidget(1, self.search_bar)  # Insert above the scroll area

        # Add the images to the layout
        self.updateImageDisplay(self.filtered_images)

        scroll_area.setWidget(scroll_images_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedHeight(400)

        self.layout().addWidget(scroll_area)
        self.scroll_area = scroll_area
        self.show()

        # Connect scroll event to load more images
        scroll_area.verticalScrollBar().valueChanged.connect(self.loadMoreImages)

    # Method to filter images based on the search query
    def filterImages(self):
        query = self.search_bar.text()
        if query.isdigit():
            label = int(query)
            if 0 <= label <= 25:
                self.filtered_images = [img for img in self.images if img[0] == label]
                self.displayed_images_count = len(self.filtered_images)
            else:
                self.filtered_images = []
                qtw.QMessageBox.warning(self, "Error", "Please enter a label between 0 and 25.")
        else:
            if query:
                self.filtered_images = []
                qtw.QMessageBox.warning(self, "Error", "Please enter a valid digit between 0 and 25.")
            else:
                self.filtered_images = self.images[:IMAGES_BATCH_SIZE]
                self.displayed_images_count = len(self.filtered_images)

        # Update the image display
        self.updateImageDisplay(self.filtered_images)

    # Method to load more images when scrolling
    def loadMoreImages(self):
        if self.scroll_area.verticalScrollBar().value() == self.scroll_area.verticalScrollBar().maximum():
            next_images = self.images[self.displayed_images_count:self.displayed_images_count + IMAGES_BATCH_SIZE]
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
        # Creating a title for the window
        self.window = None
        self.setWindowTitle("User Options")
        self.setWindowIcon(QIcon('signsysweblogoturq.png'))

        # Setting geometry of the window
        self.setGeometry(950, 450, 720, 600)

        # Changing bg color to #31B1C8
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
