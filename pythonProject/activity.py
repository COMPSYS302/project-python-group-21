# activity.py
import pandas as pd
import numpy as np
import time
import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QMutexLocker
import logging

from train import TrainingWindow  # Ensure this import is correct
from styles import ActivityStyles

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

class DataLoaderThread(QThread):
    progress = pyqtSignal(int)
    data_loaded = pyqtSignal(list)
    time_remaining = pyqtSignal(str)

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.start_time = None
        self.mutex = QMutex()
        self.stopped = False

    def run(self):
        self.start_time = time.time()
        data = pd.read_csv(self.filepath)
        images = []
        progress_num = len(data)

        for i in range(progress_num):
            with QMutexLocker(self.mutex):
                if self.stopped:
                    break

            label = data.iloc[i, 0]
            label = label_remap.get(label, label)

            pixels = data.iloc[i, 1:].values
            image_array = np.array(pixels, dtype=np.uint8).reshape((28, 28))
            qimage = QImage(image_array.data, 28, 28, QImage.Format_Grayscale8)
            images.append((label, QPixmap.fromImage(qimage)))
            self.progress.emit(int((i + 1) / progress_num * 100))

            elapsed_time = time.time() - self.start_time
            if i > 0:
                remaining_time = elapsed_time / (i + 1) * (progress_num - i - 1)
                minutes, seconds = divmod(remaining_time, 60)
                self.time_remaining.emit(f"{int(minutes)} min {int(seconds)} sec left")

        if not self.stopped:
            self.data_loaded.emit(images)

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True


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
            logging.debug("Opening Training Window")
            self.training_window = TrainingWindow(self, self.file_path)
            self.training_window.show()
            self.hide()

    def loadFile(self):
        fname = qtw.QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv);;All Files (*)")
        if fname[0]:
            self.file_path = fname[0]
            logging.debug(f"Loading file: {self.file_path}")
            self.loading_thread = DataLoaderThread(fname[0])
            self.loading_thread.progress.connect(self.updateProgressBar)
            self.loading_thread.data_loaded.connect(self.onDataLoaded)
            self.loading_thread.time_remaining.connect(self.updateTimeRemaining)
            self.loading_thread.start()
            self.loadingProgressBar.setRange(0, 100)
            self.loadingProgressBar.setAlignment(Qt.AlignCenter)
            self.loadingProgressBar.show()
            self.stopButton.show()

    def updateProgressBar(self, value):
        self.loadingProgressBar.setValue(value)

    def updateTimeRemaining(self, time_left):
        self.loadingProgressBar.setFormat(f"%p% - {time_left}")

    def onDataLoaded(self, images):
        self.images = images
        self.filtered_images = images[:IMAGES_BATCH_SIZE]
        self.displayed_images_count = len(self.filtered_images)
        self.loadingProgressBar.hide()
        self.stopButton.hide()
        qtw.QMessageBox.information(self, "Success!", "Data loaded successfully.")

    def stopLoading(self):
        if hasattr(self, 'loading_thread'):
            self.loading_thread.stop()
            self.loadingProgressBar.hide()
            self.stopButton.hide()

    def viewConvertedImages(self):
        if not self.images:
            qtw.QMessageBox.warning(self, "Warning", "No data loaded. Please load data first.")
            return

        if hasattr(self, 'image_display_layout') and hasattr(self, 'search_bar'):
            for i in reversed(range(self.image_display_layout.count())):
                self.image_display_layout.itemAt(i).widget().setParent(None)

            self.updateImageDisplay(self.filtered_images)
            self.show()
            return

        self.image_display_layout = qtw.QGridLayout()
        scroll_area = qtw.QScrollArea()
        scroll_images_widget = qtw.QWidget()
        scroll_images_widget.setLayout(self.image_display_layout)

        self.search_bar = qtw.QLineEdit(self)
        self.search_bar.setPlaceholderText("Search by label")
        self.search_bar.textChanged.connect(self.filterImages)
        self.search_bar.setStyleSheet(activitystyles.line_edit_style)

        self.layout().insertWidget(1, self.search_bar)
        self.updateImageDisplay(self.filtered_images)

        scroll_area.setWidget(scroll_images_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedHeight(400)

        self.layout().addWidget(scroll_area)
        self.scroll_area = scroll_area
        self.show()

        scroll_area.verticalScrollBar().valueChanged.connect(self.loadMoreImages)

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

        self.updateImageDisplay(self.filtered_images)

    def loadMoreImages(self):
        if self.scroll_area.verticalScrollBar().value() == self.scroll_area.verticalScrollBar().maximum():
            next_images = self.filtered_images[self.displayed_images_count:self.displayed_images_count + IMAGES_BATCH_SIZE]
            self.filtered_images.extend(next_images)
            self.displayed_images_count += len(next_images)
            self.updateImageDisplay(self.filtered_images, append=True)

    def updateImageDisplay(self, images, append=False):
        if not append:
            if hasattr(self, 'image_display_layout'):
                for i in reversed(range(self.image_display_layout.count())):
                    self.image_display_layout.itemAt(i).widget().setParent(None)

        for i, (label, pixmap) in enumerate(images):
            label_widget = qtw.QLabel()
            label_widget.setPixmap(pixmap)
            self.image_display_layout.addWidget(label_widget, (self.image_display_layout.count() // 10), self.image_display_layout.count() % 10)

    def __init__(self, previouswindow):
        super().__init__()

        self.prevWindow = previouswindow
        self.data = None
        self.images = []
        self.filtered_images = []
        self.displayed_images_count = 0
        self.file_path = None

        self.window = None
        self.setWindowTitle("User Options")
        self.setWindowIcon(QIcon('signsysweblogoturq.png'))

        self.setGeometry(950, 450, 720, 600)
        self.setStyleSheet("background-color: #8C52FF;")

        parent_layout = qtw.QVBoxLayout()
        self.setLayout(parent_layout)

        self.back_button = qtw.QPushButton("Return")
        self.back_button.setStyleSheet(activitystyles.button_style)
        self.back_button.clicked.connect(self.returnToHome)

        self.question_text = qtw.QLabel("What do you want to do?", self)
        self.question_text.setStyleSheet(activitystyles.text_styles)
        parent_layout.addWidget(self.back_button, alignment=Qt.AlignLeft)
        parent_layout.addWidget(self.question_text, alignment=Qt.AlignTop)

        horizontal_grid = qtw.QGridLayout()

        self.load_data_button = qtw.QPushButton("Load Data")
        self.view_data_button = qtw.QPushButton("View Data")
        self.train_button = qtw.QPushButton("Train")
        self.test_button = qtw.QPushButton("Test")

        self.load_data_button.setStyleSheet(activitystyles.button_style)
        self.view_data_button.setStyleSheet(activitystyles.button_style)
        self.train_button.setStyleSheet(activitystyles.button_style)
        self.test_button.setStyleSheet(activitystyles.button_style)

        self.train_button.clicked.connect(self.openTrainingWindow)

        horizontal_grid.addWidget(self.load_data_button, 0, 0)
        horizontal_grid.addWidget(self.view_data_button, 0, 1)
        horizontal_grid.addWidget(self.train_button, 0, 2)
        horizontal_grid.addWidget(self.test_button, 0, 3)

        parent_layout.addLayout(horizontal_grid)

        self.progressNum = None
        self.loadingProgressBar = qtw.QProgressBar()
        self.loadingProgressBar.setStyleSheet(activitystyles.loading_bar_style)
        self.loadingProgressBar.hide()

        self.stopButton = qtw.QPushButton("Stop")
        self.stopButton.setStyleSheet(activitystyles.button_style)
        self.stopButton.clicked.connect(self.stopLoading)
        self.stopButton.hide()

        progress_layout = qtw.QVBoxLayout()
        progress_layout.addWidget(self.loadingProgressBar)
        progress_layout.addWidget(self.stopButton, alignment=Qt.AlignCenter)

        parent_layout.addLayout(progress_layout)
        parent_layout.addStretch()

        self.load_data_button.clicked.connect(self.loadFile)
        self.view_data_button.clicked.connect(self.viewConvertedImages)
        self.show()
