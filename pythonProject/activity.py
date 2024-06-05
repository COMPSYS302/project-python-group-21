import pandas as pd
import numpy as np
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import PyQt5.QtWidgets as qtw
import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QMutexLocker
import logging

from train import TrainingWindow  # Ensure this import is correct
from styles import ActivityStyles
from alexnet import build_alexnet
from inception import build_inception_v3
from camera_utils import CameraHandler  # Import the camera handler

activitystyles = ActivityStyles()

IMAGES_BATCH_SIZE = 100
INITIAL_LOAD_SIZE = 200

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
        try:
            data = pd.read_csv(self.filepath)
        except Exception as e:
            logging.error(f"Error reading CSV file: {e}")
            return

        images = []
        progress_num = len(data)

        for i in range(progress_num):
            with QMutexLocker(self.mutex):
                if self.stopped:
                    break

            try:
                label = data.iloc[i, 0]
                label = label_remap.get(label, label)

                pixels = data.iloc[i, 1:].values
                image_array = np.array(pixels, dtype=np.uint8).reshape((28, 28))
                qimage = QImage(image_array.data, 28, 28, QImage.Format_Grayscale8)
                images.append((label, QPixmap.fromImage(qimage), image_array))
                self.progress.emit(int((i + 1) / progress_num * 100))

                elapsed_time = time.time() - self.start_time
                if i > 0:
                    remaining_time = elapsed_time / (i + 1) * (progress_num - i - 1)
                    minutes, seconds = divmod(remaining_time, 60)
                    self.time_remaining.emit(f"{int(minutes)} min {int(seconds)} sec left")
            except Exception as e:
                logging.error(f"Error processing row {i}: {e}")
                continue

        if not self.stopped:
            self.data_loaded.emit(images)

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True

class ProbabilityWindow(qtw.QWidget):
    def __init__(self, probabilities, predicted_label, pixmap):
        super().__init__()
        self.setWindowTitle("Prediction Probabilities")
        self.setWindowIcon(QIcon('signsysweblogoturq.png'))
        self.setGeometry(100, 100, 800, 750)  # X Y Width Height
        self.setStyleSheet("background-color: #8C52FF;")
        layout = qtw.QVBoxLayout()
        self.setLayout(layout)

        # Display the clicked image zoomed in
        image_label = qtw.QLabel()
        image_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))
        layout.addWidget(image_label, alignment=Qt.AlignCenter)

        # Display the predicted label
        label = qtw.QLabel(f"Predicted Label: {predicted_label}")
        label.setStyleSheet(activitystyles.regular_text)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        # Plotting the probabilities
        figure, ax = plt.subplots()
        ax.bar(range(len(probabilities)), probabilities)
        ax.set_xlabel('Class')
        ax.set_ylabel('Probability')
        ax.grid(True)

        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)

class ClickableLabel(qtw.QLabel):
    def __init__(self, label, pixmap, image_array, model, model_name):
        super().__init__()
        self.label = label
        self.setPixmap(pixmap)
        self.image_array = image_array
        self.model = model
        self.model_name = model_name
        self.pixmap = pixmap

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.predict_and_show()

    def predict_and_show(self):
        try:
            self.model.eval()
            with torch.no_grad():
                image_tensor = torch.tensor(self.image_array).unsqueeze(0).unsqueeze(0).float()
                output = self.model(image_tensor)
                probabilities = F.softmax(output, dim=1).numpy().flatten()
                _, predicted = torch.max(output, 1)
                predicted_label = predicted.item()

            self.prob_window = ProbabilityWindow(probabilities, predicted_label, self.pixmap)
            self.prob_window.show()

        except Exception as e:
            qtw.QMessageBox.critical(self, "Error", f"Failed to predict: {str(e)}")

class TestModelWindow(qtw.QWidget):
    def __init__(self, images, model, model_name):
        super().__init__()
        self.images = images
        self.model = model
        self.model_name = model_name
        self.page_size = 100
        self.current_page = 0
        self.filtered_images = self.images
        self.total_pages = (len(self.images) + self.page_size - 1)
        self.setWindowTitle("Test Model")
        self.setWindowIcon(QIcon('signsysweblogoturq.png'))
        self.setGeometry(300, 100, 800, 600)
        self.setStyleSheet("background-color: #8C52FF;")

        self.layout = qtw.QVBoxLayout(self)

        # Search bar for test window
        self.search_bar = qtw.QLineEdit(self)
        self.search_bar.setPlaceholderText("Search by predicted value")
        self.search_bar.setStyleSheet(activitystyles.line_edit_style)
        self.search_bar.textChanged.connect(self.filter_images)
        self.layout.addWidget(self.search_bar)

        # Test images are displayed here
        self.image_display_layout = qtw.QGridLayout()
        scroll_area = qtw.QScrollArea()
        scroll_images_widget = qtw.QWidget()
        scroll_images_widget.setLayout(self.image_display_layout)

        scroll_area.setWidget(scroll_images_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedHeight(400)

        self.layout.addWidget(scroll_area)

        # Page navigation layout
        self.test_bottom_layout = qtw.QHBoxLayout()
        self.prev_button = qtw.QPushButton("Previous")
        self.next_button = qtw.QPushButton("Next")

        # Setting the buttons designs
        self.prev_button.setStyleSheet(activitystyles.button_style)
        self.next_button.setStyleSheet(activitystyles.button_style)

        # Adding the buttons to the layout
        self.test_bottom_layout.addWidget(self.prev_button)
        self.test_bottom_layout.addWidget(self.next_button)
        self.layout.addLayout(self.test_bottom_layout)

        self.prev_button.clicked.connect(self.prev_page)
        self.next_button.clicked.connect(self.next_page)

        # Cache the icons to prevent lag
        self.purple_cam_icon = qtg.QIcon("purple_cam.png")
        self.white_cam_icon = qtg.QIcon("white_cam.png")

        # Camera Button set up
        self.camera_button = qtw.QPushButton("Camera")
        self.camera_button.setIcon(self.purple_cam_icon)
        self.camera_button.setIconSize(qtc.QSize(16, 16))
        self.camera_button.setStyleSheet(activitystyles.button_style)
        self.layout.addWidget(self.camera_button, alignment=Qt.AlignCenter)
        self.camera_button.installEventFilter(self)
        self.camera_button.clicked.connect(self.open_camera)

        self.display_images()

    def display_images(self):
        # Clear current images
        for i in reversed(range(self.image_display_layout.count())):
            widget_to_remove = self.image_display_layout.itemAt(i).widget()
            self.image_display_layout.removeWidget(widget_to_remove)
            widget_to_remove.setParent(None)

        # Display images for the current page
        start_index = self.current_page * self.page_size
        end_index = min(start_index + self.page_size, len(self.filtered_images))
        for i in range(start_index, end_index):
            label, pixmap, image_array = self.filtered_images[i]
            label_widget = ClickableLabel(label, pixmap, image_array, self.model, self.model_name)
            self.image_display_layout.addWidget(label_widget, (i - start_index) // 10, (i - start_index) % 10)

    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.display_images()

    def next_page(self):
        if self.current_page < (len(self.filtered_images) + self.page_size - 1) // self.page_size - 1:
            self.current_page += 1
            self.display_images()

    def filter_images(self):
        query = self.search_bar.text()
        if query.isdigit():
            predicted_label = int(query)
            self.filtered_images = [img for img in self.images if img[0] == predicted_label]
        else:
            self.filtered_images = self.images
        self.current_page = 0
        self.display_images()

    def eventFilter(self, obj, event):
        if obj == self.camera_button:
            if event.type() == qtc.QEvent.Enter:
                self.camera_button.setIcon(self.white_cam_icon)
            elif event.type() == qtc.QEvent.Leave:
                self.camera_button.setIcon(self.purple_cam_icon)
        return super().eventFilter(obj, event)

    def open_camera(self):
        camera_handler = CameraHandler(self.model)
        camera_handler.open_camera()

class ActivityOptionsWindow(qtw.QWidget):
    def returnToHome(self):
        msg = qtw.QMessageBox(self)
        msg.setWindowTitle("Return to Home")
        msg.setText("Returning will clear loaded data. Do you want to continue?")
        msg.setIcon(qtw.QMessageBox.Question)
        msg.setStyleSheet(activitystyles.msg_box_design)

        yes_button = qtw.QPushButton("Yes")
        yes_button.setStyleSheet(activitystyles.msg_box_button)
        msg.addButton(yes_button, qtw.QMessageBox.YesRole)

        no_button = qtw.QPushButton("No")
        no_button.setStyleSheet(activitystyles.msg_box_button)
        msg.addButton(no_button, qtw.QMessageBox.NoRole)

        response = msg.exec_()
        if response == 0:
            self.prevWindow.show()
            self.close()

    def openTrainingWindow(self):
        if not self.images:
            msg = qtw.QMessageBox(self)
            msg.setWindowTitle("Warning")
            msg.setText("No data loaded. Please load data first.")
            msg.setIcon(qtw.QMessageBox.Information)
            msg.setStyleSheet(activitystyles.msg_box_design)
            msg.exec_()
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
            model_info = torch.load(model_path, map_location=self.device)  # Load to CPU first
            model_name = model_info['model_name']
            model_state_dict = model_info['model_state_dict']

            if model_name == "AlexNet":
                model = build_alexnet(num_classes=36)
            elif model_name == "InceptionV3":
                model = build_inception_v3(num_classes=36)
            else:
                raise ValueError("Unknown model name")

            model.load_state_dict(model_state_dict)
            model.to(self.device)
            model.eval()

            msg = qtw.QMessageBox(self)
            msg.setWindowTitle("Success")
            msg.setText(f"Model {model_name} loaded successfully for testing.")
            msg.setStyleSheet(activitystyles.msg_box_design)
            msg.exec_()

            self.model = model
            self.model_name = model_name

            self.openTestModelWindowWithImages(model, model_name)

        except Exception as e:
            qtw.QMessageBox.critical(self, "Error", f"Failed to load the model: {str(e)}")

    def openTestModelWindowWithImages(self, model, model_name):
        if not self.images:
            qtw.QMessageBox.warning(self, "Warning", "No data loaded. Please load data first.")
            return

        self.test_model_window = TestModelWindow(self.images, model, model_name)
        self.test_model_window.show()

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
            self.loadingProgressBar.setAlignment(Qt.AlignCenter)
            self.loadingProgressBar.show()
            self.stopButton.show()

    def updateProgressBar(self, value):
        self.loadingProgressBar.setValue(value)

    def updateTimeRemaining(self, time_left):
        self.loadingProgressBar.setFormat(f"%p% - {time_left}")

    def onDataLoaded(self, images):
        self.images = images
        self.filtered_images = images[:INITIAL_LOAD_SIZE]
        self.displayed_images_count = len(self.filtered_images)
        self.loadingProgressBar.hide()
        self.stopButton.hide()

        msg = qtw.QMessageBox(self)
        msg.setWindowTitle("Success!")
        msg.setText("Data loaded successfully.")
        msg.setIcon(qtw.QMessageBox.Information)
        msg.setStyleSheet(activitystyles.msg_box_design)
        msg.exec_()

    def stopLoading(self):
        if hasattr(self, 'loading_thread'):
            self.loading_thread.stop()
            self.loadingProgressBar.hide()
            self.stopButton.hide()

    def viewConvertedImages(self, model=None, model_name=None):
        if not self.images:
            msg = qtw.QMessageBox(self)
            msg.setWindowTitle("Warning")
            msg.setText("No data loaded. Please load data first.")
            msg.setIcon(qtw.QMessageBox.Information)
            msg.setStyleSheet(activitystyles.msg_box_design)
            msg.exec_()
            return

        # Clear previous search bar and scroll area if they exist
        if hasattr(self, 'search_bar') and self.search_bar:
            self.search_bar.deleteLater()
            self.search_bar = None

        if hasattr(self, 'scroll_area') and self.scroll_area:
            self.scroll_area.deleteLater()
            self.scroll_area = None

        self.search_bar = qtw.QLineEdit(self)
        self.search_bar.setPlaceholderText("Search by label")
        self.search_bar.textChanged.connect(self.filterImages)
        self.search_bar.setStyleSheet(activitystyles.line_edit_style)

        self.layout().insertWidget(1, self.search_bar)

        # Create new scroll area and image display layout
        self.image_display_layout = qtw.QGridLayout()
        self.scroll_area = qtw.QScrollArea()
        scroll_images_widget = qtw.QWidget()
        scroll_images_widget.setLayout(self.image_display_layout)

        self.updateImageDisplay(self.filtered_images, model, model_name)

        self.scroll_area.setWidget(scroll_images_widget)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedHeight(400)

        self.layout().addWidget(self.scroll_area)

        self.scroll_area.verticalScrollBar().valueChanged.connect(self.loadMoreImages)

    def filterImages(self):
        query = self.search_bar.text()
        if query.isdigit():
            label = int(query)
            if 0 <= label <= 35:
                self.filtered_images = [img for img in self.images if img[0] == label]
                self.displayed_images_count = len(self.filtered_images)
            else:
                self.filtered_images = []
                msg = qtw.QMessageBox(self)
                msg.setWindowTitle("Error")
                msg.setText("Please enter a label between 0 and 35.")
                msg.setIcon(qtw.QMessageBox.Information)
                msg.setStyleSheet(activitystyles.msg_box_design)
                msg.exec_()
        else:
            if query:
                self.filtered_images = []
                msg = qtw.QMessageBox(self)
                msg.setWindowTitle("Error")
                msg.setText("Please enter a valid digit between 0 and 35.")
                msg.setIcon(qtw.QMessageBox.Information)
                msg.setStyleSheet(activitystyles.msg_box_design)
                msg.exec_()
            else:
                self.filtered_images = self.images[:INITIAL_LOAD_SIZE]
                self.displayed_images_count = len(self.filtered_images)

        self.viewFilteredImages()

    def viewFilteredImages(self):
        self.image_display_layout = qtw.QGridLayout()
        scroll_images_widget = qtw.QWidget()
        scroll_images_widget.setLayout(self.image_display_layout)

        self.updateImageDisplay(self.filtered_images)

        self.scroll_area.setWidget(scroll_images_widget)

    def loadMoreImages(self):
        if self.scroll_area.verticalScrollBar().value() == self.scroll_area.verticalScrollBar().maximum():
            next_images = self.filtered_images[
                          self.displayed_images_count:self.displayed_images_count + IMAGES_BATCH_SIZE]
        if self.scroll_area.verticalScrollBar().value() == self.scroll_area.verticalScrollBar().maximum() and not self.search_bar.text().isdigit():
            next_images = self.images[self.displayed_images_count:self.displayed_images_count + IMAGES_BATCH_SIZE]
            self.filtered_images.extend(next_images)
            self.displayed_images_count += len(next_images)
            self.updateImageDisplay(self.filtered_images, self.model, self.model_name, append=True)

    def updateImageDisplay(self, images, model=None, model_name=None, append=False):
        if not append:
            if hasattr(self, 'image_display_layout'):
                for i in reversed(range(self.image_display_layout.count())):
                    self.image_display_layout.itemAt(i).widget().setParent(None)

        for i, (label, pixmap, image_array) in enumerate(images):
            if model:
                label_widget = ClickableLabel(label, pixmap, image_array, model, model_name)
            else:
                label_widget = qtw.QLabel()
                label_widget.setPixmap(pixmap)
            self.image_display_layout.addWidget(label_widget, (self.image_display_layout.count() // 10),
                                                self.image_display_layout.count() % 10)

    def __init__(self, previouswindow):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.prevWindow = previouswindow
        self.data = None
        self.images = []
        self.filtered_images = []
        self.displayed_images_count = 0
        self.file_path = None
        self.model = None
        self.model_name = None

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
        self.test_button.clicked.connect(self.openTestModelWindow)

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
        self.view_data_button.clicked.connect(lambda: self.viewConvertedImages(self.model, self.model_name))
        self.show()
