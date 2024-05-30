import pandas as pd
import numpy as np
import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from styles import ActivityStyles

activitystyles = ActivityStyles()

#  Created 'DataLoaderThread' class which handles the data loading and
#  image conversion in a background thread.
class DataLoaderThread(QThread):
    # Signal to indicate the progress of data loading
    progress = pyqtSignal(int)
    # Signal to indicate that the data loading is complete and to send the loaded images
    data_loaded = pyqtSignal(list)

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath  # Path to the CSV file to be loaded

    def run(self):
        # Read the CSV file into a DataFrame
        data = pd.read_csv(self.filepath)
        images = []  # List to hold the QPixmap images
        progress_num = len(data)  # Total number of rows in the DataFrame

        # Loop through each row in the DataFrame
        for i in range(progress_num):
            pixels = data.iloc[i, 1:].values  # Extract pixel values from the row
            # Convert pixel values to a NumPy array and reshape to a 28x28 image
            image_array = np.array(pixels, dtype=np.uint8).reshape((28, 28))
            # Create a QImage from the NumPy array
            qimage = QImage(image_array.data, 28, 28, QImage.Format_Grayscale8)
            # Convert QImage to QPixmap and add to the list
            images.append(QPixmap.fromImage(qimage))
            # Emit the progress signal with the percentage completed
            self.progress.emit(int((i + 1) / progress_num * 100))

        # Emit the data_loaded signal with the list of QPixmap images
        self.data_loaded.emit(images)


class ActivityOptionsWindow(qtw.QWidget):

    def returnToHome(self):
        self.prevWindow.show()
        self.close()

    # method to load file
    def loadFile(self):
        fname = qtw.QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv);;All Files (*)")
        if fname[0]:
            self.loading_thread = DataLoaderThread(fname[0])
            self.loading_thread.progress.connect(self.updateProgressBar)
            self.loading_thread.data_loaded.connect(self.onDataLoaded)
            self.loading_thread.start()
            self.loadingProgressBar.setRange(0, 100)
            self.loadingProgressBar.show()

    def updateProgressBar(self, value):
        self.loadingProgressBar.setValue(value)

    def onDataLoaded(self, images):
        self.images = images
        self.loadingProgressBar.hide()
        qtw.QMessageBox.information(self, "Success!", "Data loaded successfully.")

    # method to view the images after loading and converting the data into images
    def viewConvertedImages(self):
        # Showing a warning message if the user has not loaded any data
        if not self.images:
            qtw.QMessageBox.warning(self, "Warning", "No data loaded. Please load data first.")
            return

        # Clear any existing image display layout
        if hasattr(self, 'image_display_layout'):
            for i in reversed(range(self.image_display_layout.count())):
                self.image_display_layout.itemAt(i).widget().setParent(None)

        # Creating a grid layout for the loaded images to be displayted
        self.image_display_layout = qtw.QGridLayout()

        # creating a scroll area to contain the grid layout
        scroll_area = qtw.QScrollArea()
        scroll_images_widget = qtw.QWidget()
        scroll_images_widget.setLayout(self.image_display_layout)

        # populating the image array with images that have been converted from the
        # .csv file
        for i, pixmap in enumerate(self.images):
            label = qtw.QLabel()
            label.setPixmap(pixmap)
            self.image_display_layout.addWidget(label, i // 10, i % 10)  # 10 images per row

        scroll_area.setWidget(scroll_images_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedHeight(400)

        self.layout().addWidget(scroll_area)
        self.scroll_area = scroll_area
        self.show()

    # class initialisation method
    def __init__(self, previouswindow):
        super().__init__()

        self.prevWindow = previouswindow
        self.data = None
        self.images = []
        # Creating a title for the window
        self.window = None
        self.setWindowTitle("User Options")
        self.setWindowIcon(QIcon('signsysweblogoturq.png'))

        # Setting geometry of the window
        self.setGeometry(950, 450, 720, 600)

        # Changing bg color to #31B1C8
        self.setStyleSheet("background-color: #8C52FF;")

        # primary layout
        parent_layout = qtw.QVBoxLayout()

        self.setLayout(parent_layout)

        self.back_button = qtw.QPushButton("Return")
        self.back_button.setStyleSheet(activitystyles.button_style)
        self.back_button.clicked.connect(self.returnToHome)

        # Creating a text to ask the user what they want to do
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
        parent_layout.addWidget(self.loadingProgressBar, alignment=Qt.AlignTop)
        parent_layout.addStretch()
        self.load_data_button.clicked.connect(self.loadFile)
        self.view_data_button.clicked.connect(self.viewConvertedImages)
        self.show()
