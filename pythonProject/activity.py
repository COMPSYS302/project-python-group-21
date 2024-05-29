import time

import pandas as pd
import numpy as np
import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import Qt

from styles import ActivityStyles

activitystyles = ActivityStyles()


class ActivityOptionsWindow(qtw.QWidget):

    def returnToHome(self):
        self.prevWindow.show()
        self.close()

    # method to load file
    def loadFile(self):
        fname = qtw.QFileDialog.getOpenFileName(self, "Open File", "",
                                                "CSV Files (*.csv);;All Files (*)")
        if fname[0]:
            try:
                data = pd.read_csv(fname[0])
                self.convertDataToImages(data)
                success_message = qtw.QMessageBox()
                success_message.information(self, "Success!", "Data loaded successfully.")
            except Exception as e:
                qtw.QMessageBox.critical(self, "Error", f"Could not load file: {e}")

    # ===========================================================================================
    # method to convert CSV data to images
    def convertDataToImages(self, data):
        self.data = data
        self.images = []
        self.progressNum = len(data)
        self.loadingProgressBar.setRange(0, self.progressNum)
        self.loadingProgressBar.show()

        for i in range (len(data)):
            self.loadingProgressBar.setValue(i+1)
            pixels = data.iloc[i,1:].values
            image_array = np.array(pixels, dtype=np.uint8).reshape((28,28))
            qimage = QImage(image_array.data, 28,28, QImage.Format_Grayscale8)
            self.images.append(QPixmap.fromImage(qimage))
        self.loadingProgressBar.hide()

    # ============================================================================================
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

    # ============================================================================================
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


