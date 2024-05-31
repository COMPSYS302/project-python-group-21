import sys

import pandas as pd
import numpy as np
import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from styles import ActivityStyles
from styles import TrainingStyles

activitystyles = ActivityStyles()
trainingstyles = TrainingStyles()

# Drop down items style / delegate
class CenterDropdownDelegate(qtw.QStyledItemDelegate):
    def __init__(self, parent=None):
        super(CenterDropdownDelegate, self).__init__(parent)

    def initStyleOption(self, option, index):
        super(CenterDropdownDelegate, self).initStyleOption(option, index)
        option.displayAlignment = Qt.AlignCenter

class TrainingWindow(qtw.QWidget):
    def returnToActivity(self):
        self.prev_window.show()
        self.close()

    def updateSliderLabel(self, slider, label):
        label.setText(f"{slider.value()}")

    def __init__(self, prev_window):
        super().__init__()

        # Window set up for training page
        self.prev_window = prev_window
        self.setWindowTitle('Sign-SYS Training')
        self.setWindowIcon(QIcon('signsysweblogoturq.png'))
        self.setGeometry(950, 450, 720, 600)
        self.setStyleSheet('background-color: #8C52FF;')

        # Set up the layout for the training window. Use as reference for positioning
        training_layout = qtw.QGridLayout()
        self.setLayout(training_layout)

        # Return button to go to activity
        self.return_btn = qtw.QPushButton('Return')
        self.return_btn.setStyleSheet(activitystyles.button_style)
        self.return_btn.clicked.connect(self.returnToActivity)
        training_layout.addWidget(self.return_btn, 0, 0,  alignment=Qt.AlignLeft)  # Aligned to left of top layout

        # Model Selection
        # Text to select model
        self.model_select = qtw.QLabel("Select model to train")
        self.model_select.setStyleSheet(trainingstyles.title_styles)
        training_layout.addWidget(self.model_select, 0, 1, alignment=Qt.AlignCenter)  # Aligned to center of top layout

        # Drop down combo box for user to select model
        self.cnn_dropdown = qtw.QComboBox()
        # List of model options to select from
        self.cnn_dropdown.addItems(["-- Select Model --", "Sign-SYS CNN", "Inception V-3", "AlexNet"])
        self.cnn_dropdown.setStyleSheet(trainingstyles.combobox_style)
        self.cnn_dropdown.setItemDelegate(CenterDropdownDelegate(self))
        training_layout.addWidget(self.cnn_dropdown, 1, 1, alignment=Qt.AlignCenter) # Aligned to center of top layout

        # Train / Validation Ratio Selection
        # Text for Train/Validation Ratio Selection
        self.validation_text = qtw.QLabel("Select Train/Validation Ratio")
        self.validation_text.setStyleSheet(trainingstyles.title_styles)
        training_layout.addWidget(self.validation_text, 2, 1, alignment=Qt.AlignCenter)

        # Slider for Train/Validation Ratio Selection
        self.validation_slider = qtw.QSlider(Qt.Horizontal)
        self.validation_slider.setMinimum(0)
        self.validation_slider.setMaximum(100)
        self.validation_slider.setValue(80)
        self.validation_slider.setTickInterval(5)
        self.validation_slider.setTickPosition(qtw.QSlider.TicksBelow)

        # Labels for Train/Validation Ratio Selection
        self.validation_min_label = qtw.QLabel("0")
        self.validation_max_label = qtw.QLabel("100")
        self.validation_value_label = qtw.QLabel(f"{self.validation_slider.value()}")
        self.validation_slider.valueChanged.connect(lambda: self.updateSliderLabel(self.validation_slider,
                                                                                   self.validation_value_label))
        training_layout.addWidget(self.validation_min_label, 3, 0, alignment=Qt.AlignCenter)
        training_layout.addWidget(self.validation_slider, 3, 1, alignment=Qt.AlignCenter)
        training_layout.addWidget(self.validation_value_label, 3, 2, alignment=Qt.AlignCenter)
        training_layout.addWidget(self.validation_max_label, 3, 3, alignment=Qt.AlignCenter)

        # Batch Size
        # Text for Batch Size Selection
        self.batchsize_text = qtw.QLabel("Select a batch size")
        self.batchsize_text.setStyleSheet(trainingstyles.title_styles)
        training_layout.addWidget(self.batchsize_text, 4, 1, alignment=Qt.AlignCenter) # Aligned to center of training layout

        # Slider for Batch Size Selection
        self.batchsize_slider = qtw.QSlider(Qt.Horizontal)
        self.batchsize_slider.setMinimum(8)
        self.batchsize_slider.setMaximum(256)
        self.batchsize_slider.setValue(32)
        self.batchsize_slider.setTickInterval(8)
        self.batchsize_slider.setTickPosition(qtw.QSlider.TicksBelow)

        # Labels for Batch Size Selection
        self.batchsize_min_label = qtw.QLabel("8")
        self.batchsize_max_label = qtw.QLabel("256")
        self.batchsize_value_label = qtw.QLabel(f"{self.batchsize_slider.value()}")
        self.batchsize_slider.valueChanged.connect(lambda: self.updateSliderLabel(self.batchsize_slider,
                                                                                   self.batchsize_value_label))
        training_layout.addWidget(self.batchsize_min_label, 5, 0, alignment=Qt.AlignCenter)
        training_layout.addWidget(self.batchsize_slider, 5, 1, alignment=Qt.AlignCenter)
        training_layout.addWidget(self.batchsize_value_label, 5, 2, alignment=Qt.AlignCenter)
        training_layout.addWidget(self.batchsize_max_label, 5, 3, alignment=Qt.AlignCenter)

        # Epochs Selection
        # Text for Epochs Selection
        self.epochs_text = qtw.QLabel("Select amount of epochs")
        self.epochs_text.setStyleSheet(trainingstyles.title_styles)
        training_layout.addWidget(self.epochs_text, 6, 1, alignment=Qt.AlignCenter) # Aligned to center of training layout

        # Slider for Epochs Selection
        self.epochs_slider = qtw.QSlider(Qt.Horizontal)
        self.epochs_slider.setMinimum(10)
        self.epochs_slider.setMaximum(100)
        self.epochs_slider.setValue(30)
        self.epochs_slider.setTickInterval(10)
        self.epochs_slider.setTickPosition(qtw.QSlider.TicksBelow)
        training_layout.addWidget(self.epochs_slider, 7,1, alignment=Qt.AlignCenter)
if __name__ == '__main__':
    app = qtw.QApplication([])
    main_win = TrainingWindow(None)
    main_win.show()
    app.exec_()
