import sys
import pandas as pd
import numpy as np
import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

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

    def __init__(self, prev_window):
        super().__init__()

        # Window set up for training page
        self.prev_window = prev_window
        self.setWindowTitle('Sign-SYS Training')
        self.setWindowIcon(QIcon('signsysweblogoturq.png'))
        self.setGeometry(950, 450, 720, 600)
        self.setStyleSheet('background-color: #8C52FF;')

        # Set up the layout for the training window. Use as reference for positioning
        training_layout = qtw.QVBoxLayout()
        self.setLayout(training_layout)

        # Horizontal layout for upper section
        top_layout = qtw.QVBoxLayout()

        # Return button to go to activity
        self.return_btn = qtw.QPushButton('Return')
        self.return_btn.setStyleSheet(activitystyles.button_style)
        self.return_btn.clicked.connect(self.returnToActivity)
        top_layout.addWidget(self.return_btn, alignment=Qt.AlignLeft) # Aligned to left of top layout

        # Text to select model
        self.model_select = qtw.QLabel("Select model to train")
        self.model_select.setStyleSheet(trainingstyles.title_styles)
        top_layout.addWidget(self.model_select, alignment=Qt.AlignCenter) # Aligned to center of top layout

        training_layout.addLayout(top_layout) # Adding top layout to the training layout

        # Drop down combo box for user to select model
        self.cnn_dropdown = qtw.QComboBox()
        self.cnn_dropdown.addItems(["-- Select Model --", "Sign-SYS CNN", "Inception V-3", "AlexNet"]) # List of model options to select from
        self.cnn_dropdown.setStyleSheet(trainingstyles.combobox_style)
        self.cnn_dropdown.setItemDelegate(CenterDropdownDelegate(self))
        training_layout.addWidget(self.cnn_dropdown, alignment=Qt.AlignTop)  # Aligned to top of training layout
        training_layout.addWidget(self.cnn_dropdown, alignment=Qt.AlignCenter) # Aligned to center of training layout

        # Text for Train/Validation Ratio Selection
        self.validation_text = qtw.QLabel("Select Train/Validation Ratio")
        self.validation_text.setStyleSheet(trainingstyles.title_styles)
        training_layout.addWidget(self.validation_text, alignment=Qt.AlignCenter) # Aligned to center of training layout

        # Train/Test Ratio Slider
        self.train_test_ratio_slider = qtw.QSlider(Qt.Horizontal)
        self.train_test_ratio_slider.setMinimum(1)
        self.train_test_ratio_slider.setMaximum(99)
        self.train_test_ratio_slider.setValue(80)
        self.train_test_ratio_slider.setTickPosition(qtw.QSlider.TicksBelow)
        self.train_test_ratio_slider.setTickInterval(10)
        self.train_test_ratio_slider.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Fixed)
        self.train_test_ratio_slider_label = qtw.QLabel(f'Train/Test Ratio: {self.train_test_ratio_slider.value()}%', self)

        training_layout.addWidget(self.train_test_ratio_slider_label, alignment=Qt.AlignCenter)
        training_layout.addWidget(self.train_test_ratio_slider, alignment=Qt.AlignCenter)

        # Text for Batch Size Selection
        self.batchsize_text = qtw.QLabel("Select a batch size")
        self.batchsize_text.setStyleSheet(trainingstyles.title_styles)
        training_layout.addWidget(self.batchsize_text, alignment=Qt.AlignCenter) # Aligned to center of training layout

        # Batch Size Slider
        self.batch_size_slider = qtw.QSlider(Qt.Horizontal)
        self.batch_size_slider.setMinimum(1)
        self.batch_size_slider.setMaximum(256)
        self.batch_size_slider.setValue(32)
        self.batch_size_slider.setTickPosition(qtw.QSlider.TicksBelow)
        self.batch_size_slider.setTickInterval(1000)
        self.batch_size_slider.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Fixed)
        self.batch_size_slider_label = qtw.QLabel(f'Batch Size: {self.batch_size_slider.value()}', self)

        training_layout.addWidget(self.batch_size_slider_label, alignment=Qt.AlignCenter)
        training_layout.addWidget(self.batch_size_slider, alignment=Qt.AlignCenter)

        # Text for Epochs Selection
        self.epochs_text = qtw.QLabel("Select amount of epochs")
        self.epochs_text.setStyleSheet(trainingstyles.title_styles)
        training_layout.addWidget(self.epochs_text, alignment=Qt.AlignCenter) # Aligned to center of training layout

        # Epochs Slider
        self.epochs_slider = qtw.QSlider(Qt.Horizontal)
        self.epochs_slider.setMinimum(1)
        self.epochs_slider.setMaximum(100)
        self.epochs_slider.setValue(10)
        self.epochs_slider.setTickPosition(qtw.QSlider.TicksBelow)
        self.epochs_slider.setTickInterval(10)
        self.epochs_slider.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Fixed)
        self.epochs_slider_label = qtw.QLabel(f'Epochs: {self.epochs_slider.value()}', self)

        training_layout.addWidget(self.epochs_slider_label, alignment=Qt.AlignCenter)
        training_layout.addWidget(self.epochs_slider, alignment=Qt.AlignCenter)

        # Connect sliders to their update methods
        self.train_test_ratio_slider.valueChanged.connect(lambda value: self.train_test_ratio_slider_label.setText(f'Train/Test Ratio: {value}%'))
        self.batch_size_slider.valueChanged.connect(lambda value: self.batch_size_slider_label.setText(f'Batch Size: {value}'))
        self.epochs_slider.valueChanged.connect(lambda value: self.epochs_slider_label.setText(f'Epochs: {value}'))

        # Ensure sliders expand to the full width of the window
        self.train_test_ratio_slider.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Fixed)
        self.batch_size_slider.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Fixed)
        self.epochs_slider.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Fixed)

