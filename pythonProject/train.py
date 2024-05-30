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
        self.setLayout(top_layout)

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

        # Text for Batch Size Selection
        self.batchsize_text = qtw.QLabel("Select a batch size")
        self.batchsize_text.setStyleSheet(trainingstyles.title_styles)
        training_layout.addWidget(self.batchsize_text, alignment=Qt.AlignCenter) # Aligned to center of training layout

        # Text for Epochs Selection
        self.epochs_text = qtw.QLabel("Select amount of epochs")
        self.epochs_text.setStyleSheet(trainingstyles.title_styles)
        training_layout.addWidget(self.epochs_text, alignment=Qt.AlignCenter) # Aligned to center of training layout

if __name__ == '__main__':
    app = qtw.QApplication([])
    main_win = TrainingWindow(None)
    main_win.show()
    app.exec_()
