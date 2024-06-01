# train.py
import sys
import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
import threading
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from styles import ActivityStyles
from styles import TrainingStyles
import train_model

activitystyles = ActivityStyles()
trainingstyles = TrainingStyles()

class TrainingProgressWindow(qtw.QWidget):
    update_plot_signal = pyqtSignal(list, list, list, list)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Sign-SYS Training Progress')
        self.setWindowIcon(QIcon('signsysweblogoturq.png'))
        self.setGeometry(850, 450, 850, 850)
        self.setStyleSheet('background-color: #8C52FF;')

        layout = qtw.QVBoxLayout()
        self.setLayout(layout)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.timer_label = qtw.QLabel('Time elapsed: 00:00:00')
        self.timer_label.setStyleSheet(activitystyles.text_styles)
        layout.addWidget(self.timer_label, alignment=Qt.AlignCenter)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)
        self.start_time = 0

        self.training_losses = []
        self.validation_accuracies = []
        self.epochs = []
        self.val_epochs = []

        self.update_plot_signal.connect(self.update_plots)

    def start_timer(self):
        self.start_time = 0
        self.timer.start(1000)

    def stop_timer(self):
        self.timer.stop()

    def update_timer(self):
        self.start_time += 1
        hours, remainder = divmod(self.start_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.timer_label.setText(f'Time elapsed: {hours:02}:{minutes:02}:{seconds:02}')

    def update_plots(self, training_losses, validation_accuracies, epochs, val_epochs):
        self.figure.clear()

        ax1 = self.figure.add_subplot(211)
        ax1.plot(epochs, training_losses, 'r-')
        ax1.set_title('Training Losses')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.margins(0.1)

        ax2 = self.figure.add_subplot(212)
        ax2.plot(val_epochs, validation_accuracies, 'b-')
        ax2.set_title('Validation Accuracies')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.margins(0.1)

        self.figure.tight_layout(pad=2.0)

        self.canvas.draw()

    def add_data(self, epoch, train_loss, val_accuracy):
        self.epochs.append(epoch)
        self.val_epochs.append(epoch)
        self.training_losses.append(train_loss)
        self.validation_accuracies.append(val_accuracy)
        self.update_plot_signal.emit(self.training_losses, self.validation_accuracies, self.epochs, self.val_epochs)


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

    def startTraining(self):
        model_name = self.cnn_dropdown.currentText()
        batch_size = self.batch_size_slider.value()
        epochs = self.epochs_slider.value()
        validation_split = self.train_test_ratio_slider.value() / 100.0

        if model_name == "AlexNet" and self.file_path:
            print("Starting training...")
            self.progress_window = TrainingProgressWindow()
            self.progress_window.show()
            self.progress_window.start_timer()
            training_thread = threading.Thread(target=train_model.train_alexnet_model, args=(self.file_path, epochs, batch_size,
                                                                                             validation_split))
            training_thread.start()
        else:
            qtw.QMessageBox.warning(self, "Warning", "Please select a valid model and load data first.")

    def __init__(self, prev_window, file_path):
        super().__init__()

        self.prev_window = prev_window
        self.file_path = file_path
        self.setWindowTitle('Sign-SYS Training')
        self.setWindowIcon(QIcon('signsysweblogoturq.png'))
        self.setGeometry(950, 450, 720, 600)
        self.setStyleSheet('background-color: #8C52FF;')

        training_layout = qtw.QVBoxLayout()
        self.setLayout(training_layout)

        button_layout = qtw.QHBoxLayout()

        self.return_btn = qtw.QPushButton('Return')
        self.return_btn.setStyleSheet(activitystyles.button_style)
        self.return_btn.clicked.connect(self.returnToActivity)
        button_layout.addWidget(self.return_btn, alignment=Qt.AlignLeft)

        self.train_btn = qtw.QPushButton('Start Training')
        self.train_btn.setStyleSheet(activitystyles.button_style)
        self.train_btn.clicked.connect(self.startTraining)
        button_layout.addWidget(self.train_btn, alignment=Qt.AlignRight)

        top_layout = qtw.QVBoxLayout()
        top_layout.addLayout(button_layout)

        self.model_select = qtw.QLabel("Select model to train")
        self.model_select.setStyleSheet(trainingstyles.title_styles)
        top_layout.addWidget(self.model_select, alignment=Qt.AlignCenter)

        training_layout.addLayout(top_layout)

        self.cnn_dropdown = qtw.QComboBox()
        self.cnn_dropdown.addItems(["-- Select Model --", "AlexNet"])
        self.cnn_dropdown.setStyleSheet(trainingstyles.combobox_style)
        self.cnn_dropdown.setItemDelegate(CenterDropdownDelegate(self))
        training_layout.addWidget(self.cnn_dropdown, alignment=Qt.AlignTop)
        training_layout.addWidget(self.cnn_dropdown, alignment=Qt.AlignCenter)

        self.validation_text = qtw.QLabel("Select Train/Validation Ratio")
        self.validation_text.setStyleSheet(trainingstyles.title_styles)
        training_layout.addWidget(self.validation_text, alignment=Qt.AlignCenter)

        self.train_test_ratio_slider = qtw.QSlider(Qt.Horizontal)
        self.train_test_ratio_slider.setMinimum(1)
        self.train_test_ratio_slider.setMaximum(99)
        self.train_test_ratio_slider.setValue(50)
        self.train_test_ratio_slider.setTickPosition(qtw.QSlider.TicksBelow)
        self.train_test_ratio_slider.setTickInterval(10)
        self.train_test_ratio_slider.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Fixed)
        self.train_test_ratio_slider_label = qtw.QLabel(f'Train/Test Ratio: {self.train_test_ratio_slider.value()}%',
                                                        self)

        training_layout.addWidget(self.train_test_ratio_slider_label, alignment=Qt.AlignCenter)
        training_layout.addWidget(self.train_test_ratio_slider, alignment=Qt.AlignCenter)

        self.batchsize_text = qtw.QLabel("Select a batch size")
        self.batchsize_text.setStyleSheet(trainingstyles.title_styles)
        training_layout.addWidget(self.batchsize_text, alignment=Qt.AlignCenter)

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

        self.epochs_text = qtw.QLabel("Select amount of epochs")
        self.epochs_text.setStyleSheet(trainingstyles.title_styles)
        training_layout.addWidget(self.epochs_text, alignment=Qt.AlignCenter)

        self.epochs_slider = qtw.QSlider(Qt.Horizontal)
        self.epochs_slider.setMinimum(1)
        self.epochs_slider.setMaximum(30)
        self.epochs_slider.setValue(10)
        self.epochs_slider.setTickPosition(qtw.QSlider.TicksBelow)
        self.epochs_slider.setTickInterval(10)
        self.epochs_slider.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Fixed)
        self.epochs_slider_label = qtw.QLabel(f'Epochs: {self.epochs_slider.value()}', self)

        training_layout.addWidget(self.epochs_slider_label, alignment=Qt.AlignCenter)
        training_layout.addWidget(self.epochs_slider, alignment=Qt.AlignCenter)

        self.train_test_ratio_slider.valueChanged.connect(
            lambda value: self.train_test_ratio_slider_label.setText(f'Train/Test Ratio: {value}%'))
        self.batch_size_slider.valueChanged.connect(
            lambda value: self.batch_size_slider_label.setText(f'Batch Size: {value}'))
        self.epochs_slider.valueChanged.connect(lambda value: self.epochs_slider_label.setText(f'Epochs: {value}'))

        self.train_test_ratio_slider.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Fixed)
        self.batch_size_slider.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Fixed)
        self.epochs_slider.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Fixed)
