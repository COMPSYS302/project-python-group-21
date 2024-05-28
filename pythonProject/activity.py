import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt

from styles import ActivityStyles

activitystyles = ActivityStyles()


class ActivityOptionsWindow(qtw.QWidget):

    def returnToHome(self):
        self.prevWindow.show()
        self.close()

    def loadFile(self):
        fname = qtw.QFileDialog.getOpenFileName(self, "Open File", "",
                                                "All Files (*);; Python Files(*.py)")


    def __init__(self, previouswindow):
        super().__init__()

        self.prevWindow = previouswindow

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
        parent_layout.addStretch()


        self.load_data_button.clicked.connect(self.loadFile)
        self.show()


