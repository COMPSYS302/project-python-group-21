import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt

from activity import ActivityOptionsWindow
from styles import HomeWindowStyles

homewindowstyles = HomeWindowStyles()


class HomeWindow(qtw.QWidget):

    # function for enter button
    def openWhenEnter(self):
        self.window = ActivityOptionsWindow(self)
        self.hide()

    # ------------------------------------------------

    def __init__(self):
        # super definition is for QWidget class
        super().__init__()


        # Adding a title
        self.window = None
        self.setWindowTitle("Sign-Sys Application")
        self.setWindowIcon(QIcon('signsysweblogoturq.png'))

        # Setting geometry of the window
        self.setGeometry(950, 450, 720, 540)

        # Changing bg color to #31B1C8
        self.setStyleSheet("background-color: #8C52FF;")

        # Set Vertical layout
        main_layout = qtw.QVBoxLayout()
        self.setLayout(main_layout)

        # creating logo label
        self.logo_label = qtw.QLabel(self)

        # loading image
        self.pixmap = QPixmap('signsyspurple.png')

        # adding image to label
        self.logo_label.setPixmap(self.pixmap)

        # Optional, resize label to image size
        self.logo_label.resize(self.pixmap.width(),
                               self.pixmap.height())

        # Creating label for the text
        self.welcome_text_label = qtw.QLabel("Welcome to", self)
        self.welcome_text_label.setAlignment(Qt.AlignCenter)
        self.welcome_text_label.setStyleSheet(homewindowstyles.text_style)

        # Creating a button where the variable name is enter_button
        """
        This is a multi-line comment
        """

        # creating an enter button
        self.enter_button = qtw.QPushButton("Enter")
        self.enter_button.setStyleSheet(homewindowstyles.button_style)
        self.enter_button.clicked.connect(self.openWhenEnter)

        # performing center alignment on widgets
        main_layout.addWidget(self.welcome_text_label, alignment=Qt.AlignCenter)
        main_layout.addWidget(self.logo_label, alignment=Qt.AlignCenter)
        main_layout.addWidget(self.enter_button, alignment=Qt.AlignCenter)

        # qtRectangle = self.frameGeometry()
        # centerPoint = qtw.QDesktopWidget().availableGeometry().center()
        # qtRectangle.moveCenter(centerPoint)
        # self.move(qtRectangle.topLeft())
        self.show()


# intialising app first. note that this is only needed for the first (main) window
app = qtw.QApplication([])

# creating object of class HomeWindow() which initialises the ui
# since the ui is in the __init__ method that is automatically
# executed when creating the respective object
home_w = HomeWindow()

# executing application when the file is run
app.exec_()
