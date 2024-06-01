# Styles for the Home Window
class HomeWindowStyles:
    def __init__(self):
        self.text_style = """
            QLabel {
                   font-size: 54px; 
                   color: white;     
                   font-weight: bold;
                   margin-top: 40px;
                   margin-bottom: 20px;
                   
            }
        """

        self.button_style = """
            QPushButton {
                    background-color: white;
                    color: #3D0289;
                    border-style: inset;
                    border-width: 2px;
                    border-radius: 5px;
                    border-color: beige;
                    font: bold 24px;
                    min-width: 10em;
                    padding: 10px;
                    margin-bottom: 40px;
            }
            
            QPushButton:hover {
                background-color: #3D0289;
                color: white
            }
            
            QPushButton:pressed {
                background-color: #545454;
            }
        """

# Styles for the Activity Window
class ActivityStyles:
    def __init__(self):
        self.text_styles = """
                    QLabel {
                        font-size: 36px; 
                        color: white;     
                        font-weight: bold;
                        margin-top: 20px;
                        margin-bottom: 20px;

                    }
                """
        self.button_style = """
                    QPushButton {
                            background-color: white;
                            color: #3D0289;
                            border-style: inset;
                            border-width: 2px;
                            border-radius: 5px;
                            border-color: beige;
                            font: bold 14px;
                            min-width: 4em;
                            padding: 10px;
                            margin:10px;
                    }

                    QPushButton:hover {
                        background-color: #3D0289;
                        color: white
                    }

                    QPushButton:pressed {
                        background-color: #545454;
                    }
                """

        self.loading_bar_style = """
                    QProgressBar {
                        border-style: solid;
                        color: white;
                        font: bold 12px;
                        border-width: 2px;
                        border-radius: 5px;
                        border-color: white;
                    }
                    
                    QProgressBar::chunk {
                        background : #3D0289;
                    }
                """

        self.line_edit_style = """
                    QLineEdit {
                        background-color: white;
                        border-radius: 10px;
                        padding: 5px 30px 5px 30px;
                        background: url(search_icon_30.png) no-repeat left;
                        background-size: 16px;
                    }
                """

# Styles for the Training Window
class TrainingStyles:
    def __init__(self):
        self.title_styles = """
                            QLabel {
                                font-size: 30px; 
                                color: white;     
                                font-weight: bold;
                            }
                        """
        self.combobox_style = """
                            QComboBox {
                                background-color: white;
                                color: #3D0289;
                                border-style: inset;
                                border-width: 2px;
                                border-radius: 10px;
                                border-color: beige;
                                font: bold 14px;
                                min-width: 10em;
                                padding: 5px;
                            }

                            QComboBox::drop-down {
                                subcontrol-origin: padding;
                                subcontrol-position: top right;
                                width: 20px;
                                border-left-width: 1px;
                                border-left-color: beige;
                                border-left-style: solid;
                                border-top-right-radius: 3px;
                                border-bottom-right-radius: 3px;
                            }

                            QComboBox::down-arrow {
                                image: url(dropdown.png);
                                width: 10px;
                                height: 10px;
                            }

                            QComboBox QAbstractItemView {
                                background: white;
                                color: #3D0289;
                                selection-background-color: #3D0289;
                                selection-color: white;
                            }
                        """
        self.text_styles = """
                                    QLabel {
                                        font-size: 20px; 
                                        color: white;     
                                        font-weight: bold;
                                    }
                                """