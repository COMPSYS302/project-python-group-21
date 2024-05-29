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