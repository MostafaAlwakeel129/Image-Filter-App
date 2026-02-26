import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from tab_noise_filters import NoiseTab # import your tab here
from tab_edge_freq import EdgeTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Computer Vision Suite")
        self.resize(1920, 1080)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tabs.addTab(NoiseTab(), "1. Noise & Filters") # then add it to the intialization here
        self.tabs.addTab(EdgeTab(), "2. Edge Detection")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())