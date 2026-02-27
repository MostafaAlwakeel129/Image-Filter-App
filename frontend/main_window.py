import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget

from tab_noise_filters import NoiseTab
from tab_color_hybrid import FrequencyFilterTab, HybridImageTab
from tab_hist_contrast import HistogramContrastTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Computer Vision Suite")
        self.resize(700, 700)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tabs.addTab(NoiseTab(),             "Noise & Filters")
        self.tabs.addTab(HistogramContrastTab(), "Histogram & Contrast")
        self.tabs.addTab(FrequencyFilterTab(),   "Frequency Domain Filters")
        self.tabs.addTab(HybridImageTab(),       "Hybrid Images")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    