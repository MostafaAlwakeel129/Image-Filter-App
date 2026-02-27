import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from tab_noise_filters import NoiseTab
from tab_edge_freq import EdgeTab
from tab_hist_contrast import HistogramContrastTab
from tab_color_hybrid import FrequencyFilterTab, HybridImageTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Computer Vision Suite")
        self.resize(1920, 1080)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tabs.addTab(NoiseTab(),             "1. Noise & Filters")
        self.tabs.addTab(EdgeTab(),              "2. Edge Detection")
        self.tabs.addTab(HistogramContrastTab(), "3. Histogram & Contrast")
        self.tabs.addTab(FrequencyFilterTab(),   "4. Frequency Domain Filters")
        self.tabs.addTab(HybridImageTab(),       "5. Hybrid Images")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())