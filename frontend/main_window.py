import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget,
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QShortcut
)
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt
from tab_noise_filters import NoiseTab
from tab_edge_freq import EdgeTab
from tab_hist_contrast import HistogramContrastTab
from tab_color_hybrid import ColorHybridTab
from Helpers.styles import load_app_font
from Helpers.undo_manager import UndoManager


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Computer Vision Suite")
        self.resize(700, 750)

        self.setStyleSheet("""
            * {
                font-family: "Poppins", "Segoe UI", "SF Pro Display", "Helvetica Neue", Arial, sans-serif;
            }
            QMainWindow { background-color: #f5f5f5; }
            QTabWidget::pane {
                border: 2px solid #87ceeb;
                border-radius: 8px;
                background-color: white;
                margin: 2px;
            }
            QTabBar::tab {
                background-color: #bbbbbb;
                color: #2c3e50;
                padding: 10px 20px;
                margin-right: 4px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: 600;
                font-size: 11px;
                letter-spacing: 0.3px;
            }
            QTabBar::tab:selected  { background-color: #87ceeb; color: white; }
            QTabBar::tab:hover:!selected { background-color: #98d8c8; color: white; }
        """)

        # ── Central widget ──────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Tab widget ──────────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.tabs.addTab(NoiseTab(),             "1. Noise & Filters")
        self.tabs.addTab(EdgeTab(),              "2. Edge Detection")
        self.tabs.addTab(HistogramContrastTab(), "3. Histogram & Contrast")
        self.tabs.addTab(ColorHybridTab(),       "4. Color & Hybrid")

        # ── Undo button placed in the top-right corner of the tab bar ───
        self._undo_btn = QPushButton("↩  Undo")
        self._undo_btn.setEnabled(False)
        self._undo_btn.setToolTip("Nothing to undo")
        self._undo_btn.setStyleSheet("""
            QPushButton {
                background-color: #87ceeb;
                color: white;
                border: none;
                padding: 6px 14px;
                border-radius: 5px;
                font-weight: 700;
                font-size: 11px;
                letter-spacing: 0.4px;
            }
            QPushButton:hover   { background-color: #98d8c8; }
            QPushButton:pressed { background-color: #aaaaaa; }
            QPushButton:disabled { background-color: #bbbbbb; color: #666; }
        """)
        self._undo_btn.clicked.connect(UndoManager.undo)

        # setCornerWidget places the widget flush with the tab bar on the right
        self.tabs.setCornerWidget(self._undo_btn, Qt.TopRightCorner)

        root.addWidget(self.tabs)

        # ── Register undo button with the manager ───────────────────────
        UndoManager.set_button(self._undo_btn)

        # ── Ctrl+Z shortcut ─────────────────────────────────────────────
        shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        shortcut.activated.connect(UndoManager.undo)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    load_app_font(app)   # registers Poppins and sets it as the app-wide font
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())