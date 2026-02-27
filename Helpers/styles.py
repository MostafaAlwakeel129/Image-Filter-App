"""
Shared Qt stylesheet and base tab class for the Computer Vision Suite.
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QFileDialog, QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontDatabase, QFont

import cv2

from Helpers.image_utils import mat_to_bytes, bytes_to_mat, set_label_image, set_status

# Image file-filter used in every open dialog
_IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp)"


def open_image_file(parent, flags=cv2.IMREAD_COLOR):
    """
    Show a standard Open Image dialog and load the chosen file with cv2.

    Parameters
    ----------
    parent : QWidget
        Parent window for the dialog.
    flags : int
        cv2.imread flags (default: IMREAD_COLOR).
        Pass cv2.IMREAD_GRAYSCALE for greyscale-only tabs.

    Returns
    -------
    (mat, filename) on success, or (None, '') if the user cancelled or the
    file could not be read.
    """
    path, _ = QFileDialog.getOpenFileName(parent, "Open Image", "", _IMAGE_FILTER)
    if not path:
        return None, ""
    mat = cv2.imread(path, flags)
    if mat is None:
        return None, ""
    return mat, path.split("/")[-1]


# ---------------------------------------------------------------------------
# Font loading
# ---------------------------------------------------------------------------

# Paths to Poppins TTF files (present on most Linux systems with google-fonts)
_POPPINS_PATHS = [
    "/usr/share/fonts/truetype/google-fonts/Poppins-Regular.ttf",
    "/usr/share/fonts/truetype/google-fonts/Poppins-Medium.ttf",
    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
    "/usr/share/fonts/truetype/google-fonts/Poppins-Light.ttf",
]

# Windows / macOS fallback stack (no loading needed — just referenced in QSS)
_FONT_FAMILY = "Poppins"
_FONT_FALLBACK = "Segoe UI, SF Pro Display, Helvetica Neue, Arial"


def load_app_font(app) -> str:
    """
    Register Poppins with Qt's font database and apply it application-wide.
    Falls back gracefully to system sans-serif fonts if files are not found.
    Returns the font family name that was applied.
    """
    db = QFontDatabase()
    registered = False
    for path in _POPPINS_PATHS:
        try:
            if db.addApplicationFont(path) != -1:
                registered = True
        except Exception:
            pass

    family = _FONT_FAMILY if registered else _FONT_FALLBACK.split(",")[0].strip()
    font = QFont(family)
    font.setPointSize(10)
    font.setHintingPreference(QFont.PreferFullHinting)
    app.setFont(font)
    return family


# ---------------------------------------------------------------------------
# Shared stylesheet
# ---------------------------------------------------------------------------

COMMON_QSS = """
    * {
        font-family: "Poppins", "Segoe UI", "SF Pro Display", "Helvetica Neue", Arial, sans-serif;
    }
    QGroupBox {
        font-weight: 600;
        font-size: 12px;
        letter-spacing: 0.3px;
        border: 2px solid #87ceeb;
        border-radius: 8px;
        margin-top: 12px;
        padding-top: 10px;
        background-color: white;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 10px 0 10px;
        color: #2c3e50;
    }
    QPushButton {
        background-color: #87ceeb;
        color: white;
        border: none;
        padding: 8px 15px;
        border-radius: 5px;
        font-weight: 600;
        font-size: 11px;
        letter-spacing: 0.4px;
        min-width: 80px;
    }
    QPushButton:hover {
        background-color: #98d8c8;
    }
    QPushButton:pressed {
        background-color: #aaaaaa;
    }
    QPushButton:disabled {
        background-color: #bbbbbb;
        color: #666666;
    }
    QComboBox {
        padding: 5px 10px;
        border: 2px solid #87ceeb;
        border-radius: 5px;
        background-color: white;
        min-width: 150px;
    }
    QComboBox:hover {
        border-color: #98d8c8;
    }
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
        border-left-width: 1px;
        border-left-color: #87ceeb;
        border-left-style: solid;
    }
    QSpinBox {
        padding: 5px;
        border: 2px solid #87ceeb;
        border-radius: 5px;
        background-color: white;
    }
    QSpinBox:hover {
        border-color: #98d8c8;
    }
    QSlider::groove:horizontal {
        border: 1px solid #bbbbbb;
        height: 6px;
        background: #f5f5f5;
        margin: 2px 0;
        border-radius: 3px;
    }
    QSlider::handle:horizontal {
        background: #87ceeb;
        border: 2px solid #98d8c8;
        width: 16px;
        height: 16px;
        margin: -5px 0;
        border-radius: 8px;
    }
    QSlider::handle:horizontal:hover {
        background: #98d8c8;
    }
    QRadioButton {
        color: #2c3e50;
        font-size: 11px;
        letter-spacing: 0.2px;
        spacing: 8px;
    }
    QRadioButton::indicator {
        width: 16px;
        height: 16px;
    }
    QRadioButton::indicator:checked {
        background-color: #87ceeb;
        border: 2px solid #98d8c8;
        border-radius: 8px;
    }
    QCheckBox {
        color: #2c3e50;
        font-size: 11px;
        letter-spacing: 0.2px;
        spacing: 8px;
    }
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
        border: 2px solid #87ceeb;
        border-radius: 4px;
    }
    QCheckBox::indicator:checked {
        background-color: #87ceeb;
    }
    QLabel {
        color: #2c3e50;
        font-size: 11px;
    }
    QFrame {
        background-color: transparent;
    }
    QSplitter::handle {
        background-color: #87ceeb;
        width: 2px;
    }
"""

TAB_BAR_QSS = """
    QTabWidget::pane {
        border: 2px solid #87ceeb;
        border-radius: 8px;
        background-color: white;
    }
    QTabBar::tab {
        background-color: #bbbbbb;
        color: #2c3e50;
        padding: 8px 16px;
        margin-right: 4px;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        font-weight: bold;
    }
    QTabBar::tab:selected {
        background-color: #87ceeb;
        color: white;
    }
    QTabBar::tab:hover:!selected {
        background-color: #98d8c8;
        color: white;
    }
"""

STATUS_QSS = (
    "color: #aaaaaa; font-style: italic; padding: 8px; "
    "background-color: white; border-radius: 5px;"
)


# ---------------------------------------------------------------------------
# Base tab with shared image-row and open/status helpers
# ---------------------------------------------------------------------------

class BaseImageTab(QWidget):
    """
    Provides:
      • _build_image_row()  — open button + Original/Processed panels
      • _open_image()       — file dialog → loads into self._original_bytes
      • _set_status()       — thin wrapper around set_status()
    Subclasses must call super().__init__() and may override _on_image_loaded().
    """

    # Fixed pixel dimensions for both image panels — subclasses can override
    IMAGE_W = 380
    IMAGE_H = 280

    def __init__(self):
        super().__init__()
        self._original_bytes: bytes | None = None
        self.setStyleSheet(COMMON_QSS)

    # ------------------------------------------------------------------
    # Image row
    # ------------------------------------------------------------------

    def _build_image_row(self) -> QWidget:
        frame = QFrame()
        layout = QHBoxLayout(frame)
        layout.setSpacing(16)

        self._open_btn = QPushButton("📂  Open Image")
        self._open_btn.setFixedWidth(130)
        self._open_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self._open_btn.clicked.connect(self._open_image)
        layout.addWidget(self._open_btn, alignment=Qt.AlignVCenter)

        orig_box = QGroupBox("Original")
        orig_layout = QVBoxLayout(orig_box)
        self._orig_label = QLabel("No image loaded")
        self._orig_label.setAlignment(Qt.AlignCenter)
        self._orig_label.setFixedSize(self.IMAGE_W, self.IMAGE_H)
        self._orig_label.setScaledContents(False)
        self._orig_label.setStyleSheet(
            "background-color: white; border: 2px solid #87ceeb; "
            "border-radius: 6px; color: #aaaaaa; font-size: 14px;"
        )
        orig_layout.addWidget(self._orig_label)
        layout.addWidget(orig_box)

        proc_box = QGroupBox("Processed")
        proc_layout = QVBoxLayout(proc_box)
        self._proc_label = QLabel("No image loaded")
        self._proc_label.setAlignment(Qt.AlignCenter)
        self._proc_label.setFixedSize(self.IMAGE_W, self.IMAGE_H)
        self._proc_label.setScaledContents(False)
        self._proc_label.setStyleSheet(
            "background-color: white; border: 2px solid #98d8c8; "
            "border-radius: 6px; color: #aaaaaa; font-size: 14px;"
        )
        proc_layout.addWidget(self._proc_label)
        layout.addWidget(proc_box)

        return frame

    # ------------------------------------------------------------------
    # Open image
    # ------------------------------------------------------------------

    def _open_image(self):
        mat, fname = open_image_file(self)
        if mat is None:
            if fname == "":   # user cancelled — stay silent
                return
            self._set_status("❌  Failed to load image.", error=True)
            return
        self._original_bytes = mat_to_bytes(mat)
        set_label_image(self._orig_label, mat, max_w=self.IMAGE_W, max_h=self.IMAGE_H)
        set_label_image(self._proc_label, mat, max_w=self.IMAGE_W, max_h=self.IMAGE_H)
        self._set_status(f"✅  Loaded: {fname}")
        self._on_image_loaded(mat)

    def _on_image_loaded(self, mat):
        """Hook for subclasses — called after a new image is loaded."""
        pass

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _set_status(self, msg: str, error: bool = False):
        set_status(self._status, msg, error)