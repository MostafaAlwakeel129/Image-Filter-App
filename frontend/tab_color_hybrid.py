import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QSlider, QComboBox, QGroupBox,
    QSplitter, QTabWidget
)
from PyQt5.QtCore import Qt
import cv_backend
from Helpers.image_utils import bytes_to_mat, mat_to_bytes, set_label_image, set_status
from Helpers.styles import COMMON_QSS, TAB_BAR_QSS, STATUS_QSS, open_image_file


# ---------------------------------------------------------------------------
# Reusable image display widget
# ---------------------------------------------------------------------------

class ImageDisplayWidget(QWidget):
    """Reusable widget: title bar + image panel + info bar."""

    def __init__(self, title="Image"):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            font-weight: bold; font-size: 13px; color: #2c3e50;
            padding: 5px; background-color: #f5f5f5;
            border-top-left-radius: 8px; border-top-right-radius: 8px;
        """)
        layout.addWidget(self.title_label)

        self.image_label = QLabel("No Image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(300, 280)
        self.image_label.setScaledContents(False)
        self.image_label.setStyleSheet("""
            border: 2px solid #87ceeb; border-radius: 8px;
            background-color: #f5f5f5; color: #aaaaaa; font-size: 14px;
        """)
        layout.addWidget(self.image_label)

        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("""
            color: #87ceeb; font-size: 11px; font-family: monospace;
            padding: 5px; background-color: #f5f5f5;
            border-bottom-left-radius: 8px; border-bottom-right-radius: 8px;
        """)
        layout.addWidget(self.info_label)

    def set_image(self, data: bytes):
        if data is None:
            self.clear()
            return
        mat = bytes_to_mat(data)
        set_label_image(self.image_label, mat, max_w=300, max_h=300)
        h, w = mat.shape[:2]
        self.info_label.setText(f"{w}x{h} | {mat.dtype} | range: {mat.min()}–{mat.max()}")

    def clear(self):
        self.image_label.clear()
        self.image_label.setText("No Image")
        self.info_label.setText("")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_cutoff_label() -> QLabel:
    lbl = QLabel("30")
    lbl.setStyleSheet("""
        font-weight: bold; color: #87ceeb;
        padding: 2px 8px; background-color: #f5f5f5; border-radius: 4px;
    """)
    return lbl


def _make_status_label() -> QLabel:
    lbl = QLabel("")
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet(STATUS_QSS)
    return lbl


# ---------------------------------------------------------------------------
# Frequency Filter Tab
# ---------------------------------------------------------------------------

class FrequencyFilterTab(QWidget):
    """Low-pass and high-pass frequency domain filtering."""

    _FILTER_MAP = {
        "Low-Pass Filter":  "lowpass",
        "High-Pass Filter": "highpass",
    }
    _FILTER_INFO = {
        "Low-Pass Filter":  "Low-pass: smooths image (removes high frequencies)",
        "High-Pass Filter": "High-pass: enhances edges (removes low frequencies)",
    }

    def __init__(self):
        super().__init__()
        self.setStyleSheet(COMMON_QSS)
        self.image_bytes    = None
        self.filtered_bytes = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.addWidget(self._build_control_panel())

        splitter = QSplitter(Qt.Horizontal)
        self.original_display  = ImageDisplayWidget("Original Image")
        self.filtered_display  = ImageDisplayWidget("Filtered Image")
        self.spectrum_display  = ImageDisplayWidget("Frequency Spectrum")
        for w in (self.original_display, self.filtered_display, self.spectrum_display):
            splitter.addWidget(w)
        splitter.setSizes([400, 400, 400])
        layout.addWidget(splitter)

        layout.addWidget(self._build_filter_controls())

        self._status = _make_status_label()
        layout.addWidget(self._status)

    def _build_control_panel(self):
        panel  = QWidget()
        layout = QHBoxLayout(panel)
        layout.setSpacing(15)

        self.btn_load = QPushButton("📂  Load Image")
        self.btn_load.clicked.connect(self._load_image)
        layout.addWidget(self.btn_load)

        layout.addWidget(QLabel("Filter Type:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(list(self._FILTER_MAP.keys()))
        self.filter_combo.currentTextChanged.connect(self._on_filter_changed)
        layout.addWidget(self.filter_combo)

        self.btn_reset = QPushButton("↺  Reset")
        self.btn_reset.clicked.connect(self._reset)
        self.btn_reset.setEnabled(False)
        layout.addWidget(self.btn_reset)

        layout.addStretch()
        return panel

    def _build_filter_controls(self):
        panel  = QGroupBox("Filter Parameters")
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        row = QHBoxLayout()
        row.addWidget(QLabel("Cutoff Frequency:"))
        self.cutoff_slider = QSlider(Qt.Horizontal)
        self.cutoff_slider.setRange(5, 100)
        self.cutoff_slider.setValue(30)
        self.cutoff_slider.valueChanged.connect(self._on_cutoff_changed)
        row.addWidget(self.cutoff_slider)
        self.cutoff_label = _make_cutoff_label()
        row.addWidget(self.cutoff_label)
        layout.addLayout(row)

        self.filter_info = QLabel(
            "Low-pass: smooths image (removes high frequencies)\n"
            "High-pass: enhances edges (removes low frequencies)"
        )
        self.filter_info.setStyleSheet(
            "color: #aaaaaa; font-style: italic; padding: 8px; "
            "background-color: #f5f5f5; border-radius: 5px;"
        )
        layout.addWidget(self.filter_info)
        return panel

    # Slots
    def _load_image(self):
        mat, fname = open_image_file(self)
        if mat is None:
            if fname != "":
                set_status(self._status, "❌  Failed to load image.", error=True)
            return
        self.image_bytes = mat_to_bytes(mat)
        self.original_display.set_image(self.image_bytes)
        self.spectrum_display.set_image(cv_backend.get_spectrum(self.image_bytes))
        self.btn_reset.setEnabled(True)
        set_status(self._status, f"✅  Loaded: {fname}")
        self._apply_filter()

    def _on_filter_changed(self, text: str):
        self.filter_info.setText(self._FILTER_INFO.get(text, ""))
        self._apply_filter()

    def _on_cutoff_changed(self, value: int):
        self.cutoff_label.setText(str(value))
        self._apply_filter()

    def _apply_filter(self):
        if not self.image_bytes:
            return
        try:
            filter_key = self._FILTER_MAP[self.filter_combo.currentText()]
            cutoff = float(self.cutoff_slider.value())

            if filter_key == "lowpass":
                self.filtered_bytes = cv_backend.lowpass_filter(
                    self.image_bytes, cutoff
                )
            else:
                self.filtered_bytes = cv_backend.highpass_filter(
                    self.image_bytes, cutoff
                )

            # ✅ update filtered image
            self.filtered_display.set_image(self.filtered_bytes)

            # ✅ update spectrum OF FILTERED IMAGE
            spectrum = cv_backend.get_spectrum(self.filtered_bytes)
            self.spectrum_display.set_image(spectrum)

        except Exception as e:
            set_status(self._status, f"❌  Error: {e}", error=True)
    def _reset(self):
        if not self.image_bytes:
            return
        self.cutoff_slider.setValue(30)
        self.filter_combo.setCurrentIndex(0)
        self._apply_filter()


# ---------------------------------------------------------------------------
# Hybrid Image Tab
# ---------------------------------------------------------------------------

class HybridImageTab(QWidget):
    """Create hybrid images from two inputs."""

    def __init__(self):
        super().__init__()
        self.setStyleSheet(COMMON_QSS)
        self.low_freq_bytes  = None
        self.high_freq_bytes = None
        self.hybrid_bytes    = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.addWidget(self._build_control_panel())

        splitter = QSplitter(Qt.Horizontal)
        self.low_display    = ImageDisplayWidget("Low-Freq Image\n(background)")
        self.high_display   = ImageDisplayWidget("High-Freq Image\n(details)")
        self.hybrid_display = ImageDisplayWidget("Hybrid Result")
        for w in (self.low_display, self.high_display, self.hybrid_display):
            splitter.addWidget(w)
        splitter.setSizes([400, 400, 400])
        layout.addWidget(splitter)

        layout.addWidget(self._build_hybrid_controls())

        self._status = _make_status_label()
        layout.addWidget(self._status)

    def _build_control_panel(self):
        panel  = QWidget()
        layout = QHBoxLayout(panel)
        layout.setSpacing(15)

        self.btn_load_low = QPushButton("📂  Load Low-Freq Image")
        self.btn_load_low.clicked.connect(lambda: self._load_image("low"))
        layout.addWidget(self.btn_load_low)

        self.btn_load_high = QPushButton("📂  Load High-Freq Image")
        self.btn_load_high.clicked.connect(lambda: self._load_image("high"))
        layout.addWidget(self.btn_load_high)

        self.btn_create = QPushButton("✨  Create Hybrid Image")
        self.btn_create.clicked.connect(self._create_hybrid)
        self.btn_create.setEnabled(False)
        layout.addWidget(self.btn_create)

        self.btn_save = QPushButton("💾  Save Result")
        self.btn_save.clicked.connect(self._save_result)
        self.btn_save.setEnabled(False)
        layout.addWidget(self.btn_save)

        layout.addStretch()
        return panel

    def _build_hybrid_controls(self):
        panel  = QGroupBox("Hybrid Parameters")
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        row = QHBoxLayout()
        row.addWidget(QLabel("Frequency Cutoff:"))
        self.cutoff_slider = QSlider(Qt.Horizontal)
        self.cutoff_slider.setRange(5, 50)
        self.cutoff_slider.setValue(30)
        self.cutoff_slider.valueChanged.connect(self._on_cutoff_changed)
        row.addWidget(self.cutoff_slider)
        self.cutoff_label = _make_cutoff_label()
        row.addWidget(self.cutoff_label)
        layout.addLayout(row)

        instructions = QLabel(
            "1. Load an image for low frequencies (background)\n"
            "2. Load another image for high frequencies (details)\n"
            "3. Adjust the cutoff to control the blend\n"
            "4. Click 'Create Hybrid Image'\n\n"
            "Images with different sizes are resized automatically."
        )
        instructions.setStyleSheet(
            "color: #aaaaaa; font-style: italic; padding: 10px; "
            "background-color: #f5f5f5; border-radius: 5px;"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        return panel

    # Slots
    def _load_image(self, image_type: str):
        mat, fname = open_image_file(self, flags=cv2.IMREAD_GRAYSCALE)
        if mat is None:
            if fname != "":
                set_status(self._status, "❌  Failed to load image.", error=True)
            return
        img_bytes = mat_to_bytes(mat)
        if image_type == "low":
            self.low_freq_bytes = img_bytes
            self.low_display.set_image(img_bytes)
        else:
            self.high_freq_bytes = img_bytes
            self.high_display.set_image(img_bytes)
        set_status(self._status, f"✅  Loaded {image_type}-freq: {fname}")
        self._check_ready()

    def _check_ready(self):
        if self.low_freq_bytes is not None and self.high_freq_bytes is not None:
            self.btn_create.setEnabled(True)

    def _on_cutoff_changed(self, value: int):
        self.cutoff_label.setText(str(value))
        if self.low_freq_bytes is not None and self.high_freq_bytes is not None:
            self._create_hybrid()

    def _create_hybrid(self):
        if not self.low_freq_bytes or not self.high_freq_bytes:
            return
        try:
            cutoff = float(self.cutoff_slider.value())
            self.hybrid_bytes = cv_backend.create_hybrid_image(
                self.low_freq_bytes, self.high_freq_bytes, cutoff
            )
            self.hybrid_display.set_image(self.hybrid_bytes)
            self.btn_save.setEnabled(True)
            set_status(self._status, f"✅  Hybrid image created (cutoff={cutoff:.0f}).")
        except Exception as e:
            set_status(self._status, f"❌  Error: {e}", error=True)

    def _save_result(self):
        if not self.hybrid_bytes:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Hybrid Image", "", "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)"
        )
        if path:
            cv2.imwrite(path, bytes_to_mat(self.hybrid_bytes))
            set_status(self._status, f"✅  Saved to {path.split('/')[-1]}")


# ---------------------------------------------------------------------------
# Container tab
# ---------------------------------------------------------------------------

class ColorHybridTab(QWidget):
    """Container tab holding both the filter and hybrid sub-tabs."""

    def __init__(self):
        super().__init__()
        self.setStyleSheet(COMMON_QSS + TAB_BAR_QSS)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        tab_widget = QTabWidget()
        tab_widget.addTab(FrequencyFilterTab(), "Frequency Domain Filters")
        tab_widget.addTab(HybridImageTab(),     "Hybrid Images")
        layout.addWidget(tab_widget)