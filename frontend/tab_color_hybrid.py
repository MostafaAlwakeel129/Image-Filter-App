import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QSlider, QComboBox, QGroupBox,
    QSplitter, QTabWidget
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv_backend


class ImageDisplayWidget(QWidget):
    """Reusable widget for displaying a grayscale image with a title and info bar."""

    def __init__(self, title="Image"):
        super().__init__()
        layout = QVBoxLayout()

        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)

        self.image_label = QLabel("No Image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(300, 300)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        layout.addWidget(self.image_label)

        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)

        self.setLayout(layout)

    def set_image(self, img: np.ndarray):
        """Display a numpy image (grayscale or BGR — always shown as gray)."""
        if img is None:
            self.clear()
            return

        # Normalise to grayscale regardless of what was passed in
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Ensure contiguous memory — QImage requires it
        gray = np.ascontiguousarray(gray)

        h, w = gray.shape
        qimg = QImage(gray.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)

        scaled = pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        self.info_label.setText(
            f"{w}x{h} | {gray.dtype} | range: {gray.min()}–{gray.max()}"
        )

    def clear(self):
        self.image_label.clear()
        self.image_label.setText("No Image")
        self.info_label.setText("")


# ─────────────────────────────────────────────────────────────────────────────

class FrequencyFilterTab(QWidget):
    """Low-pass and high-pass frequency domain filtering."""

    # Maps combo-box text → backend function
    _FILTER_MAP = {
        "Low-Pass Filter":  "lowpass",
        "High-Pass Filter": "highpass",
    }

    def __init__(self):
        super().__init__()
        self.image          = None
        self.filtered_image = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.addWidget(self._build_control_panel())

        splitter = QSplitter(Qt.Horizontal)
        self.original_display = ImageDisplayWidget("Original Image")
        self.filtered_display = ImageDisplayWidget("Filtered Image")
        self.spectrum_display = ImageDisplayWidget("Frequency Spectrum")
        for w in (self.original_display, self.filtered_display, self.spectrum_display):
            splitter.addWidget(w)
        splitter.setSizes([400, 400, 400])
        layout.addWidget(splitter)

        layout.addWidget(self._build_filter_controls())
        self.setLayout(layout)

    def _build_control_panel(self):
        panel = QWidget()
        layout = QHBoxLayout()

        self.btn_load = QPushButton("Load Image")
        self.btn_load.clicked.connect(self._load_image)
        layout.addWidget(self.btn_load)

        layout.addWidget(QLabel("Filter Type:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(list(self._FILTER_MAP.keys()))
        self.filter_combo.currentTextChanged.connect(self._on_filter_changed)
        layout.addWidget(self.filter_combo)

        self.btn_apply = QPushButton("Apply Filter")
        self.btn_apply.clicked.connect(self._apply_filter)
        self.btn_apply.setEnabled(False)
        layout.addWidget(self.btn_apply)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self._reset)
        self.btn_reset.setEnabled(False)
        layout.addWidget(self.btn_reset)

        layout.addStretch()
        panel.setLayout(layout)
        return panel

    def _build_filter_controls(self):
        panel = QGroupBox("Filter Parameters")
        layout = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel("Cutoff Frequency:"))

        self.cutoff_slider = QSlider(Qt.Horizontal)
        self.cutoff_slider.setRange(5, 100)
        self.cutoff_slider.setValue(30)
        self.cutoff_slider.valueChanged.connect(self._on_cutoff_changed)
        row.addWidget(self.cutoff_slider)

        self.cutoff_label = QLabel("30")
        row.addWidget(self.cutoff_label)
        layout.addLayout(row)

        self.filter_info = QLabel(
            "Low-pass: smooths image (removes high frequencies)\n"
            "High-pass: enhances edges (removes low frequencies)"
        )
        self.filter_info.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.filter_info)

        panel.setLayout(layout)
        return panel

    # ── slots ──────────────────────────────────────────────────────────────

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.bmp *.tif)"
        )
        if not path:
            return

        self.image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        self.original_display.set_image(self.image)
        self.spectrum_display.set_image(cv_backend.get_spectrum(self.image))

        self.btn_apply.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self._apply_filter()

    def _on_filter_changed(self, text: str):
        info = {
            "Low-Pass Filter":  "Low-pass: smooths image (removes high frequencies)",
            "High-Pass Filter": "High-pass: enhances edges (removes low frequencies)",
        }
        self.filter_info.setText(info.get(text, ""))
        self._apply_filter()

    def _on_cutoff_changed(self, value: int):
        self.cutoff_label.setText(str(value))
        self._apply_filter()

    def _apply_filter(self):
        if self.image is None:
            return

        filter_key = self._FILTER_MAP[self.filter_combo.currentText()]
        cutoff     = float(self.cutoff_slider.value())

        if filter_key == "lowpass":
            self.filtered_image = cv_backend.lowpass_filter(self.image, cutoff)
        else:
            self.filtered_image = cv_backend.highpass_filter(self.image, cutoff)

        self.filtered_display.set_image(self.filtered_image)

    def _reset(self):
        if self.image is None:
            return
        self.cutoff_slider.setValue(30)
        self.filter_combo.setCurrentIndex(0)
        # Re-apply so filtered_image stays consistent with the UI state
        self._apply_filter()


# ─────────────────────────────────────────────────────────────────────────────

class HybridImageTab(QWidget):
    """Create hybrid images from two grayscale inputs."""

    def __init__(self):
        super().__init__()
        self.low_freq_image  = None
        self.high_freq_image = None
        self.hybrid_image    = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
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
        self.setLayout(layout)

    def _build_control_panel(self):
        panel = QWidget()
        layout = QHBoxLayout()

        self.btn_load_low = QPushButton("Load Low-Freq Image")
        self.btn_load_low.clicked.connect(lambda: self._load_image("low"))
        layout.addWidget(self.btn_load_low)

        self.btn_load_high = QPushButton("Load High-Freq Image")
        self.btn_load_high.clicked.connect(lambda: self._load_image("high"))
        layout.addWidget(self.btn_load_high)

        self.btn_create = QPushButton("Create Hybrid Image")
        self.btn_create.clicked.connect(self._create_hybrid)
        self.btn_create.setEnabled(False)
        layout.addWidget(self.btn_create)

        self.btn_save = QPushButton("Save Result")
        self.btn_save.clicked.connect(self._save_result)
        self.btn_save.setEnabled(False)
        layout.addWidget(self.btn_save)

        layout.addStretch()
        panel.setLayout(layout)
        return panel

    def _build_hybrid_controls(self):
        panel = QGroupBox("Hybrid Parameters")
        layout = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel("Frequency Cutoff:"))

        self.cutoff_slider = QSlider(Qt.Horizontal)
        self.cutoff_slider.setRange(5, 50)
        self.cutoff_slider.setValue(30)
        self.cutoff_slider.valueChanged.connect(self._on_cutoff_changed)
        row.addWidget(self.cutoff_slider)

        self.cutoff_label = QLabel("30")
        row.addWidget(self.cutoff_label)
        layout.addLayout(row)

        instructions = QLabel(
            "1. Load an image for low frequencies (background)\n"
            "2. Load another image for high frequencies (details)\n"
            "3. Adjust the cutoff to control the blend\n"
            "4. Click 'Create Hybrid Image'\n\n"
            "Images with different sizes are resized automatically."
        )
        instructions.setStyleSheet("color: gray; font-style: italic; padding: 5px;")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        panel.setLayout(layout)
        return panel

    # ── slots ──────────────────────────────────────────────────────────────

    def _load_image(self, image_type: str):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Open {image_type}-frequency image", "",
            "Images (*.png *.jpg *.bmp *.tif)"
        )
        if not path:
            return

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if image_type == "low":
            self.low_freq_image = img
            self.low_display.set_image(img)
        else:
            self.high_freq_image = img
            self.high_display.set_image(img)

        self._check_ready()

    def _check_ready(self):
        """Enable the Create button once both images are loaded.
        If sizes differ, resize the smaller one to match the larger."""
        if self.low_freq_image is None or self.high_freq_image is None:
            return

        lh, lw = self.low_freq_image.shape
        hh, hw = self.high_freq_image.shape

        if (lh, lw) != (hh, hw):
            # Resize both to the larger of the two dimensions
            target = (max(lw, hw), max(lh, hh))
            self.low_freq_image  = cv2.resize(self.low_freq_image,  target)
            self.high_freq_image = cv2.resize(self.high_freq_image, target)
            self.low_display.set_image(self.low_freq_image)
            self.high_display.set_image(self.high_freq_image)
            self.low_display.info_label.setText(
                self.low_display.info_label.text() + "  [resized]"
            )
            self.high_display.info_label.setText(
                self.high_display.info_label.text() + "  [resized]"
            )

        self.btn_create.setEnabled(True)

    def _on_cutoff_changed(self, value: int):
        self.cutoff_label.setText(str(value))
        # Live update only when both images are already loaded
        if self.low_freq_image is not None and self.high_freq_image is not None:
            self._create_hybrid()

    def _create_hybrid(self):
        if self.low_freq_image is None or self.high_freq_image is None:
            return

        cutoff = float(self.cutoff_slider.value())
        try:
            self.hybrid_image = cv_backend.create_hybrid_image(
                self.low_freq_image, self.high_freq_image, cutoff
            )
            self.hybrid_display.set_image(self.hybrid_image)
            self.btn_save.setEnabled(True)
        except Exception as e:
            self.hybrid_display.image_label.setText(f"Error: {e}")

    def _save_result(self):
        if self.hybrid_image is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Hybrid Image", "",
            "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)"
        )
        if path:
            cv2.imwrite(path, self.hybrid_image)


# ─────────────────────────────────────────────────────────────────────────────

class ColorHybridTab(QWidget):
    """Container tab holding both the filter and hybrid sub-tabs."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        tab_widget = QTabWidget()
        tab_widget.addTab(FrequencyFilterTab(), "Frequency Domain Filters")
        tab_widget.addTab(HybridImageTab(),     "Hybrid Images")

        layout.addWidget(tab_widget)
        self.setLayout(layout)