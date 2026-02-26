import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QRadioButton, QButtonGroup, QComboBox,
    QSpinBox, QFileDialog, QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt

import cv_backend
from Helpers.image_utils import bytes_to_mat, mat_to_bytes, set_label_image, set_status

# ---------------------------------------------------------------------------
# Predefined noise presets
# ---------------------------------------------------------------------------

NOISE_PRESETS = {
    "Uniform": [
        {"label": "Subtle  (±10)",    "low": -10,  "high": 10},
        {"label": "Moderate (±30)",   "low": -30,  "high": 30},
        {"label": "Strong  (±60)",    "low": -60,  "high": 60},
    ],
    "Gaussian": [
        {"label": "Light   (σ = 10)", "mean": 0.0, "stddev": 10.0},
        {"label": "Medium  (σ = 30)", "mean": 0.0, "stddev": 30.0},
        {"label": "Heavy   (σ = 60)", "mean": 0.0, "stddev": 60.0},
    ],
    "Salt & Pepper": [
        {"label": "Mild    (1%)",     "salt_prob": 0.01, "pepper_prob": 0.01},
        {"label": "Moderate (5%)",    "salt_prob": 0.05, "pepper_prob": 0.05},
        {"label": "Heavy   (15%)",    "salt_prob": 0.15, "pepper_prob": 0.15},
    ],
}

NOISE_TYPES = list(NOISE_PRESETS.keys())   # ["Uniform", "Gaussian", "Salt & Pepper"]
FILTER_TYPES = ["Average", "Gaussian", "Median"]


# ---------------------------------------------------------------------------
# Main tab widget
# ---------------------------------------------------------------------------

class NoiseTab(QWidget):
    def __init__(self):
        super().__init__()

        # ---- state ----
        self._original_bytes: bytes | None = None   # bytes of the loaded image
        self._noisy_bytes:    bytes | None = None   # bytes AFTER noise was applied

        # ---- root layout ----
        root = QVBoxLayout(self)
        root.setSpacing(10)

        # ── Image row ──────────────────────────────────────────────────────
        root.addWidget(self._build_image_row())

        # ── Controls row ───────────────────────────────────────────────────
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(self._build_noise_group())
        ctrl_row.addWidget(self._build_filter_group())
        root.addLayout(ctrl_row)

        # ── Status bar ─────────────────────────────────────────────────────
        self._status = QLabel("Open an image to get started.")
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setStyleSheet("color: #555; font-style: italic;")
        root.addWidget(self._status)

    # -----------------------------------------------------------------------
    # Image panels
    # -----------------------------------------------------------------------

    def _build_image_row(self) -> QWidget:
        frame = QFrame()
        layout = QHBoxLayout(frame)
        layout.setSpacing(16)

        # Open button (left side, vertically centred)
        self._open_btn = QPushButton("📂  Open Image")
        self._open_btn.setFixedWidth(130)
        self._open_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self._open_btn.clicked.connect(self._open_image)
        layout.addWidget(self._open_btn, alignment=Qt.AlignVCenter)

        # Original panel
        orig_box = QGroupBox("Original")
        orig_layout = QVBoxLayout(orig_box)
        self._orig_label = QLabel("No image loaded")
        self._orig_label.setAlignment(Qt.AlignCenter)
        self._orig_label.setMinimumSize(400, 300)
        self._orig_label.setStyleSheet("background: #1a1a2e; border-radius: 6px; color: #888;")
        orig_layout.addWidget(self._orig_label)
        layout.addWidget(orig_box)

        # Processed panel
        proc_box = QGroupBox("Processed")
        proc_layout = QVBoxLayout(proc_box)
        self._proc_label = QLabel("No image loaded")
        self._proc_label.setAlignment(Qt.AlignCenter)
        self._proc_label.setMinimumSize(400, 300)
        self._proc_label.setStyleSheet("background: #1a1a2e; border-radius: 6px; color: #888;")
        proc_layout.addWidget(self._proc_label)
        layout.addWidget(proc_box)

        return frame

    # -----------------------------------------------------------------------
    # Noise group
    # -----------------------------------------------------------------------

    def _build_noise_group(self) -> QGroupBox:
        box = QGroupBox("1.  Noise Addition")
        layout = QVBoxLayout(box)
        layout.setSpacing(8)

        # Radio buttons
        self._noise_btn_group = QButtonGroup(self)
        radio_row = QHBoxLayout()
        self._noise_radios: dict[str, QRadioButton] = {}
        for idx, name in enumerate(NOISE_TYPES):
            rb = QRadioButton(name)
            self._noise_radios[name] = rb
            self._noise_btn_group.addButton(rb, idx)
            radio_row.addWidget(rb)
        self._noise_radios[NOISE_TYPES[0]].setChecked(True)
        layout.addLayout(radio_row)

        # Preset dropdown
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self._noise_combo = QComboBox()
        self._noise_combo.setMinimumWidth(220)
        preset_row.addWidget(self._noise_combo)
        layout.addLayout(preset_row)

        # Populate the dropdown for the default noise type
        self._refresh_noise_presets(NOISE_TYPES[0])
        self._noise_btn_group.buttonClicked.connect(self._on_noise_type_changed)

        # Buttons
        btn_row = QHBoxLayout()
        self._apply_noise_btn = QPushButton("▶  Apply Noise")
        self._apply_noise_btn.clicked.connect(self._apply_noise)
        btn_row.addWidget(self._apply_noise_btn)

        self._undo_noise_btn = QPushButton("↩  Undo Noise")
        self._undo_noise_btn.clicked.connect(self._undo_noise)
        self._undo_noise_btn.setEnabled(False)
        btn_row.addWidget(self._undo_noise_btn)
        layout.addLayout(btn_row)

        return box

    # -----------------------------------------------------------------------
    # Filter group
    # -----------------------------------------------------------------------

    def _build_filter_group(self) -> QGroupBox:
        box = QGroupBox("2.  Spatial Domain Filtering (Low-Pass)")
        layout = QVBoxLayout(box)
        layout.setSpacing(8)

        # Radio buttons
        self._filter_btn_group = QButtonGroup(self)
        radio_row = QHBoxLayout()
        self._filter_radios: dict[str, QRadioButton] = {}
        for idx, name in enumerate(FILTER_TYPES):
            rb = QRadioButton(name)
            self._filter_radios[name] = rb
            self._filter_btn_group.addButton(rb, idx)
            radio_row.addWidget(rb)
        self._filter_radios[FILTER_TYPES[0]].setChecked(True)
        layout.addLayout(radio_row)

        # Kernel size spinbox
        kernel_row = QHBoxLayout()
        kernel_row.addWidget(QLabel("Kernel size (odd, 3–21):"))
        self._kernel_spin = QSpinBox()
        self._kernel_spin.setRange(3, 21)
        self._kernel_spin.setSingleStep(2)   # forces odd steps
        self._kernel_spin.setValue(3)
        self._kernel_spin.valueChanged.connect(self._enforce_odd_kernel)
        kernel_row.addWidget(self._kernel_spin)
        layout.addLayout(kernel_row)

        # Apply filter button
        self._apply_filter_btn = QPushButton("▶  Apply Filter")
        self._apply_filter_btn.clicked.connect(self._apply_filter)
        layout.addWidget(self._apply_filter_btn)

        return box

    # -----------------------------------------------------------------------
    # Slots – image loading
    # -----------------------------------------------------------------------

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        if not path:
            return
        mat = cv2.imread(path)
        if mat is None:
            self._set_status("❌  Failed to load image.", error=True)
            return
        self._original_bytes = mat_to_bytes(mat)
        self._noisy_bytes = None
        self._undo_noise_btn.setEnabled(False)
        set_label_image(self._orig_label, mat)
        set_label_image(self._proc_label, mat)
        self._set_status(f"✅  Loaded: {path.split('/')[-1]}")

    # -----------------------------------------------------------------------
    # Slots – noise
    # -----------------------------------------------------------------------

    def _on_noise_type_changed(self, btn):
        idx = self._noise_btn_group.id(btn)
        self._refresh_noise_presets(NOISE_TYPES[idx])

    def _refresh_noise_presets(self, noise_type: str):
        self._noise_combo.clear()
        for preset in NOISE_PRESETS[noise_type]:
            self._noise_combo.addItem(preset["label"])

    def _selected_noise_type(self) -> str:
        return NOISE_TYPES[self._noise_btn_group.checkedId()]

    def _apply_noise(self):
        if not self._original_bytes:
            self._set_status("⚠️  Please open an image first.", error=True)
            return
        noise_type = self._selected_noise_type()
        preset_idx = self._noise_combo.currentIndex()
        preset = NOISE_PRESETS[noise_type][preset_idx]

        try:
            if noise_type == "Uniform":
                result = cv_backend.add_uniform_noise(
                    self._original_bytes, preset["low"], preset["high"]
                )
            elif noise_type == "Gaussian":
                result = cv_backend.add_gaussian_noise(
                    self._original_bytes, preset["mean"], preset["stddev"]
                )
            else:  # Salt & Pepper
                result = cv_backend.add_salt_pepper_noise(
                    self._original_bytes, preset["salt_prob"], preset["pepper_prob"]
                )
        except Exception as e:
            self._set_status(f"❌  Error: {e}", error=True)
            return

        self._noisy_bytes = result
        self._undo_noise_btn.setEnabled(True)
        set_label_image(self._proc_label, bytes_to_mat(result))
        self._set_status(
            f"✅  {noise_type} noise applied — \"{preset['label'].strip()}\"."
        )

    def _undo_noise(self):
        """Remove noise: restore processed panel to the original image bytes."""
        if not self._original_bytes:
            return
        self._noisy_bytes = None
        self._undo_noise_btn.setEnabled(False)
        set_label_image(self._proc_label, bytes_to_mat(self._original_bytes))
        self._set_status("↩  Noise removed — showing original.")

    # -----------------------------------------------------------------------
    # Slots – filters
    # -----------------------------------------------------------------------

    def _enforce_odd_kernel(self, val: int):
        """Keep kernel size always odd."""
        if val % 2 == 0:
            self._kernel_spin.setValue(val + 1)

    def _selected_filter_type(self) -> str:
        return FILTER_TYPES[self._filter_btn_group.checkedId()]

    def _apply_filter(self):
        # Apply to noisy image if noise was added, otherwise to original
        src = self._noisy_bytes or self._original_bytes
        if not src:
            self._set_status("⚠️  Please open an image first.", error=True)
            return
        filter_type = self._selected_filter_type()
        k = self._kernel_spin.value()
        # Enforce odd kernel as a final guard
        if k % 2 == 0:
            k += 1

        try:
            if filter_type == "Average":
                result = cv_backend.apply_average_filter(src, k)
            elif filter_type == "Gaussian":
                result = cv_backend.apply_gaussian_filter(src, k)
            else:  # Median
                result = cv_backend.apply_median_filter(src, k)
        except Exception as e:
            self._set_status(f"❌  Error: {e}", error=True)
            return

        set_label_image(self._proc_label, bytes_to_mat(result))
        src_label = "noisy" if self._noisy_bytes else "original"
        self._set_status(
            f"✅  {filter_type} filter (k={k}) applied to {src_label} image."
        )

    # -----------------------------------------------------------------------
    # Status helper
    # -----------------------------------------------------------------------

    def _set_status(self, msg: str, error: bool = False):
        set_status(self._status, msg, error)