import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QRadioButton, QButtonGroup, QComboBox, QSpinBox
)
from PyQt5.QtCore import Qt

import cv_backend
from Helpers.image_utils import bytes_to_mat, set_label_image
from Helpers.styles import BaseImageTab, STATUS_QSS

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

NOISE_TYPES  = list(NOISE_PRESETS.keys())
FILTER_TYPES = ["Average", "Gaussian", "Median"]


# ---------------------------------------------------------------------------
# Main tab widget
# ---------------------------------------------------------------------------

class NoiseTab(BaseImageTab):
    def __init__(self):
        super().__init__()

        self._noisy_bytes: bytes | None = None

        root = QVBoxLayout(self)
        root.setSpacing(10)

        root.addWidget(self._build_image_row())

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(self._build_noise_group())
        ctrl_row.addWidget(self._build_filter_group())
        root.addLayout(ctrl_row)

        self._status = QLabel("Open an image to get started.")
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setStyleSheet(STATUS_QSS)
        root.addWidget(self._status)

    # -----------------------------------------------------------------------
    # Hook called by BaseImageTab._open_image()
    # -----------------------------------------------------------------------

    def _on_image_loaded(self, mat):
        self._noisy_bytes = None
        self._undo_noise_btn.setEnabled(False)

    # -----------------------------------------------------------------------
    # Noise group
    # -----------------------------------------------------------------------

    def _build_noise_group(self) -> QGroupBox:
        box = QGroupBox("1.  Noise Addition")
        layout = QVBoxLayout(box)
        layout.setSpacing(8)

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

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self._noise_combo = QComboBox()
        self._noise_combo.setMinimumWidth(220)
        preset_row.addWidget(self._noise_combo)
        layout.addLayout(preset_row)

        self._refresh_noise_presets(NOISE_TYPES[0])
        self._noise_btn_group.buttonClicked.connect(self._on_noise_type_changed)

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

        kernel_row = QHBoxLayout()
        kernel_row.addWidget(QLabel("Kernel size (odd, 3–21):"))
        self._kernel_spin = QSpinBox()
        self._kernel_spin.setRange(3, 21)
        self._kernel_spin.setSingleStep(2)
        self._kernel_spin.setValue(3)
        self._kernel_spin.valueChanged.connect(self._enforce_odd_kernel)
        kernel_row.addWidget(self._kernel_spin)
        layout.addLayout(kernel_row)

        self._apply_filter_btn = QPushButton("▶  Apply Filter")
        self._apply_filter_btn.clicked.connect(self._apply_filter)
        layout.addWidget(self._apply_filter_btn)

        return box

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
            else:
                result = cv_backend.add_salt_pepper_noise(
                    self._original_bytes, preset["salt_prob"], preset["pepper_prob"]
                )
        except Exception as e:
            self._set_status(f"❌  Error: {e}", error=True)
            return

        self._noisy_bytes = result
        self._undo_noise_btn.setEnabled(True)
        self._snapshot()   # ← save state before displaying result
        set_label_image(self._proc_label, bytes_to_mat(result), max_w=380, max_h=280)
        self._current_proc_bytes = result
        self._set_status(f"✅  {noise_type} noise applied — \"{preset['label'].strip()}\".")

    def _undo_noise(self):
        if not self._original_bytes:
            return
        self._noisy_bytes = None
        self._undo_noise_btn.setEnabled(False)
        set_label_image(self._proc_label, bytes_to_mat(self._original_bytes), max_w=380, max_h=280)
        self._set_status("↩  Noise removed — showing original.")

    # -----------------------------------------------------------------------
    # Slots – filters
    # -----------------------------------------------------------------------

    def _enforce_odd_kernel(self, val: int):
        if val % 2 == 0:
            self._kernel_spin.setValue(val + 1)

    def _selected_filter_type(self) -> str:
        return FILTER_TYPES[self._filter_btn_group.checkedId()]

    def _apply_filter(self):
        src = self._noisy_bytes or self._original_bytes
        if not src:
            self._set_status("⚠️  Please open an image first.", error=True)
            return
        filter_type = self._selected_filter_type()
        k = self._kernel_spin.value()
        if k % 2 == 0:
            k += 1

        try:
            if filter_type == "Average":
                result = cv_backend.apply_average_filter(src, k)
            elif filter_type == "Gaussian":
                result = cv_backend.apply_gaussian_filter(src, k)
            else:
                result = cv_backend.apply_median_filter(src, k)
        except Exception as e:
            self._set_status(f"❌  Error: {e}", error=True)
            return

        self._snapshot()   # ← save state before displaying result
        set_label_image(self._proc_label, bytes_to_mat(result), max_w=380, max_h=280)
        self._current_proc_bytes = result
        src_label = "noisy" if self._noisy_bytes else "original"
        self._set_status(f"✅  {filter_type} filter (k={k}) applied to {src_label} image.")