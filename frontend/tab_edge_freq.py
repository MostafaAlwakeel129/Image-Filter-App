import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QComboBox, QSpinBox, QSlider, QRadioButton, QButtonGroup
)
from PyQt5.QtCore import Qt

import cv_backend
from Helpers.image_utils import bytes_to_mat, set_label_image
from Helpers.styles import BaseImageTab, STATUS_QSS


# ---------------------------------------------------------------------------
# Edge Detection Tab
# ---------------------------------------------------------------------------

class EdgeTab(BaseImageTab):
    def __init__(self):
        super().__init__()

        root = QVBoxLayout(self)
        root.setSpacing(10)

        root.addWidget(self._build_image_row())

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(self._build_edge_group())
        root.addLayout(ctrl_row)

        self._status = QLabel("Open an image to get started.")
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setStyleSheet(STATUS_QSS)
        root.addWidget(self._status)

    # -----------------------------------------------------------------------
    # Edge detection controls group
    # -----------------------------------------------------------------------

    def _build_edge_group(self) -> QGroupBox:
        box = QGroupBox("Edge Detection")
        layout = QVBoxLayout(box)
        layout.setSpacing(12)

        # Method dropdown
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self._method_combo = QComboBox()
        self._method_combo.setMinimumWidth(220)
        self._method_combo.addItems(["Canny", "Sobel", "Prewitt", "Roberts"])
        self._method_combo.currentTextChanged.connect(self._on_method_changed)
        method_row.addWidget(self._method_combo)
        method_row.addStretch()
        layout.addLayout(method_row)

        # Canny parameters
        self._canny_params = QWidget()
        canny_layout = QVBoxLayout(self._canny_params)
        canny_layout.setContentsMargins(0, 10, 0, 0)
        canny_layout.setSpacing(15)

        t_low_layout = QHBoxLayout()
        t_low_layout.addWidget(QLabel("T_low:"))
        self._t_low_val = QLabel("50")
        self._t_low_val.setFixedWidth(30)
        self._t_low_val.setStyleSheet("font-weight: bold; color: #87ceeb;")
        self._t_low_slider = QSlider(Qt.Horizontal)
        self._t_low_slider.setRange(0, 255)
        self._t_low_slider.setValue(50)
        self._t_low_slider.setFixedWidth(150)
        self._t_low_slider.valueChanged.connect(lambda v: self._t_low_val.setText(str(v)))
        t_low_layout.addWidget(self._t_low_slider)
        t_low_layout.addWidget(self._t_low_val)
        t_low_layout.addStretch()
        canny_layout.addLayout(t_low_layout)

        t_high_layout = QHBoxLayout()
        t_high_layout.addWidget(QLabel("T_high:"))
        self._t_high_val = QLabel("150")
        self._t_high_val.setFixedWidth(30)
        self._t_high_val.setStyleSheet("font-weight: bold; color: #87ceeb;")
        self._t_high_slider = QSlider(Qt.Horizontal)
        self._t_high_slider.setRange(0, 255)
        self._t_high_slider.setValue(150)
        self._t_high_slider.setFixedWidth(150)
        self._t_high_slider.valueChanged.connect(lambda v: self._t_high_val.setText(str(v)))
        t_high_layout.addWidget(self._t_high_slider)
        t_high_layout.addWidget(self._t_high_val)
        t_high_layout.addStretch()
        canny_layout.addLayout(t_high_layout)

        kernel_layout = QHBoxLayout()
        kernel_layout.addWidget(QLabel("Kernel size:"))
        self._kernel_spin = QSpinBox()
        self._kernel_spin.setRange(3, 7)
        self._kernel_spin.setSingleStep(2)
        self._kernel_spin.setValue(3)
        self._kernel_spin.setToolTip("Gaussian blur kernel size applied before edge detection")
        self._kernel_spin.valueChanged.connect(self._enforce_odd_kernel)
        kernel_layout.addWidget(self._kernel_spin)
        kernel_layout.addStretch()
        canny_layout.addLayout(kernel_layout)

        layout.addWidget(self._canny_params)

        # Sobel / Prewitt / Roberts direction
        self._sobel_params = QWidget()
        sobel_layout = QHBoxLayout(self._sobel_params)
        sobel_layout.setContentsMargins(0, 10, 0, 0)
        sobel_layout.setSpacing(20)

        sobel_layout.addWidget(QLabel("Direction:"))
        self._sobel_btn_group = QButtonGroup(self)
        self._sobel_x    = QRadioButton("X")
        self._sobel_y    = QRadioButton("Y")
        self._sobel_both = QRadioButton("Both")
        self._sobel_both.setChecked(True)
        for btn_id, btn in enumerate([self._sobel_x, self._sobel_y, self._sobel_both]):
            self._sobel_btn_group.addButton(btn, btn_id)
            sobel_layout.addWidget(btn)
        sobel_layout.addStretch()

        layout.addWidget(self._sobel_params)
        self._sobel_params.setVisible(False)

        # Apply button
        self._apply_btn = QPushButton("▶  Apply Edge Detection")
        self._apply_btn.clicked.connect(self._apply_edge_detection)
        layout.addWidget(self._apply_btn)

        return box

    # -----------------------------------------------------------------------
    # Slots
    # -----------------------------------------------------------------------

    def _on_method_changed(self, method: str):
        self._canny_params.setVisible(method == "Canny")
        self._sobel_params.setVisible(method in ("Sobel", "Prewitt", "Roberts"))

        # Dynamically change radio button labels for Roberts
        if method == "Roberts":
            self._sobel_x.setText("Diag +45°")
            self._sobel_y.setText("Diag -45°")
            self._sobel_both.setText("Magnitude")
        else:
            self._sobel_x.setText("X")
            self._sobel_y.setText("Y")
            self._sobel_both.setText("Both")

        if getattr(self, '_original_bytes', None):
            self._set_status(f"Method changed to {method}. Click Apply.")
        else:
            self._set_status("Open an image to get started.")

    def _enforce_odd_kernel(self, val: int):
        if val % 2 == 0:
            self._kernel_spin.setValue(val + 1)

    def _apply_edge_detection(self):
        if not getattr(self, '_original_bytes', None):
            self._set_status("⚠️  Please open an image first.", error=True)
            return

        method = self._method_combo.currentText()

        try:
            if method == "Canny":
                t_low  = self._t_low_slider.value()
                t_high = self._t_high_slider.value()
                k = self._kernel_spin.value()
                if k % 2 == 0:
                    k += 1
                if t_low >= t_high:
                    self._set_status("⚠️  T_low must be less than T_high.", error=True)
                    return
                result = cv_backend.apply_canny(self._original_bytes, t_low, t_high, k)
                status_msg = f"✅  Canny applied — T_low={t_low}, T_high={t_high}, kernel={k}."

            elif method in ("Sobel", "Prewitt", "Roberts"):
                direction = self._sobel_btn_group.checkedId()

                if method == "Sobel":
                    result = cv_backend.apply_sobel(self._original_bytes, direction)
                    dir_label = {0: "X direction", 1: "Y direction", 2: "Both"}[direction]
                elif method == "Prewitt":
                    result = cv_backend.apply_prewitt(self._original_bytes, direction)
                    dir_label = {0: "X direction", 1: "Y direction", 2: "Both"}[direction]
                else:  # Roberts
                    result = cv_backend.apply_roberts(self._original_bytes, direction)
                    dir_label = {0: "Diag +45°", 1: "Diag -45°", 2: "Magnitude"}[direction]

                status_msg = f"✅  {method} applied — {dir_label}."

            else:
                return

        except Exception as e:
            self._set_status(f"❌  Error: {e}", error=True)
            return

        # _update_proc snapshots the previous state into UndoManager,
        # then updates the processed panel and status bar.
        self._update_proc(result, status_msg)