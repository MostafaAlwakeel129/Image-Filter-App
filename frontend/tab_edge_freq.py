import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QComboBox, QSpinBox, QSlider, QRadioButton, QButtonGroup,
    QFileDialog, QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt

import cv_backend
from Helpers.image_utils import bytes_to_mat, mat_to_bytes, set_label_image, set_status


# ---------------------------------------------------------------------------
# Edge Detection Tab
# ---------------------------------------------------------------------------

class EdgeTab(QWidget):
    def __init__(self):
        super().__init__()

        self._original_bytes: bytes | None = None

        root = QVBoxLayout(self)
        root.setSpacing(10)

        root.addWidget(self._build_image_row())

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(self._build_edge_group())
        root.addLayout(ctrl_row)

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

        self._open_btn = QPushButton("📂  Open Image")
        self._open_btn.setFixedWidth(130)
        self._open_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self._open_btn.clicked.connect(self._open_image)
        layout.addWidget(self._open_btn, alignment=Qt.AlignVCenter)

        orig_box = QGroupBox("Original")
        orig_layout = QVBoxLayout(orig_box)
        self._orig_label = QLabel("No image loaded")
        self._orig_label.setAlignment(Qt.AlignCenter)
        self._orig_label.setMinimumSize(400, 300)
        self._orig_label.setStyleSheet("background: #1a1a2e; border-radius: 6px; color: #888;")
        orig_layout.addWidget(self._orig_label)
        layout.addWidget(orig_box)

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
    # Edge detection controls group
    # -----------------------------------------------------------------------

    def _build_edge_group(self) -> QGroupBox:
        box = QGroupBox("Edge Detection")
        layout = QVBoxLayout(box)
        layout.setSpacing(8)

        # ── Method dropdown ────────────────────────────────────────────────
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self._method_combo = QComboBox()
        self._method_combo.setMinimumWidth(220)
        self._method_combo.addItems(["Canny", "Sobel", "Prewitt", "Roberts"])
        self._method_combo.currentTextChanged.connect(self._on_method_changed)
        method_row.addWidget(self._method_combo)
        method_row.addStretch()
        layout.addLayout(method_row)

        # ── Canny parameters ───────────────────────────────────────────────
        self._canny_params = QWidget()
        canny_layout = QHBoxLayout(self._canny_params)
        canny_layout.setContentsMargins(0, 0, 0, 0)
        canny_layout.setSpacing(20)

        # T_low
        self._t_low_val = QLabel("50")
        self._t_low_val.setFixedWidth(30)
        self._t_low_slider = QSlider(Qt.Horizontal)
        self._t_low_slider.setRange(0, 255)
        self._t_low_slider.setValue(50)
        self._t_low_slider.setFixedWidth(150)
        self._t_low_slider.valueChanged.connect(
            lambda v: self._t_low_val.setText(str(v))
        )
        t_low_row = QHBoxLayout()
        t_low_row.setSpacing(6)
        t_low_row.addWidget(QLabel("T_low:"))
        t_low_row.addWidget(self._t_low_slider)
        t_low_row.addWidget(self._t_low_val)
        canny_layout.addLayout(t_low_row)

        # T_high
        self._t_high_val = QLabel("150")
        self._t_high_val.setFixedWidth(30)
        self._t_high_slider = QSlider(Qt.Horizontal)
        self._t_high_slider.setRange(0, 255)
        self._t_high_slider.setValue(150)
        self._t_high_slider.setFixedWidth(150)
        self._t_high_slider.valueChanged.connect(
            lambda v: self._t_high_val.setText(str(v))
        )
        t_high_row = QHBoxLayout()
        t_high_row.setSpacing(6)
        t_high_row.addWidget(QLabel("T_high:"))
        t_high_row.addWidget(self._t_high_slider)
        t_high_row.addWidget(self._t_high_val)
        canny_layout.addLayout(t_high_row)

        # Kernel size spinbox
        kernel_row = QHBoxLayout()
        kernel_row.addWidget(QLabel("Kernel size (odd, 3–7):"))
        self._kernel_spin = QSpinBox()
        self._kernel_spin.setRange(3, 7)
        self._kernel_spin.setSingleStep(2)
        self._kernel_spin.setValue(3)
        self._kernel_spin.setFixedWidth(60)
        self._kernel_spin.setToolTip("Gaussian blur kernel size applied before edge detection")
        self._kernel_spin.valueChanged.connect(self._enforce_odd_kernel)
        kernel_row.addWidget(self._kernel_spin)
        canny_layout.addLayout(kernel_row)

        canny_layout.addStretch()
        layout.addWidget(self._canny_params)

        # ── Sobel parameters ───────────────────────────────────────────────
        self._sobel_params = QWidget()
        sobel_layout = QHBoxLayout(self._sobel_params)
        sobel_layout.setContentsMargins(0, 0, 0, 0)
        sobel_layout.setSpacing(20)

        sobel_layout.addWidget(QLabel("Direction:"))

        self._sobel_btn_group = QButtonGroup(self)
        self._sobel_x   = QRadioButton("X direction")
        self._sobel_y   = QRadioButton("Y direction")
        self._sobel_both = QRadioButton("Both (X + Y)")
        self._sobel_both.setChecked(True)   # default to combined result

        self._sobel_btn_group.addButton(self._sobel_x,    0)
        self._sobel_btn_group.addButton(self._sobel_y,    1)
        self._sobel_btn_group.addButton(self._sobel_both, 2)

        sobel_layout.addWidget(self._sobel_x)
        sobel_layout.addWidget(self._sobel_y)
        sobel_layout.addWidget(self._sobel_both)
        sobel_layout.addStretch()

        layout.addWidget(self._sobel_params)
        self._sobel_params.setVisible(False)   # hidden until Sobel is selected

        # ── Apply button ───────────────────────────────────────────────────
        self._apply_btn = QPushButton("▶  Apply Edge Detection")
        self._apply_btn.clicked.connect(self._apply_edge_detection)
        layout.addWidget(self._apply_btn)

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
        set_label_image(self._orig_label, mat)
        set_label_image(self._proc_label, mat)
        self._set_status(f"✅  Loaded: {path.split('/')[-1]}")

    # -----------------------------------------------------------------------
    # Slots – method changed
    # -----------------------------------------------------------------------

    def _on_method_changed(self, method: str):
        self._canny_params.setVisible(method == "Canny")
        self._sobel_params.setVisible(method in ("Sobel", "Prewitt", "Roberts"))

        # Reset state and both image panels back to default
        self._original_bytes = None
        self._orig_label.clear()
        self._orig_label.setText("No image loaded")
        self._proc_label.clear()
        self._proc_label.setText("No image loaded")
        self._set_status("Open an image to get started.")

    # -----------------------------------------------------------------------
    # Slots – apply
    # -----------------------------------------------------------------------

    def _enforce_odd_kernel(self, val: int):
        if val % 2 == 0:
            self._kernel_spin.setValue(val + 1)

    def _apply_edge_detection(self):
        if not self._original_bytes:
            self._set_status("⚠️  Please open an image first.", error=True)
            return

        method = self._method_combo.currentText()

        try:
            if method == "Canny":
                t_low  = self._t_low_slider.value()
                t_high = self._t_high_slider.value()
                k      = self._kernel_spin.value()
                if k % 2 == 0:
                    k += 1
                if t_low >= t_high:
                    self._set_status("⚠️  T_low must be less than T_high.", error=True)
                    return
                result = cv_backend.apply_canny(self._original_bytes, t_low, t_high, k)
                self._set_status(f"✅  Canny applied — T_low={t_low}, T_high={t_high}, kernel={k}.")

            elif method in ("Sobel", "Prewitt", "Roberts"):
                direction = self._sobel_btn_group.checkedId()
                if method == "Sobel":
                    result = cv_backend.apply_sobel(self._original_bytes, direction)
                elif method == "Prewitt":
                    result = cv_backend.apply_prewitt(self._original_bytes, direction)
                else:
                    result = cv_backend.apply_roberts(self._original_bytes, direction)
                dir_label = {0: "X direction", 1: "Y direction", 2: "Both (X + Y)"}[direction]
                self._set_status(f"✅  {method} applied — {dir_label}.")

        except Exception as e:
            self._set_status(f"❌  Error: {e}", error=True)
            return

        set_label_image(self._proc_label, bytes_to_mat(result))

    # -----------------------------------------------------------------------
    # Status helper
    # -----------------------------------------------------------------------
    
    def _set_status(self, msg: str, error: bool = False):
        set_status(self._status, msg, error)