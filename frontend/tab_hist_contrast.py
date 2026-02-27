import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pyqtgraph as pg
import traceback
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QLabel, QComboBox, QGroupBox, QTextEdit, QCheckBox, QMessageBox
)
from PyQt5.QtCore import Qt
import cv_backend
from Helpers.image_utils import bytes_to_mat, mat_to_bytes, set_label_image, set_status


class HistogramContrastTab(QWidget):
    def __init__(self):
        super().__init__()
        self.current_bytes  = None   # py::bytes of the working image
        self.original_bytes = None   # py::bytes of the original loaded image
        self.is_color = False
        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout()

        # ── Left panel — Controls ─────────────────────────────────────────
        left_panel = QWidget()
        left_layout = QVBoxLayout()

        # Load
        load_group = QGroupBox("Image Input")
        load_layout = QVBoxLayout()
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        load_layout.addWidget(self.load_btn)
        self.image_path_label = QLabel("No image loaded")
        self.image_path_label.setWordWrap(True)
        load_layout.addWidget(self.image_path_label)
        self.image_mode_label = QLabel("")
        self.image_mode_label.setStyleSheet("color: gray; font-style: italic;")
        load_layout.addWidget(self.image_mode_label)
        load_group.setLayout(load_layout)
        left_layout.addWidget(load_group)

        # Operations
        ops_group = QGroupBox("Operations")
        ops_layout = QVBoxLayout()

        self.gray_btn = QPushButton("Convert to Grayscale")
        self.gray_btn.clicked.connect(self.convert_to_gray)
        self.gray_btn.setEnabled(False)
        ops_layout.addWidget(self.gray_btn)

        equalize_layout = QHBoxLayout()
        self.equalize_gray_btn = QPushButton("Equalize (Gray)")
        self.equalize_gray_btn.clicked.connect(lambda: self.equalize_image(False))
        self.equalize_gray_btn.setEnabled(False)
        equalize_layout.addWidget(self.equalize_gray_btn)
        self.equalize_rgb_btn = QPushButton("Equalize (RGB)")
        self.equalize_rgb_btn.clicked.connect(lambda: self.equalize_image(True))
        self.equalize_rgb_btn.setEnabled(False)
        equalize_layout.addWidget(self.equalize_rgb_btn)
        ops_layout.addLayout(equalize_layout)

        normalize_layout = QHBoxLayout()
        self.normalize_gray_btn = QPushButton("Normalize (Gray)")
        self.normalize_gray_btn.clicked.connect(lambda: self.normalize_image(False))
        self.normalize_gray_btn.setEnabled(False)
        normalize_layout.addWidget(self.normalize_gray_btn)
        self.normalize_rgb_btn = QPushButton("Normalize (RGB)")
        self.normalize_rgb_btn.clicked.connect(lambda: self.normalize_image(True))
        self.normalize_rgb_btn.setEnabled(False)
        normalize_layout.addWidget(self.normalize_rgb_btn)
        ops_layout.addLayout(normalize_layout)

        self.reset_btn = QPushButton("Reset to Original")
        self.reset_btn.clicked.connect(self.reset_image)
        self.reset_btn.setEnabled(False)
        ops_layout.addWidget(self.reset_btn)

        ops_group.setLayout(ops_layout)
        left_layout.addWidget(ops_group)

        # Statistics
        stats_group = QGroupBox("Image Statistics")
        stats_layout = QVBoxLayout()
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)
        self.update_stats_btn = QPushButton("Update Statistics")
        self.update_stats_btn.clicked.connect(self.update_statistics)
        self.update_stats_btn.setEnabled(False)
        stats_layout.addWidget(self.update_stats_btn)
        stats_group.setLayout(stats_layout)
        left_layout.addWidget(stats_group)

        left_layout.addStretch()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(300)

        # ── Right panel — Visualisation ───────────────────────────────────
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        image_group = QGroupBox("Image Display")
        image_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        image_layout.addWidget(self.image_label)
        image_group.setLayout(image_layout)
        right_layout.addWidget(image_group)

        hist_group = QGroupBox("Histogram and Distribution")
        hist_layout = QVBoxLayout()
        self.hist_plot = pg.PlotWidget()
        self.hist_plot.setLabel('left', 'Frequency')
        self.hist_plot.setLabel('bottom', 'Pixel Value')
        self.hist_plot.showGrid(x=True, y=True)
        hist_layout.addWidget(self.hist_plot)

        hist_control_layout = QHBoxLayout()
        self.hist_type = QComboBox()
        self.hist_type.addItems(["Grayscale", "RGB Combined", "RGB Separate"])
        self.hist_type.currentTextChanged.connect(self.update_histogram)
        hist_control_layout.addWidget(QLabel("Histogram Type:"))
        hist_control_layout.addWidget(self.hist_type)
        self.show_cdf = QCheckBox("Show CDF")
        self.show_cdf.toggled.connect(self.update_histogram)
        hist_control_layout.addWidget(self.show_cdf)
        self.show_pdf = QCheckBox("Show PDF")
        self.show_pdf.toggled.connect(self.update_histogram)
        hist_control_layout.addWidget(self.show_pdf)
        hist_layout.addLayout(hist_control_layout)

        hist_group.setLayout(hist_layout)
        right_layout.addWidget(hist_group)

        right_panel.setLayout(right_layout)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)
        self.setLayout(main_layout)

    # ── Image loading ─────────────────────────────────────────────────────

    def _detect_image_mode(self, mat):
        """
        Determine whether a loaded BGR cv2 mat is truly color or grayscale.
        Returns 'gray' or 'color'.
        """
        if mat.ndim == 2:
            return 'gray'
        # Check whether all three BGR channels are identical
        if np.array_equal(mat[:, :, 0], mat[:, :, 1]) and \
           np.array_equal(mat[:, :, 0], mat[:, :, 2]):
            return 'gray'
        return 'color'

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if not path:
            return

        try:
            mat = cv2.imread(path)
            if mat is None:
                raise RuntimeError("cv2.imread returned None — unsupported format or bad path.")

            mode = self._detect_image_mode(mat)
            self.is_color = (mode == 'color')

            # Store everything as bytes — single source of truth
            self.original_bytes = mat_to_bytes(mat)
            self.current_bytes  = mat_to_bytes(mat)

            self.image_path_label.setText(path.split('/')[-1])
            self.image_mode_label.setText(
                "Detected: Color (RGB)" if self.is_color else "Detected: Grayscale"
            )

            set_label_image(self.image_label, mat)

            # Button states
            self.gray_btn.setEnabled(self.is_color)
            self.equalize_gray_btn.setEnabled(True)
            self.equalize_rgb_btn.setEnabled(self.is_color)
            self.normalize_gray_btn.setEnabled(True)
            self.normalize_rgb_btn.setEnabled(self.is_color)
            self.reset_btn.setEnabled(True)
            self.update_stats_btn.setEnabled(True)

            self._update_histogram_selector()
            self.update_histogram()

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to load image: {e}")

    def _update_histogram_selector(self):
        self.hist_type.blockSignals(True)
        self.hist_type.clear()
        if self.is_color:
            self.hist_type.addItems(["Grayscale", "RGB Combined", "RGB Separate"])
        else:
            self.hist_type.addItems(["Grayscale"])
        self.hist_type.blockSignals(False)

    # ── Operations ────────────────────────────────────────────────────────

    def convert_to_gray(self):
        if not self.current_bytes or not self.is_color:
            return
        try:
            self.current_bytes = cv_backend.color_to_gray(self.current_bytes)
            self.is_color = False

            self.image_mode_label.setText(
                "Detected: Color (RGB) → converted to Grayscale"
            )
            self.gray_btn.setEnabled(False)
            self.equalize_rgb_btn.setEnabled(False)
            self.normalize_rgb_btn.setEnabled(False)

            self._update_histogram_selector()
            set_label_image(self.image_label, bytes_to_mat(self.current_bytes))
            self.update_histogram()

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Conversion failed: {e}")

    def equalize_image(self, rgb_mode):
        if not self.current_bytes:
            return
        try:
            if rgb_mode and self.is_color:
                self.current_bytes = cv_backend.equalize_bgr(self.current_bytes)
            else:
                self.current_bytes = cv_backend.equalize_image(self.current_bytes)

            set_label_image(self.image_label, bytes_to_mat(self.current_bytes))
            self.update_histogram()
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Equalization failed: {e}")

    def normalize_image(self, rgb_mode):
        if not self.current_bytes:
            return
        try:
            if rgb_mode and self.is_color:
                self.current_bytes = cv_backend.normalize_bgr(self.current_bytes)
            else:
                self.current_bytes = cv_backend.normalize_image(self.current_bytes)

            set_label_image(self.image_label, bytes_to_mat(self.current_bytes))
            self.update_histogram()
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Normalization failed: {e}")

    def reset_image(self):
        if not self.original_bytes:
            return

        self.current_bytes = self.original_bytes

        # Re-detect mode from the original
        mat = bytes_to_mat(self.original_bytes)
        self.is_color = (self._detect_image_mode(mat) == 'color')

        self.gray_btn.setEnabled(self.is_color)
        self.equalize_rgb_btn.setEnabled(self.is_color)
        self.normalize_rgb_btn.setEnabled(self.is_color)
        self.image_mode_label.setText(
            "Detected: Color (RGB)" if self.is_color else "Detected: Grayscale"
        )

        self._update_histogram_selector()
        set_label_image(self.image_label, mat)
        self.update_histogram()

    # ── Histogram ─────────────────────────────────────────────────────────

    def update_histogram(self):
        if not self.current_bytes:
            return

        self.hist_plot.clear()
        hist_type = self.hist_type.currentText()

        try:
            if hist_type == "Grayscale":
                data = cv_backend.get_gray_histogram_and_cdf(self.current_bytes)
                self._plot_histogram([data], ['w'])

            elif hist_type == "RGB Combined":
                data = cv_backend.get_bgr_histograms_and_cdfs(self.current_bytes)
                self._plot_histogram(data, ['b', 'g', 'r'])

            elif hist_type == "RGB Separate":
                data = cv_backend.get_bgr_histograms_and_cdfs(self.current_bytes)
                self._plot_separate_histograms(data, ['b', 'g', 'r'])

        except Exception as e:
            traceback.print_exc()

    def _plot_histogram(self, hist_data_list, colors):
        x = np.arange(256)
        color_names = {'b': 'Blue', 'g': 'Green', 'r': 'Red', 'w': 'Gray'}
        for i, channel_data in enumerate(hist_data_list):
            hist, cdf, pdf = channel_data[0], channel_data[1], channel_data[2]
            color    = colors[i] if i < len(colors) else 'w'
            label    = color_names.get(color, color)
            max_hist = max(hist) if max(hist) > 0 else 1

            self.hist_plot.plot(x, hist,
                                pen=pg.mkPen(color=color, width=1),
                                name=f'{label} Histogram')
            if self.show_cdf.isChecked():
                self.hist_plot.plot(x, [v * max_hist for v in cdf],
                                    pen=pg.mkPen(color=color, width=2, style=Qt.DashLine),
                                    name=f'{label} CDF')
            if self.show_pdf.isChecked():
                self.hist_plot.plot(x, [v * max_hist for v in pdf],
                                    pen=pg.mkPen(color=color, width=2, style=Qt.DotLine),
                                    name=f'{label} PDF')

    def _plot_separate_histograms(self, hist_data_list, colors):
        x           = np.arange(256)
        color_names = {'b': 'Blue', 'g': 'Green', 'r': 'Red', 'w': 'Gray'}
        max_vals    = [max(ch[0]) for ch in hist_data_list]
        offset_step = max(max_vals) * 1.2 if max_vals else 1000

        for i, channel_data in enumerate(hist_data_list):
            hist, cdf, pdf = channel_data[0], channel_data[1], channel_data[2]
            color    = colors[i]
            label    = color_names.get(color, color)
            offset   = i * offset_step
            max_hist = max(hist) if max(hist) > 0 else 1

            self.hist_plot.plot(x, [h + offset for h in hist],
                                pen=pg.mkPen(color=color, width=1),
                                name=f'{label} Histogram')
            if self.show_cdf.isChecked():
                self.hist_plot.plot(x, [v * max_hist + offset for v in cdf],
                                    pen=pg.mkPen(color=color, width=2, style=Qt.DashLine),
                                    name=f'{label} CDF')
            if self.show_pdf.isChecked():
                self.hist_plot.plot(x, [v * max_hist + offset for v in pdf],
                                    pen=pg.mkPen(color=color, width=2, style=Qt.DotLine),
                                    name=f'{label} PDF')

    # ── Statistics ────────────────────────────────────────────────────────

    def update_statistics(self):
        if not self.current_bytes:
            return
        try:
            mat = bytes_to_mat(self.current_bytes)

            if self.is_color:
                text = "RGB Image Statistics:\n\n"
                for i, name in enumerate(['Blue', 'Green', 'Red']):
                    # Extract single channel, encode to bytes, pass to backend
                    ch_mat = mat[:, :, i].copy()
                    ch_bgr = cv2.cvtColor(ch_mat, cv2.COLOR_GRAY2BGR)
                    s = cv_backend.compute_stats(mat_to_bytes(ch_bgr))
                    text += (f"{name} Channel:\n"
                             f"  Mean:    {s.mean:.2f}\n"
                             f"  Std Dev: {s.stddev:.2f}\n"
                             f"  Min:     {s.min_val:.0f}\n"
                             f"  Max:     {s.max_val:.0f}\n\n")
            else:
                s = cv_backend.compute_stats(self.current_bytes)
                text = (f"Grayscale Image Statistics:\n\n"
                        f"Mean:    {s.mean:.2f}\n"
                        f"Std Dev: {s.stddev:.2f}\n"
                        f"Min:     {s.min_val:.0f}\n"
                        f"Max:     {s.max_val:.0f}\n")

            self.stats_text.setText(text)

        except Exception as e:
            traceback.print_exc()
            self.stats_text.setText(f"Error computing statistics: {e}")