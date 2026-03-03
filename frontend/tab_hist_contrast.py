import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pyqtgraph as pg
import traceback
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QGroupBox, QTextEdit, QCheckBox, QMessageBox
)
from PyQt5.QtCore import Qt
import cv_backend
from Helpers.image_utils import bytes_to_mat, mat_to_bytes, set_label_image
from Helpers.styles import COMMON_QSS, STATUS_QSS, open_image_file
from Helpers.undo_manager import UndoManager


class HistogramContrastTab(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(COMMON_QSS + """
            QTextEdit {
                border: 2px solid #87ceeb;
                border-radius: 5px;
                padding: 5px;
                background-color: #f5f5f5;
                font-family: monospace;
                font-size: 11px;
            }
        """)

        self.current_bytes  = None
        self.original_bytes = None
        self.is_color       = False
        self._init_ui()

    def _snapshot(self):
        """Push current state to UndoManager before an operation."""
        saved_bytes    = self.current_bytes
        saved_is_color = self.is_color
        tab = self

        def _restore(b, s):
            if b is None:
                return
            # ── Restore image data ──────────────────────────────────────
            tab.current_bytes = b
            tab.is_color = saved_is_color

            # ── Restore image display ───────────────────────────────────
            set_label_image(tab.image_label, bytes_to_mat(b), max_w=480, max_h=320)

            # ── Restore mode label ──────────────────────────────────────
            tab.image_mode_label.setText(
                "🎨 Color (RGB)" if saved_is_color else "⚫ Grayscale"
            )

            # ── Restore button states ───────────────────────────────────
            tab.gray_btn.setEnabled(saved_is_color)
            tab.equalize_rgb_btn.setEnabled(saved_is_color)
            tab.normalize_rgb_btn.setEnabled(saved_is_color)
            tab._set_gray_ops_enabled(not saved_is_color)

            # ── Restore histogram selector & redraw ─────────────────────
            tab._update_histogram_selector()
            tab.update_histogram()

        UndoManager.push(saved_bytes, "", _restore)

    def _init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setSpacing(15)

        # ── Left panel — Controls ─────────────────────────────────────────
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setSpacing(15)

        # Load
        load_group = QGroupBox("Image Input")
        load_layout = QVBoxLayout()
        load_layout.setSpacing(10)
        self.load_btn = QPushButton("📂  Load Image")
        self.load_btn.clicked.connect(self.load_image)
        load_layout.addWidget(self.load_btn)

        self.image_path_label = QLabel("No image loaded")
        self.image_path_label.setWordWrap(True)
        self.image_path_label.setStyleSheet("color: #aaaaaa; font-style: italic; padding: 5px;")
        load_layout.addWidget(self.image_path_label)

        self.image_mode_label = QLabel("")
        self.image_mode_label.setStyleSheet("color: #87ceeb; font-weight: bold; padding: 5px;")
        load_layout.addWidget(self.image_mode_label)
        load_group.setLayout(load_layout)
        left_layout.addWidget(load_group)

        # Operations
        ops_group = QGroupBox("Operations")
        ops_layout = QVBoxLayout()
        ops_layout.setSpacing(10)

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

        self.reset_btn = QPushButton("↺  Reset to Original")
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

        self.update_stats_btn = QPushButton("📊  Update Statistics")
        self.update_stats_btn.clicked.connect(self.update_statistics)
        self.update_stats_btn.setEnabled(False)
        stats_layout.addWidget(self.update_stats_btn)
        stats_group.setLayout(stats_layout)
        left_layout.addWidget(stats_group)

        left_layout.addStretch()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(350)

        # ── Right panel — Visualisation ───────────────────────────────────
        right_panel  = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)

        image_group = QGroupBox("Image Display")
        image_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setFixedSize(480, 320)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(False)
        self.image_label.setStyleSheet(
            "border: 2px solid #87ceeb; border-radius: 8px; background-color: white;"
        )
        image_layout.addWidget(self.image_label)
        image_group.setLayout(image_layout)
        right_layout.addWidget(image_group)

        hist_group = QGroupBox("Histogram and Distribution")
        hist_layout = QVBoxLayout()

        self.hist_plot = pg.PlotWidget()
        self.hist_plot.setLabel('left',   'Frequency',   color='#2c3e50', size='12pt')
        self.hist_plot.setLabel('bottom', 'Pixel Value', color='#2c3e50', size='12pt')
        self.hist_plot.showGrid(x=True, y=True, alpha=0.3)
        self.hist_plot.setBackground('white')
        self.hist_plot.getAxis('bottom').setPen('#2c3e50')
        self.hist_plot.getAxis('left').setPen('#2c3e50')
        hist_layout.addWidget(self.hist_plot, stretch=2)

        hist_control_layout = QHBoxLayout()
        hist_control_layout.setSpacing(15)
        hist_control_layout.addWidget(QLabel("Histogram Type:"))

        self.hist_type = QComboBox()
        self.hist_type.addItems(["Grayscale", "RGB Combined", "RGB Separate"])
        self.hist_type.currentTextChanged.connect(self.update_histogram)
        hist_control_layout.addWidget(self.hist_type)

        self.show_cdf = QCheckBox("Show CDF")
        self.show_cdf.toggled.connect(self.update_histogram)
        hist_control_layout.addWidget(self.show_cdf)

        self.show_pdf = QCheckBox("Show PDF")
        self.show_pdf.toggled.connect(self.update_histogram)
        hist_control_layout.addWidget(self.show_pdf)
        hist_control_layout.addStretch()
        hist_layout.addLayout(hist_control_layout)

        hist_group.setLayout(hist_layout)
        right_layout.addWidget(hist_group, stretch=1)

        right_panel.setLayout(right_layout)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 2)
        self.setLayout(main_layout)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    # Bright amber — visually distinct from blue/green/red channel colors
    # and from the teal/dash-line CDF curves.
    PDF_COLOR = (255, 200, 0)

    def _detect_image_mode(self, mat):
        if mat.ndim == 2:
            return 'gray'
        if np.array_equal(mat[:, :, 0], mat[:, :, 1]) and \
           np.array_equal(mat[:, :, 0], mat[:, :, 2]):
            return 'gray'
        return 'color'

    def _update_histogram_selector(self):
        """Rebuild histogram type combo. Grayscale option is only shown once
        the image is actually grayscale (i.e. after conversion or on load of a
        gray image). While the image is still color, only the RGB options are
        available."""
        self.hist_type.blockSignals(True)
        self.hist_type.clear()
        if self.is_color:
            self.hist_type.addItems(["RGB Combined", "RGB Separate"])
        else:
            self.hist_type.addItems(["Grayscale"])
        self.hist_type.blockSignals(False)

    def _set_gray_ops_enabled(self, enabled: bool):
        """Enable or disable operations that require a grayscale image."""
        self.equalize_gray_btn.setEnabled(enabled)
        self.normalize_gray_btn.setEnabled(enabled)

    # -----------------------------------------------------------------------
    # Slots
    # -----------------------------------------------------------------------

    def load_image(self):
        mat, fname = open_image_file(self)
        if mat is None:
            if fname != "":
                QMessageBox.critical(self, "Error", "Failed to load image.")
            return

        try:
            self.is_color = (self._detect_image_mode(mat) == 'color')
            self.original_bytes = mat_to_bytes(mat)
            self.current_bytes  = mat_to_bytes(mat)

            self.image_path_label.setText(f"📁 {fname}")
            self.image_mode_label.setText("🎨 Color (RGB)" if self.is_color else "⚫ Grayscale")

            set_label_image(self.image_label, mat, max_w=480, max_h=320)

            self.gray_btn.setEnabled(self.is_color)
            self.equalize_rgb_btn.setEnabled(self.is_color)
            self.normalize_rgb_btn.setEnabled(self.is_color)
            self._set_gray_ops_enabled(not self.is_color)

            self.reset_btn.setEnabled(True)
            self.update_stats_btn.setEnabled(True)
            self.show_cdf.setEnabled(True)
            self.show_pdf.setEnabled(True)

            self._update_histogram_selector()
            self.update_histogram()

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to load image: {e}")

    def convert_to_gray(self):
        if not self.current_bytes or not self.is_color:
            return
        try:
            self._snapshot()
            self.current_bytes = cv_backend.color_to_gray(self.current_bytes)
            self.is_color = False
            self.image_mode_label.setText("⚫ Converted to Grayscale")

            self.gray_btn.setEnabled(False)
            self.equalize_rgb_btn.setEnabled(False)
            self.normalize_rgb_btn.setEnabled(False)
            self._set_gray_ops_enabled(True)

            self._update_histogram_selector()
            set_label_image(self.image_label, bytes_to_mat(self.current_bytes), max_w=480, max_h=320)
            self.update_histogram()
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Conversion failed: {e}")

    def equalize_image(self, rgb_mode):
        if not self.current_bytes:
            return
        try:
            self._snapshot()
            if rgb_mode and self.is_color:
                self.current_bytes = cv_backend.equalize_bgr(self.current_bytes)
            else:
                self.current_bytes = cv_backend.equalize_image(self.current_bytes)
            set_label_image(self.image_label, bytes_to_mat(self.current_bytes), max_w=480, max_h=320)
            self.update_histogram()
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Equalization failed: {e}")

    def normalize_image(self, rgb_mode):
        if not self.current_bytes:
            return
        try:
            self._snapshot()
            if rgb_mode and self.is_color:
                self.current_bytes = cv_backend.normalize_bgr(self.current_bytes)
            else:
                self.current_bytes = cv_backend.normalize_image(self.current_bytes)
            set_label_image(self.image_label, bytes_to_mat(self.current_bytes), max_w=480, max_h=320)
            self.update_histogram()
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Normalization failed: {e}")

    def reset_image(self):
        if not self.original_bytes:
            return
        self.current_bytes = self.original_bytes
        mat = bytes_to_mat(self.original_bytes)
        self.is_color = (self._detect_image_mode(mat) == 'color')

        self.gray_btn.setEnabled(self.is_color)
        self.equalize_rgb_btn.setEnabled(self.is_color)
        self.normalize_rgb_btn.setEnabled(self.is_color)
        self._set_gray_ops_enabled(not self.is_color)

        self.image_mode_label.setText("🎨 Color (RGB)" if self.is_color else "⚫ Grayscale")
        self._update_histogram_selector()
        set_label_image(self.image_label, mat, max_w=480, max_h=320)
        self.update_histogram()

    def update_histogram(self):
        if not self.current_bytes:
            return
        self.hist_plot.clear()
        hist_type = self.hist_type.currentText()
        try:
            if hist_type == "Grayscale":
                data = cv_backend.get_gray_histogram_and_cdf(self.current_bytes)
                self._plot_histogram([data], ['#555560'])
            elif hist_type == "RGB Combined":
                data = cv_backend.get_bgr_histograms_and_cdfs(self.current_bytes)
                self._plot_histogram(data, ['#5B9BF5', '#4CAF7D', '#F05C5C'])
            elif hist_type == "RGB Separate":
                data = cv_backend.get_bgr_histograms_and_cdfs(self.current_bytes)
                self._plot_separate_histograms(data, ['#5B9BF5', '#4CAF7D', '#F05C5C'])
        except Exception:
            traceback.print_exc()

    # -----------------------------------------------------------------------
    # Histogram rendering helpers
    # -----------------------------------------------------------------------

    def _hex_to_rgba(self, hex_color: str, alpha: int = 180):
        """Convert '#RRGGBB' to an (R, G, B, A) tuple."""
        h = hex_color.lstrip('#')
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)

    def _style_axes(self):
        """Apply consistent axis styling after plotting."""
        self.hist_plot.setXRange(0, 255, padding=0.005)
        self.hist_plot.getAxis('bottom').setTicks([
            [(0, '0'), (64, '64'), (128, '128'), (192, '192'), (255, '255')]
        ])
        for ax in ('left', 'bottom'):
            self.hist_plot.getAxis(ax).setPen(pg.mkPen('#cccccc', width=1))
            self.hist_plot.getAxis(ax).setTextPen(pg.mkPen('#555555'))
        self.hist_plot.setLabel('bottom', 'Pixel Value (0 – 255)',
                                color='#555555', size='9pt')
        self.hist_plot.showGrid(x=False, y=True, alpha=0.15)

    def _plot_histogram(self, hist_data_list, colors):
        """
        Grayscale / RGB-Combined view.

        CDF is a normalized 0→1 curve from the backend. We scale it to
        match the histogram's Y axis by multiplying by the histogram's peak
        (max bin count), so it stays within the plot and aligns visually.
        PDF uses a fixed bright amber color so it is always visually distinct
        from the channel bars and the CDF dashed line.
        """
        x        = np.arange(256)
        n        = len(hist_data_list)
        ch_names = {
            '#5B9BF5': 'Blue', '#4CAF7D': 'Green', '#F05C5C': 'Red', '#555560': 'Intensity'
        }

        for i, (hist, cdf, pdf) in enumerate(hist_data_list):
            color        = colors[i] if i < len(colors) else '#555560'
            label        = ch_names.get(color, f'Ch{i}')
            hist_arr     = np.array(hist, dtype=float)
            fill_alpha   = 200 if n == 1 else 100
            fill_rgba    = self._hex_to_rgba(color, fill_alpha)
            outline_rgba = self._hex_to_rgba(color, 255)

            # Filled bars
            bars = pg.BarGraphItem(
                x=x, height=hist_arr, width=1.0,
                brush=pg.mkBrush(*fill_rgba),
                pen=pg.mkPen(None),
            )
            self.hist_plot.addItem(bars)

            # Crisp outline
            self.hist_plot.plot(
                x, hist_arr,
                pen=pg.mkPen(color=outline_rgba[:3], width=1),
                name=f'{label} Histogram',
            )

            peak_count = float(hist_arr.max()) if hist_arr.max() > 0 else 1.0

            if self.show_cdf.isChecked():
                self.hist_plot.plot(
                    x, np.array(cdf) * peak_count,
                    pen=pg.mkPen(color=outline_rgba[:3], width=2, style=Qt.DashLine),
                    name=f'{label} CDF',
                )
            if self.show_pdf.isChecked():
                pdf_arr  = np.array(pdf, dtype=float)
                pdf_peak = float(pdf_arr.max()) if pdf_arr.max() > 0 else 1.0
                # Bright amber — distinct from all channel colors and the CDF line
                self.hist_plot.plot(
                    x, pdf_arr / pdf_peak * peak_count,
                    pen=pg.mkPen(color=self.PDF_COLOR, width=2, style=Qt.DotLine),
                    name=f'{label} PDF',
                )

        self.hist_plot.setLabel('left', 'Pixel Count', color='#555555', size='9pt')
        self._style_axes()

    def _plot_separate_histograms(self, hist_data_list, colors):
        """
        RGB-Separate view: three channels stacked vertically.

        CDF and PDF are both rescaled to the channel's peak bin count so they
        sit inside each channel's slot. PDF always renders in bright amber so
        it is easily distinguished from the per-channel CDF dashed lines.
        """
        x        = np.arange(256)
        ch_names = {'#5B9BF5': 'Blue', '#4CAF7D': 'Green', '#F05C5C': 'Red'}

        all_maxes  = [float(max(ch[0])) if max(ch[0]) > 0 else 1.0 for ch in hist_data_list]
        global_max = max(all_maxes)
        gap        = global_max * 0.18
        slot_h     = global_max + gap

        for i, (hist, cdf, pdf) in enumerate(hist_data_list):
            color       = colors[i] if i < len(colors) else '#555560'
            label       = ch_names.get(color, f'Ch{i}')
            hist_arr    = np.array(hist, dtype=float)
            offset      = i * slot_h
            fill_rgba   = self._hex_to_rgba(color, 170)
            outline_rgb = self._hex_to_rgba(color, 255)[:3]

            # Thin baseline for this channel
            baseline = pg.PlotDataItem(
                [0, 255], [offset, offset],
                pen=pg.mkPen(color='#dddddd', width=1),
            )
            self.hist_plot.addItem(baseline)

            # Filled bars
            bars = pg.BarGraphItem(
                x=x, height=hist_arr, width=1.0,
                y0=offset,
                brush=pg.mkBrush(*fill_rgba),
                pen=pg.mkPen(None),
            )
            self.hist_plot.addItem(bars)

            # Outline
            self.hist_plot.plot(
                x, hist_arr + offset,
                pen=pg.mkPen(color=outline_rgb, width=1),
                name=f'{label} Histogram',
            )

            # Channel label
            txt = pg.TextItem(label, color=outline_rgb, anchor=(1.0, 0.5))
            txt.setPos(-2, offset + global_max * 0.5)
            self.hist_plot.addItem(txt)

            peak_count = float(hist_arr.max()) if hist_arr.max() > 0 else 1.0

            if self.show_cdf.isChecked():
                self.hist_plot.plot(
                    x, np.array(cdf) * peak_count + offset,
                    pen=pg.mkPen(color=outline_rgb, width=2, style=Qt.DashLine),
                    name=f'{label} CDF',
                )
            if self.show_pdf.isChecked():
                pdf_arr  = np.array(pdf, dtype=float)
                pdf_peak = float(pdf_arr.max()) if pdf_arr.max() > 0 else 1.0
                # Bright amber — distinct from all channel colors and the CDF line
                self.hist_plot.plot(
                    x, pdf_arr / pdf_peak * peak_count + offset,
                    pen=pg.mkPen(color=self.PDF_COLOR, width=2, style=Qt.DotLine),
                    name=f'{label} PDF',
                )

        # Hide numeric left axis — channel labels replace it
        self.hist_plot.getAxis('left').setTicks([[]])
        self.hist_plot.setLabel('left', '', color='#555555', size='9pt')
        self._style_axes()

    def update_statistics(self):
        if not self.current_bytes:
            return
        try:
            mat = bytes_to_mat(self.current_bytes)
            if self.is_color:
                text = "🎨 RGB Image Statistics:\n\n"
                for i, name in enumerate(['Blue', 'Green', 'Red']):
                    ch = mat[:, :, i].astype(np.float64)
                    text += (f"{name} Channel:\n"
                             f"  Mean:    {ch.mean():.2f}\n"
                             f"  Std Dev: {ch.std():.2f}\n"
                             f"  Min:     {ch.min():.0f}\n"
                             f"  Max:     {ch.max():.0f}\n\n")
            else:
                s = cv_backend.compute_stats(self.current_bytes)
                text = (f"⚫ Grayscale Image Statistics:\n\n"
                        f"Mean:    {s.mean:.2f}\n"
                        f"Std Dev: {s.stddev:.2f}\n"
                        f"Min:     {s.min_val:.0f}\n"
                        f"Max:     {s.max_val:.0f}\n")
            self.stats_text.setText(text)
        except Exception as e:
            traceback.print_exc()
            self.stats_text.setText(f"Error computing statistics: {e}")