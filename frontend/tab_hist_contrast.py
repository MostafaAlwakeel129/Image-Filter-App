import sys
import numpy as np
import pyqtgraph as pg
import traceback
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QLabel, QComboBox, QGroupBox, QTextEdit, QCheckBox, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import cv_backend


class HistogramContrastTab(QWidget):
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.current_gray  = None
        self.original_image = None
        self.original_gray  = None
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
        # Shows the detected mode after loading
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

    def _detect_image_mode(self, pil_image):
        """
        Reliably determine whether an image is truly grayscale or color.

        PIL's mode tells us the file's encoding, but a JPEG saved as RGB
        can still contain identical R/G/B channels (i.e. visually grayscale).
        We check the actual pixel data so the user never has to do it manually.

        Returns: ('gray', gray_array) or ('color', rgb_array)
        """
        # Fast path: mode is already unambiguously grayscale
        if pil_image.mode in ('L', 'LA', '1', 'P'):
            gray = np.array(pil_image.convert('L'), dtype=np.uint8)
            return 'gray', gray

        # Convert to RGB for uniform handling
        rgb = np.array(pil_image.convert('RGB'), dtype=np.uint8)

        # Check whether all three channels are identical (visually grayscale
        # stored in an RGB container — common with phone photos, scans, etc.)
        if np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and \
           np.array_equal(rgb[:, :, 0], rgb[:, :, 2]):
            return 'gray', rgb[:, :, 0]

        return 'color', rgb

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if not path:
            return

        try:
            from PIL import Image
            pil = Image.open(path)

            mode, data = self._detect_image_mode(pil)

            if mode == 'gray':
                # Genuinely grayscale — work in gray space directly
                self.original_image = None          # no color original
                self.original_gray  = data
                self.is_color = False
                self.image_mode_label.setText("Detected: Grayscale")
            else:
                # Color image
                self.original_image = data          # RGB array
                self.original_gray  = None          # not yet converted
                self.is_color = True
                self.image_mode_label.setText("Detected: Color (RGB)")

            self.current_image = self.original_image.copy() if self.original_image is not None else None
            self.current_gray  = self.original_gray.copy()  if self.original_gray  is not None else None

            self.image_path_label.setText(path.split('/')[-1])  # just the filename

            # Display whichever we have
            self.display_image(self.current_gray if not self.is_color else self.current_image)

            # ── Button state: adapt to image type ────────────────────────
            # "Convert to Grayscale" only makes sense for color images
            self.gray_btn.setEnabled(self.is_color)

            # Equalize / normalize gray always available; RGB only for color
            self.equalize_gray_btn.setEnabled(True)
            self.equalize_rgb_btn.setEnabled(self.is_color)
            self.normalize_gray_btn.setEnabled(True)
            self.normalize_rgb_btn.setEnabled(self.is_color)

            self.reset_btn.setEnabled(True)
            self.update_stats_btn.setEnabled(True)

            # Lock histogram type selector to what makes sense
            self._update_histogram_selector()
            self.update_histogram()

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to load image: {e}")

    def _update_histogram_selector(self):
        """Restrict the histogram type dropdown to options that make sense."""
        self.hist_type.blockSignals(True)
        self.hist_type.clear()
        if self.is_color:
            self.hist_type.addItems(["Grayscale", "RGB Combined", "RGB Separate"])
        else:
            # Grayscale image — no point offering RGB views
            self.hist_type.addItems(["Grayscale"])
        self.hist_type.blockSignals(False)

    # ── Display ───────────────────────────────────────────────────────────

    def display_image(self, image):
        if image is None:
            return

        img = np.ascontiguousarray(image)
        h, w = img.shape[:2]

        if img.ndim == 3:
            qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
        else:
            qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)

        if qimg.isNull():
            return

        self.image_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    # ── Operations ────────────────────────────────────────────────────────

    def _active_display(self):
        """Return whichever array is currently being shown."""
        if not self.is_color:
            return self.current_gray
        return self.current_image

    def convert_to_gray(self):
        """Convert a color image to grayscale and switch the working mode."""
        if self.current_image is not None and self.is_color:
            try:
                gray = cv_backend.color_to_gray(self.current_image)
                self.current_gray  = gray
                self.current_image = None   # discard color working copy
                self.is_color = False

                self.image_mode_label.setText("Detected: Color (RGB) → converted to Grayscale")

                # Update button states
                self.gray_btn.setEnabled(False)         # already gray
                self.equalize_rgb_btn.setEnabled(False)
                self.normalize_rgb_btn.setEnabled(False)

                self._update_histogram_selector()
                self.display_image(self.current_gray)
                self.update_histogram()

            except Exception as e:
                traceback.print_exc()
                QMessageBox.critical(self, "Error", f"Conversion failed: {e}")

    def equalize_image(self, rgb_mode):
        if self.current_image is None and self.current_gray is None:
            return
        try:
            if rgb_mode and self.is_color:
                self.current_image = cv_backend.equalize_bgr(self.current_image)
                self.display_image(self.current_image)
            else:
                src = self.current_gray if self.current_gray is not None else self.current_image
                result = cv_backend.equalize_image(src)
                if self.current_gray is not None:
                    self.current_gray = result
                else:
                    self.current_image = result
                self.display_image(result)
            self.update_histogram()
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Equalization failed: {e}")

    def normalize_image(self, rgb_mode):
        if self.current_image is None and self.current_gray is None:
            return
        try:
            if rgb_mode and self.is_color:
                self.current_image = cv_backend.normalize_bgr(self.current_image)
                self.display_image(self.current_image)
            else:
                src = self.current_gray if self.current_gray is not None else self.current_image
                result = cv_backend.normalize_image(src)
                if self.current_gray is not None:
                    self.current_gray = result
                else:
                    self.current_image = result
                self.display_image(result)
            self.update_histogram()
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Normalization failed: {e}")

    def reset_image(self):
        if self.original_image is not None or self.original_gray is not None:
            self.current_image = self.original_image.copy() if self.original_image is not None else None
            self.current_gray  = self.original_gray.copy()  if self.original_gray  is not None else None

            # Restore is_color to whatever was detected on load
            self.is_color = self.original_image is not None

            # Restore buttons
            self.gray_btn.setEnabled(self.is_color)
            self.equalize_rgb_btn.setEnabled(self.is_color)
            self.normalize_rgb_btn.setEnabled(self.is_color)

            self.image_mode_label.setText(
                "Detected: Color (RGB)" if self.is_color else "Detected: Grayscale"
            )

            self._update_histogram_selector()
            self.display_image(self.current_gray if not self.is_color else self.current_image)
            self.update_histogram()

    # ── Histogram ─────────────────────────────────────────────────────────

    def update_histogram(self):
        """
        Redraws the histogram from the *current* working image so it always
        reflects whatever operations have been applied.
        """
        if self.current_image is None and self.current_gray is None:
            return

        self.hist_plot.clear()
        hist_type = self.hist_type.currentText()

        try:
            if hist_type == "Grayscale":
                src = self.current_gray if self.current_gray is not None else self.current_image
                if src.ndim == 3:
                    src = cv_backend.color_to_gray(src)
                self._plot_histogram([cv_backend.get_gray_histogram_and_cdf(src)], ['w'])

            elif hist_type == "RGB Combined":
                if self.current_image is not None and self.current_image.ndim == 3:
                    data = cv_backend.get_bgr_histograms_and_cdfs(self.current_image)
                    self._plot_histogram(data, ['b', 'g', 'r'])
                else:
                    src = self.current_gray
                    self._plot_histogram([cv_backend.get_gray_histogram_and_cdf(src)], ['w'])

            elif hist_type == "RGB Separate":
                if self.current_image is not None and self.current_image.ndim == 3:
                    data = cv_backend.get_bgr_histograms_and_cdfs(self.current_image)
                    self._plot_separate_histograms(data, ['b', 'g', 'r'])
                else:
                    src = self.current_gray
                    self._plot_histogram([cv_backend.get_gray_histogram_and_cdf(src)], ['w'])

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
        x            = np.arange(256)
        color_names  = {'b': 'Blue', 'g': 'Green', 'r': 'Red', 'w': 'Gray'}
        max_vals     = [max(ch[0]) for ch in hist_data_list]
        offset_step  = max(max_vals) * 1.2 if max_vals else 1000

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
        if self.current_image is None and self.current_gray is None:
            return

        try:
            src = self.current_gray if self.current_gray is not None else self.current_image

            if src.ndim == 3:
                text = "RGB Image Statistics:\n\n"
                for i, name in enumerate(['Blue', 'Green', 'Red']):
                    s = cv_backend.compute_stats(src[:, :, i].astype(np.uint8))
                    text += (f"{name} Channel:\n"
                             f"  Mean:    {s.mean:.2f}\n"
                             f"  Std Dev: {s.stddev:.2f}\n"
                             f"  Min:     {s.min_val:.0f}\n"
                             f"  Max:     {s.max_val:.0f}\n\n")
            else:
                s = cv_backend.compute_stats(src)
                text = (f"Grayscale Image Statistics:\n\n"
                        f"Mean:    {s.mean:.2f}\n"
                        f"Std Dev: {s.stddev:.2f}\n"
                        f"Min:     {s.min_val:.0f}\n"
                        f"Max:     {s.max_val:.0f}\n")

            self.stats_text.setText(text)

        except Exception as e:
            traceback.print_exc()
            self.stats_text.setText(f"Error computing statistics: {e}")