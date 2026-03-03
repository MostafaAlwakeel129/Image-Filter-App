import cv2
import numpy as np
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel


def mat_to_pixmap(mat: np.ndarray) -> QPixmap:
    """Convert an OpenCV BGR ndarray to a QPixmap."""
    rgb = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def bytes_to_mat(data: bytes) -> np.ndarray:
    """Decode PNG bytes to an OpenCV BGR ndarray."""
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def mat_to_bytes(mat: np.ndarray) -> bytes:
    """Encode an OpenCV BGR ndarray to PNG bytes."""
    ok, buf = cv2.imencode(".png", mat)
    if not ok:
        raise RuntimeError("imencode failed")
    return buf.tobytes()


def set_label_image(label: QLabel, mat: np.ndarray, max_w: int = 600, max_h: int = 500):
    """Scale and display an OpenCV mat in a QLabel."""
    px = mat_to_pixmap(mat)
    scaled = px.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    label.setPixmap(scaled)
    label.setAlignment(Qt.AlignCenter)


def set_status(label: QLabel, msg: str, error: bool = False):
    """Update a status QLabel with a coloured message."""
    color = "#c0392b" if error else "#27ae60"
    label.setStyleSheet(
        f"color: {color}; font-style: {'normal' if error else 'italic'};"
    )
    label.setText(msg)