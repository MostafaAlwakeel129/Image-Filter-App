from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
import cv_backend # This line will import the backend module that was compiled.

# Write your ui and front here
class NoiseTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        self.label = QLabel("Task 1 & 2: Noise & Smoothing\nWaiting for input...")
        layout.addWidget(self.label)

        self.btn = QPushButton("Test C++ Connection")
        self.btn.clicked.connect(self.run_cpp_test)
        layout.addWidget(self.btn)

        self.setLayout(layout)

    def run_cpp_test(self):
        # After you imported the backend module , you can use it here
        result = cv_backend.test_noise_connection(5, 10)
        self.label.setText(f"Success! C++ calculated 5 + 10 = {result}")