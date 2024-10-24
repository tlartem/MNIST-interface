import logging
import sys
import os
import subprocess

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from tensorflow import keras

from widget import PixelArtWidget

TIME_INTERVAL = 200  # Интервал прогонки через модель
MODEL_FILE = "my_model.keras"

logging.basicConfig(level=logging.INFO)

# Функция загрузки модели перемещена сюда
def load_model():
    return keras.models.load_model(MODEL_FILE)

def ensure_model_exists():
    if not os.path.exists(MODEL_FILE):
        print("Модель не найдена. Запуск fit_model.py для создания модели...")
        subprocess.run([sys.executable, "fit_model.py"])
        print("Модель создана.")

class MainWindow(QMainWindow):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Рисуй цифры")
        self.drawing_widget = PixelArtWidget()
        
        self.clear_button = QPushButton("Очистить")
        self.clear_button.clicked.connect(self.drawing_widget.clear_grid)

        self.probability_labels = [QLabel(f"Цифра {i}: 0.0%") for i in range(10)]
        for label in self.probability_labels:
            label.setStyleSheet("font-size: 16px;")

        main_layout = QHBoxLayout()
        main_layout.addLayout(self.create_left_layout())
        main_layout.addLayout(self.create_right_layout())

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.timer = QTimer()
        self.timer.timeout.connect(self.predict)
        self.timer.start(TIME_INTERVAL)

    def create_left_layout(self):
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.drawing_widget)
        left_layout.addWidget(self.clear_button)
        return left_layout

    def create_right_layout(self):
        right_layout = QVBoxLayout()
        for label in self.probability_labels:
            right_layout.addWidget(label)
        return right_layout

    def predict(self):
        grid = self.drawing_widget.get_grid()
        img = np.array(grid).reshape(1, 28, 28, 1).astype("float32") / 255
        prediction = self.model.predict(img)

        max_probability_index = np.argmax(prediction)
        for i, label in enumerate(self.probability_labels):
            probability = prediction[0][i] * 100
            label.setText(f"Цифра {i}: {probability:.2f}%")
            self.update_label_color(label, i, max_probability_index, probability)

        digit = np.argmax(prediction)
        logging.info(f"Цифра {digit}")

    def update_label_color(self, label, index, max_index, probability):
        if index == max_index:
            label.setStyleSheet("color: rgb(0, 255, 0); font-size: 16px;")
        else:
            shade_of_green = int((probability / 100) * 255)
            label.setStyleSheet(f"color: rgb(0, {shade_of_green}, 0); font-size: 16px;")

if __name__ == "__main__":
    ensure_model_exists()
    model = load_model()
    
    app = QApplication(sys.argv)
    window = MainWindow(model)
    window.show()
    sys.exit(app.exec_())
