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

# Функция загрузки модели перемещена сюда
def load_model():
    return keras.models.load_model("my_model.keras")



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Рисуй цифры")
        self.drawing_widget = PixelArtWidget()
        self.setCentralWidget(self.drawing_widget)

        self.clear_button = QPushButton("Очистить")
        self.clear_button.clicked.connect(self.drawing_widget.clear_grid)

        self.probability_labels = [QLabel(f"Цифра {i}: 0.0%") for i in range(10)]
        for label in self.probability_labels:
            label.setStyleSheet("font-size: 16px;")

        main_layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.drawing_widget)
        left_layout.addWidget(self.clear_button)

        right_layout = QVBoxLayout()
        for label in self.probability_labels:
            right_layout.addWidget(label)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.timer = QTimer()
        self.timer.timeout.connect(self.predict)
        self.timer.start(TIME_INTERVAL)

    def predict(self):
        grid = self.drawing_widget.get_grid()
        img = np.array(grid).reshape(1, 784).astype("float32") / 255
        prediction = model.predict(img)

        max_probability_index = np.argmax(prediction)
        for i, label in enumerate(self.probability_labels):
            probability = prediction[0][i] * 100
            label.setText(f"Цифра {i}: {probability:.2f}%")
            if i == max_probability_index:
                label.setStyleSheet("color: rgb(0, 255, 0); font-size: 16px;")
            else:
                shade_of_green = int((probability / 100) * 255)
                label.setStyleSheet(
                    f"color: rgb(0, {shade_of_green}, 0); font-size: 16px;"
                )

        digit = np.argmax(prediction)
        logging.info(f"Цифра {digit}")


if __name__ == "__main__":
    if not os.path.exists("my_model.keras"):
        print("Модель не найдена. Запуск fit_model.py для создания модели...")
        subprocess.run([sys.executable, "fit_model.py"])
        print("Модель создана.")

    model = load_model()
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
