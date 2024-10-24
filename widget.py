from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QWidget


class PixelArtWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid_size = 28
        self.pixel_size = 20
        self.grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        self.setFixedSize(
            self.grid_size * self.pixel_size, self.grid_size * self.pixel_size
        )
        self.drawing = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(Qt.black, 1, Qt.PenStyle.SolidLine))

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                gray_value = self.grid[y][x]
                painter.setBrush(QColor(gray_value, gray_value, gray_value))
                painter.drawRect(
                    x * self.pixel_size,
                    y * self.pixel_size,
                    self.pixel_size,
                    self.pixel_size,
                )

    def mousePressEvent(self, event):
        self.drawing = True
        self.update_pixel(event)

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.update_pixel(event)

    def mouseReleaseEvent(self, event):
        self.drawing = False

    def update_pixel(self, event):
        x = event.x() // self.pixel_size
        y = event.y() // self.pixel_size

        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                        gray_value = 255 - (event.y() % self.pixel_size) * (255 // self.pixel_size)
                        if dx == 0 and dy == 0:
                            self.grid[new_y][new_x] = gray_value
                        else:
                            self.grid[new_y][new_x] = min(255, self.grid[new_y][new_x] + 50)
            self.update()

    def get_grid(self):
        return self.grid

    def clear_grid(self):
        self.grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.update()
