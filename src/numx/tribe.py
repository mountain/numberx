import numpy as np


class Tribe:
    def __init__(self, width, height, x, y, d):
        self.width = width
        self.height = height
        self.direction = d
        self.x = x
        self.y = y

    def left_wing(self):
        xx, yy = np.meshgrid(np.linspace(-1, 1, self.height), np.linspace(-1, 1, self.width))
        xx, yy = xx - 1, yy + 0.5
        rr = np.sqrt(xx * xx + yy * yy)
        phi = np.arctan2(yy, xx) + self.direction * np.pi / 180
        xx, yy = rr * np.cos(phi) / 10, rr * np.sin(phi) / 10
        xx, yy = xx + self.x, yy + self.y
        return xx, yy

    def right_wing(self):
        xx, yy = np.meshgrid(np.linspace(-1, 1, self.height), np.linspace(-1, 1, self.width))
        xx, yy = xx + 1, yy + 0.5
        rr = np.sqrt(xx * xx + yy * yy)
        phi = np.arctan2(yy, xx) + self.direction * np.pi / 180
        xx, yy = rr * np.cos(phi) / 10, rr * np.sin(phi) / 10
        xx, yy = xx + self.x, yy + self.y
        return xx, yy
