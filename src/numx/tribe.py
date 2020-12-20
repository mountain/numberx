import numpy as np


class Tribe:
    def __init__(self, width, height, x, y, d):
        self.width = width
        self.height = height
        self.direction = d
        self.x = x
        self.y = y

        self.map = np.zeros((4 * width + 3, 4 * height + 3))
        self.ratio = 7 / (4 * width + 3)

    def head_left(self):
        xx, yy = np.meshgrid(np.linspace(-1, 1, self.height), np.linspace(-1, 1, self.width))
        xx, yy = xx - 1, yy + 1
        rr = np.sqrt(xx * xx + yy * yy)
        phi = np.arctan2(yy, xx) + self.direction * np.pi / 180
        xx, yy = rr * np.cos(phi) / 10, rr * np.sin(phi) / 10
        xx, yy = xx + self.x, yy + self.y
        return xx, yy

    def head_right(self):
        xx, yy = np.meshgrid(np.linspace(-1, 1, self.height), np.linspace(-1, 1, self.width))
        xx, yy = xx + 1, yy + 1
        rr = np.sqrt(xx * xx + yy * yy)
        phi = np.arctan2(yy, xx) + self.direction * np.pi / 180
        xx, yy = rr * np.cos(phi) / 10, rr * np.sin(phi) / 10
        xx, yy = xx + self.x, yy + self.y
        return xx, yy

    def rear_left(self):
        xx, yy = np.meshgrid(np.linspace(-1, 1, self.height), np.linspace(-1, 1, self.width))
        xx, yy = xx - 1, yy - 1
        rr = np.sqrt(xx * xx + yy * yy)
        phi = np.arctan2(yy, xx) + self.direction * np.pi / 180
        xx, yy = rr * np.cos(phi) / 10, rr * np.sin(phi) / 10
        xx, yy = xx + self.x, yy + self.y
        return xx, yy

    def rear_right(self):
        xx, yy = np.meshgrid(np.linspace(-1, 1, self.height), np.linspace(-1, 1, self.width))
        xx, yy = xx + 1, yy - 1
        rr = np.sqrt(xx * xx + yy * yy)
        phi = np.arctan2(yy, xx) + self.direction * np.pi / 180
        xx, yy = rr * np.cos(phi) / 20, rr * np.sin(phi) / 20
        xx, yy = xx + self.x, yy + self.y
        return xx, yy

    def draw_map(self, sourcex, sourcey, targetx, targety):
        IX = np.array((3.5 + np.linspace(sourcex, targetx, num=50)) / self.ratio, dtype=np.int)
        IY = np.array((3.5 - np.linspace(sourcey, targety, num=50)) / self.ratio, dtype=np.int)
        IX = IX * (IX > 0) * (IX < self.width * 4 + 3)
        IY = IY * (IY > 0) * (IY < self.width * 4 + 3)
        self.map[IY, IX] = 1.0

    def clear_map(self):
        self.map = np.zeros((4 * self.width + 3, 4 * self.height + 3))
