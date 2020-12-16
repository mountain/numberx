import numpy as np
import cv2

from affordable.game import AbstractGame
from numx.tribe import Tribe
from numx.shaman import Shaman
from numx.chief import Chief


class Serengeti(AbstractGame):
    def __init__(self, ctx, alpha=0.5, beta=0.0025, mode='hidden', device='cpu'):
        super(Serengeti, self).__init__(ctx, 'serengeti')
        self.device = device
        self.mode = mode

        self.steps = 0

        self.alpha = alpha
        self.beta = beta

        self.xmin = -2
        self.xmax = +2
        self.ymin = -2
        self.ymax = +2
        self.peakx = np.random.random() * (self.xmax - self.xmin) + self.xmin
        self.peaky = np.random.random() * (self.ymax - self.ymin) + self.ymin

        size = self.ctx['size'] if 'size' in self.ctx else 64
        self.size = size
        self.berries_left = np.zeros((size, size))
        self.berries_right = np.zeros((size, size))
        self.canvas_left = np.zeros((size, size))
        self.canvas_right = np.zeros((size, size))
        self.map = np.zeros((2 * size, 2 * size))

        self.score_img = np.zeros((self.size // 2, 2 * size), dtype=np.uint8)
        self.total_score = 0.0

        IX, IY = np.meshgrid(
            np.linspace(self.xmin, self.xmax, num=2 * size),
            np.linspace(self.ymin, self.ymax, num=2 * size),
        )
        self.XS = (IX < self.peakx + 4 / self.size) * (IX > self.peakx - 4 / self.size)
        self.YS = (IY < self.peaky + 4 / self.size) * (IY > self.peaky - 4 / self.size)

        tribex = np.random.random() * (self.xmax - self.xmin) + self.xmin
        tribey = np.random.random() * (self.ymax - self.ymin) + self.ymin
        self.tribe = Tribe(size, size, tribex, tribey, 0.0)

    def prosperity(self, xx, yy):
        dx = xx - self.peakx
        dy = yy - self.peaky
        p = np.exp(- dx * dx - dy * dy) * (1 - self.alpha)
        return p

    def probability(self, xx, yy):
        noise = self.alpha * np.random.random()
        return self.beta * (self.prosperity(xx, yy) + noise)

    def collected_berries(self, xx, yy):
        prob = self.probability(xx, yy)
        sample = np.random.rand(*xx.shape)
        berries = sample < prob
        return berries

    def score(self):
        sl = np.sum(self.berries_left)
        sr = np.sum(self.berries_right)
        st = sl + sr
        self.total_score = self.total_score + st

        if self.mode == 'revealed':
            self.score_img = np.zeros((self.size // 2, 2 * self.size), dtype=np.uint8)
            cv2.putText(self.score_img, '%03d' % sl, (2 * self.size // 8 * 1, self.size // 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(self.score_img, '%03d' % sr, (2 * self.size // 8 * 3, self.size // 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(self.score_img, '%06d' % self.total_score, (2 * self.size // 8 * 5, self.size // 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        return self.total_score / 10000

    def apply_effect(self):
        self.steps += 1
        self.apply_shaman_effect()
        self.apply_chief_effect()
        if self.steps % 20 == 0:
            self.apply_tribe_effect()
            self.score()

        if self.mode == 'revealed':
            self.map[self.YS * self.XS] = 1.0

    def apply_tribe_effect(self):
        np.copyto(self.berries_left, self.collected_berries(*self.tribe.left_wing()))
        np.copyto(self.berries_right, self.collected_berries(*self.tribe.right_wing()))

    def apply_shaman_effect(self):
        np.copyto(self.canvas_left, self.shaman_left.canvas)
        np.copyto(self.canvas_right, self.shaman_right.canvas)

    def apply_chief_effect(self):
        sourcex = self.tribe.x
        sourcey = self.tribe.y
        targetx = self.chief.x
        targety = self.chief.y

        self.tribe.x = targetx
        self.tribe.y = targety
        self.tribe.direction = self.chief.direction

        self.tribe.draw_map(sourcex, sourcey, targetx, targety)
        np.copyto(self.map, self.tribe.map)

    def apply_tribe_effect(self):
        np.copyto(self.berries_left, self.collected_berries(*self.tribe.left_wing()))
        np.copyto(self.berries_right, self.collected_berries(*self.tribe.right_wing()))

    def reset(self):
        super(Serengeti, self).reset()
        self.tribe.clear_map()
        self.map = np.zeros((2 * self.size, 2 * self.size))
        self.score_img = np.zeros((self.size // 2, 2 * self.size), dtype=np.uint8)
        self.total_score = 0
        self.steps = 0

    def all_affordables(self):
        size = self.ctx['size'] if 'size' in self.ctx else 64
        xmin = self.ctx['xmin'] if 'xmin' in self.ctx else -3.5
        xmax = self.ctx['xmax'] if 'xmax' in self.ctx else 3.5
        ymin = self.ctx['ymin'] if 'ymin' in self.ctx else -3.5
        ymax = self.ctx['ymax'] if 'ymax' in self.ctx else 3.5

        x = 2 * (np.random.random() * (xmax - xmin) + xmin)
        y = 2 * (np.random.random() * (ymax - ymin) + ymin)
        self.chief = Chief(self.ctx, x, y)

        self.shaman_left = Shaman(self.ctx, 'lshaman', size, size)
        self.shaman_right = Shaman(self.ctx, 'rshaman', size, size)

        return self.chief, self.shaman_left, self.shaman_right

    def state_space(self):
        self.apply_effect()

        berries = np.concatenate(
            (self.berries_left,  self.berries_right),
            axis=1
        )
        canvaz = np.concatenate(
            (self.canvas_left, self.canvas_right),
            axis=1
        )
        score = np.array(self.score_img, dtype=np.float) / 255

        if self.mode == 'revealed':
            state = np.concatenate(
                (berries, canvaz, self.map, score),
                axis=0
            )
        else:
            state = np.concatenate(
                (berries, canvaz, self.map),
                axis=0
            )

        return state

    def reward(self):
        return self.score()

    def exit_condition(self):
        score = self.total_score / 10000
        steps = self.steps
        return steps > 200 or score > 1

    def force_condition(self):
        return False
