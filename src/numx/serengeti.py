import numpy as np

from numx.game import AbstractGame
from numx.tribe import Tribe
from numx.shaman import Shaman
from numx.chief import Chief


class Serengeti(AbstractGame):
    def __init__(self, ctx, alpha=0.01, device='cpu'):
        super(Serengeti, self).__init__(ctx)
        self.device = device

        self.steps = 0

        self.alpha = alpha
        self.xmin = -2
        self.xmax = +2
        self.ymin = -2
        self.ymax = +2
        self.peakx = np.random.random() * (self.xmax - self.xmin) + self.xmin
        self.peaky = np.random.random() * (self.ymax - self.ymin) + self.ymin

        self.berries_left = np.zeros((16, 16))
        self.berries_right = np.zeros((16, 16))
        self.canvas_left = np.zeros((16, 16))
        self.canvas_right = np.zeros((16, 16))

        tribex = np.random.random() * (self.xmax - self.xmin) + self.xmin
        tribey = np.random.random() * (self.ymax - self.ymin) + self.ymin
        self.tribe = Tribe(16, 16, tribex, tribey, 0.0)

    def prosperity(self, xx, yy):
        dx = xx - self.peakx
        dy = yy - self.peaky
        p = np.exp(- dx * dx - dy * dy) * (1 - self.alpha)
        return p

    def probability(self, xx, yy):
        noise = self.alpha * np.random.random()
        return self.prosperity(xx, yy) + noise

    def collected_berries(self, xx, yy):
        prob = self.probability(xx, yy)
        sample = np.random.rand(*xx.shape)
        berries = sample < prob
        return berries

    def score(self):
        dx = self.tribe.x - self.peakx
        dy = self.tribe.y - self.peaky
        return - np.log(1e-8 + dx * dx + dy * dy)

    def apply_effect(self):
        self.steps += 1
        self.apply_tribe_effect()
        self.apply_shaman_effect()
        self.apply_chief_effect()

    def apply_tribe_effect(self):
        np.copyto(self.berries_left, self.collected_berries(*self.tribe.left_wing()))
        np.copyto(self.berries_right, self.collected_berries(*self.tribe.right_wing()))

    def apply_shaman_effect(self):
        np.copyto(self.canvas_left, self.shaman_left.canvas)
        np.copyto(self.canvas_right, self.shaman_right.canvas)

    def apply_chief_effect(self):
        self.tribe.x = self.chief.x
        self.tribe.y = self.chief.y
        self.tribe.direction = self.chief.direction

    def reset(self):
        super(Serengeti, self).reset()

    def all_affordables(self):
        self.chief = Chief(self.ctx, 16, 16)
        self.shaman_left = Shaman(self.ctx, 'lshaman', 16, 16)
        self.shaman_right = Shaman(self.ctx, 'rshaman', 16, 16)

        return self.chief, self.shaman_left, self.shaman_right

    def state_space(self):
        berries = np.concatenate(
            (self.berries_left,  self.berries_right),
            axis=1
        )
        canvaz = np.concatenate(
            (self.canvas_left, self.canvas_right),
            axis=1
        )
        state = np.concatenate(
            (berries, canvaz),
            axis=0
        )
        return state

    def reward(self):
        return self.score()

    def exit_condition(self):
        score = self.score()
        steps = self.steps
        return steps > 2000 or score > 5

    def force_condition(self):
        return False
