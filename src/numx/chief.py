import numpy as np

from numx.affordable import Affordable, NOOP, get_action


class Chief(Affordable):
    def __init__(self, ctx, x, y):
        super().__init__(ctx, 'chief')
        self.ctx = ctx

        self.direction = 0
        self.speed = 0.01
        self.x = x
        self.y = y

    def available_actions(self):
        return (NOOP, 'fw', 'lf', 'rt', 'gu', 'gd')

    def reset(self):
        super(Chief, self).reset()

    def act(self, action):
        action = get_action(self.ctx, action=action).chief
        if action == 'lf':
            self.lf()
        elif action == 'rt':
            self.rt()
        elif action == 'gu':
            self.gu()
        elif action == 'gd':
            self.gd()

        self.fw()

    def fw(self):
        dx = self.speed * np.cos(self.direction * np.pi / 180)
        dy = self.speed * np.sin(self.direction * np.pi / 180)
        self.x = self.x + dx
        self.y = self.y + dy

    def lf(self):
        self.direction = (self.direction + 1) % 360

    def rt(self):
        self.direction = (self.direction + 359) % 360

    def gu(self):
        self.speed = self.speed * 2

    def gd(self):
        self.speed = self.speed / 2