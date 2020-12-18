import numpy as np
import itertools

from affordable.affordable import Affordable, get_action


phi = (np.sqrt(5) - 1) / 2.0

ACTIONS = ('lf', 'rt', 'gu', 'gd', 'sc')


class Chief(Affordable):
    def __init__(self, ctx, x, y):
        super().__init__(ctx, 'chief')
        self.ctx = ctx

        self.direction = 360 * np.random.rand()
        self.speed = 0.0
        self.x = x
        self.y = y
        self.dt = self.ctx['dt'] if 'dt' in self.ctx else 0.0001

    def available_actions(self):
        return list(itertools.product(ACTIONS))

    def reset(self):
        super(Chief, self).reset()

    def act(self, action):
        action = get_action(self.ctx, action=action).chief
        for a in action:
            if a == 'lf':
                self.lf()
            elif a == 'rt':
                self.rt()
            elif a == 'gu':
                self.gu()
            elif a == 'gd':
                self.gd()
            elif a == 'sc':
                self.sc()

            self.fw()

    def fw(self):
        dx = self.speed * np.cos(self.direction * np.pi / 180)
        dy = self.speed * np.sin(self.direction * np.pi / 180)
        self.x = self.x + dx
        self.y = self.y + dy
        r = np.sqrt(1e-8 + self.x * self.x + self.y * self.y)
        if r > 7:
            self.x = self.x / r * 7
            self.y = self.y / r * 7

    def lf(self):
        self.direction = (self.direction - 180 / np.pi * self.dt) % 360

    def rt(self):
        self.direction = (self.direction + 180 / np.pi * self.dt) % 360

    def gu(self):
        self.speed = self.speed + 1.0 * self.dt
        if self.speed > 3:
            self.speed = 3

    def gd(self):
        self.speed = self.speed - 1.0 * self.dt
        if self.speed < -3:
            self.speed = -3

    def sc(self):
        self.speed = self.speed * phi
