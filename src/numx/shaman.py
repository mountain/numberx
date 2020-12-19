import numpy as np
import itertools

from affordable.affordable import Affordable, get_action


ACTIONS = ('np', 'up', 'dn', 'lf', 'rt', 'rs')


class Shaman(Affordable):
    def __init__(self, ctx, name, width, height):
        super(Shaman, self).__init__(ctx, name)
        self.width = width
        self.height = height
        self.canvas = np.zeros([height, width])
        self.x = width // 2
        self.y = height // 2

    def available_actions(self):
        return list(itertools.product(ACTIONS))

    def reset(self):
        super(Shaman, self).reset()
        self.canvas = np.zeros([self.height, self.width])
        self.x = self.width // 2
        self.y = self.height // 2

    def act(self, action):
        nm = self.name()
        if nm == 'shaman_hl':
            action = get_action(self.ctx, action=action).shaman_hl
        elif nm == 'shaman_hr':
            action = get_action(self.ctx, action=action).shaman_hr
        elif nm == 'shaman_rl':
            action = get_action(self.ctx, action=action).shaman_rl
        elif nm == 'shaman_rr':
            action = get_action(self.ctx, action=action).shaman_rr

        for a in action:
            if a == 'up':
                self.up()
            elif a == 'dn':
                self.dn()
            elif a == 'lf':
                self.lf()
            elif a == 'rt':
                self.rt()
            elif a == 'rs':
                self.rs()

    def up(self):
        self.y -= 1
        if self.y < 0:
            self.y = 0

        self.canvas[self.y, self.x] = 1.0

    def dn(self):
        self.y += 1
        if self.y > self.height - 1:
            self.y = self.height - 1

        self.canvas[self.y, self.x] = 1.0

    def lf(self):
        self.x -= 1
        if self.x < 0:
            self.x = 0

        self.canvas[self.y, self.x] = 1.0

    def rt(self):
        self.x += 1
        if self.x > self.width - 1:
            self.x = self.width - 1

        self.canvas[self.y, self.x] = 1.0

    def rs(self):
        self.canvas = np.zeros([self.height, self.width])
        self.x = self.width // 2
        self.y = self.height // 2

