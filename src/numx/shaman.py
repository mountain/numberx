import numpy as np

from numx.affordable import Affordable, NOOP, IDLE, get_action


class Shaman(Affordable):
    def __init__(self, ctx, name, width, height):
        super(Shaman, self).__init__(ctx, name)
        self.width = width
        self.height = height
        self.canvas = np.zeros([height, width])

        self.pen_status = 'up'
        self.x = 0
        self.y = 0

    def available_actions(self):
        return (NOOP, 'pu', 'pd', 'up', 'dn', 'lf', 'rt', 'rs')

    def reset(self):
        super(Shaman, self).reset()
        self.canvas = np.zeros([self.height, self.width])

    def act(self, action):
        if self.name() == 'lshaman':
            action = get_action(self.ctx, action=action).lshaman
        else:
            action = get_action(self.ctx, action=action).rshaman

        if action == 'pu':
            self.pu()
        elif action == 'pd':
            self.pd()
        elif action == 'up':
            self.up()
        elif action == 'dn':
            self.dn()
        elif action == 'lf':
            self.lf()
        elif action == 'rt':
            self.rt()
        elif action == 'rs':
            self.rs()

    def pu(self):
        self.pen_status = 'up'

    def pd(self):
        self.pen_status = 'dn'

    def up(self):
        self.y -= 1
        if self.y < 0:
            self.y = 0

        if self.pen_status == 'dn':
            self.canvas[self.y, self.x] = 1.0

    def dn(self):
        self.y += 1
        if self.y > self.height - 1:
            self.y = self.height - 1

        if self.pen_status == 'dn':
            self.canvas[self.y, self.x] = 1.0

    def lf(self):
        self.x -= 1
        if self.x < 0:
            self.x = 0

        if self.pen_status == 'dn':
            self.canvas[self.y, self.x] = 1.0

    def rt(self):
        self.x += 1
        if self.x > self.width - 1:
            self.x = self.width - 1

        if self.pen_status == 'dn':
            self.canvas[self.y, self.x] = 1.0

    def rs(self):
        self.canvas = np.zeros([self.height, self.width])