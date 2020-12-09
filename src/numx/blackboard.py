import numpy as np

from numx.affordable import Affordable, NOOP, IDLE, get_action


class Blackboard(Affordable):
    def __init__(self, ctx, width, height):
        super(Blackboard, self).__init__(ctx, 'blackboard')
        self.width = width
        self.height = height
        self.data = np.zeros([height, width])

        self.pen_status = 'up'
        self.x = 0
        self.y = 0

    def available_actions(self):
        return (NOOP, 'pu', 'pd', 'up', 'dn', 'lf', 'rt')

    def available_states(self):
        return (IDLE, )

    def reset(self):
        super(Blackboard, self).reset()
        self.data = np.zeros([self.height, self.width])

    def act(self, action):
        action = get_action(self.ctx, action=action).blackboard
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

    def pu(self):
        self.pen_status = 'up'

    def pd(self):
        self.pen_status = 'dn'

    def up(self):
        self.y -= 1
        if self.y < 0:
            self.y = 0

        if self.pen_status == 'dn':
            self.data[self.y, self.x] = 1.0

    def dn(self):
        self.y += 1
        if self.y > self.height - 1:
            self.y = self.height - 1

        if self.pen_status == 'dn':
            self.data[self.y, self.x] = 1.0

    def lf(self):
        self.x -= 1
        if self.x < 0:
            self.x = 0

        if self.pen_status == 'dn':
            self.data[self.y, self.x] = 1.0

    def rt(self):
        self.x += 1
        if self.x > self.width - 1:
            self.x = self.width - 1

        if self.pen_status == 'dn':
            self.data[self.y, self.x] = 1.0