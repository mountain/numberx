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
        self.x = x
        self.y = y
        self.dt = self.ctx['dt'] if 'dt' in self.ctx else 0.01

    def available_actions(self):
        return list(itertools.product(ACTIONS))

    def reset(self):
        super(Chief, self).reset()

    def act(self, action):
        action = get_action(self.ctx, action=action).chief
        du, dv = action
        self.direction = np.arctan2(dv, du)
        self.x = self.x + du * self.dt
        self.y = self.y + dv * self.dt
        r = np.sqrt(1e-8 + self.x * self.x + self.y * self.y)
        if r > 7:
            self.x = self.x / r * 7
            self.y = self.y / r * 7
