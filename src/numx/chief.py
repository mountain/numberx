import numpy as np

from numx.affordable import Affordable, NOOP, get_action


phi = (np.sqrt(5) - 1) / 2.0


class Chief(Affordable):
    def __init__(self, ctx, x, y):
        super().__init__(ctx, 'chief')
        self.ctx = ctx

        self.direction = 0
        self.speed = 0.5
        # self.rotation = 0.0
        self.x = x
        self.y = y

    def available_actions(self):
        return 'lf', 'rt', 'gu', 'gd'

    def reset(self):
        super(Chief, self).reset()

    def act(self, action):
        action = get_action(self.ctx, action=action).chief
        #if action == 'kp':
        #    self.kp()
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
        r = np.sqrt(1e-8 + self.x * self.x + self.y * self.y)
        if r > 7:
            self.x = self.x / r * 7
            self.y = self.y / r * 7

    # def kp(self):
    #     self.direction = (self.direction + self.rotation) % 360
    #     self.rotation = self.rotation * 0.618

    def lf(self):
        # self.direction = (self.direction + self.rotation) % 360
        # self.rotation = self.rotation * 0.618
        # self.rotation = self.rotation - 360
        self.direction = (self.direction - 360 * phi + 360) % 360

    def rt(self):
        # self.direction = (self.direction + self.rotation) % 360
        # self.rotation = self.rotation * 0.618
        # self.rotation = self.rotation + 360
        self.direction = (self.direction + 360 * phi) % 360

    def gu(self):
        self.speed = self.speed + 0.5
        if self.speed > 1.0:
            self.speed = 1.0

    def gd(self):
        self.speed = self.speed / 1.618
