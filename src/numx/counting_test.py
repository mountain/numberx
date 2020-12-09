import numpy as np
import cv2
import collections

from numx.affordable import Affordable, NOOP, IDLE, get_action


class CountTest(Affordable):
    def __init__(self, ctx, width, height, max_round, upper_bound):
        super(CountTest, self).__init__(ctx, 'counttest')
        self.width = width
        self.height = height
        self.data = np.zeros([height, width, 3])
        self.stack = collections.deque([], maxlen=max_round)

        self.max_round = max_round
        self.upper_bound = upper_bound

        for _ in range(max_round):
            self.stack.appendleft(np.zeros([height, width, 3]))

        self.counter = 0
        self.tests = 0
        self.blue = 0
        self.red = 0
        self.sum_blue = 0
        self.sum_red = 0

    def available_actions(self):
        return (NOOP,)

    def available_states(self):
        return (IDLE, )

    def reset(self):
        super(CountTest, self).reset()
        for _ in range(self.max_round):
            self.stack.appendleft(np.zeros([self.height, self.width, 3]))
        self.data = np.zeros([self.height, self.width, 3])
        self.counter = 0
        self.blue = 0
        self.red = 0
        self.sum_blue = 0
        self.sum_red = 0

    def act(self, action):
        if self.counter % 50 == 0:
            self.tests += 1
            self.blue = np.random.randint(1, self.upper_bound)
            self.red = np.random.randint(1, self.upper_bound)

            self.stack.appendleft(self.data)
            self.data = np.zeros([self.height, self.width, 3])

            for _ in range(self.blue):
                cv2.circle(
                    self.data,
                    (np.random.randint(20, self.height - 20), np.random.randint(20, self.width - 20)), # center
                    np.random.randint(5, 20), # radius
                    (255, 0, 0), #BGR
                    2, # thickness = -1 to fill
                )

            for _ in range(self.blue):
                cv2.circle(
                    self.data,
                    (np.random.randint(20, self.height - 20), np.random.randint(20, self.width - 20)), # center
                    np.random.randint(5, 20), # radius
                    (0, 0, 255), #BGR
                    2, # thickness = -1 to fill
                )

            self.sum_blue = self.sum_blue + self.blue
            self.sum_red = self.sum_red + self.red

        if self.tests >= self.max_round:
            self.reset()

        self.counter = self.counter + 1

    def get_stack(self):
        stack = list([e for e in self.stack])
        return stack

    def answer(self):
        if self.sum_red > self.sum_blue:
            return 'red'
        elif self.sum_blue > self.sum_red:
            return 'blue'
        else:
            return 'all'

    def new_round(self):
        self.tests = 0
