import unittest

import numpy as np

from numx.serengeti import Serengeti

class TestGame(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_probability(self):
        for alpha in np.linspace(0.2, 0.8, 16):
            print(alpha)

            self.game = Serengeti({}, alpha=alpha)
            xs, ys = self.game.IX, self.game.IY
            self.game.peaky = 0.0
            self.game.peaky = 0.0
            p = self.game.probability(xs, ys)

            r = np.sqrt(xs * xs + ys * ys)
            s = (r > (3 - 0.05)) * (r < (3 + 0.05))
            b = np.mean(s * p) * self.game.size * self.game.size
            self.assertGreaterEqual(b, -1.0)
            self.assertLessEqual(b, 4.0)
