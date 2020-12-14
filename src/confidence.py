# -*- coding: utf-8 -*-

import gym_numberx

import gym
import numpy as np

import matplotlib.pyplot as plt


env = gym.make('numberx-serengeti-v0')
xx, yy = np.meshgrid(np.linspace(-5, 5, 256), np.linspace(-5, 5, 256))


def significent(jx, ix):
    env.game.tribe.x = xx[jx, ix]
    env.game.tribe.y = yy[jx, ix]
    env.game.tribe.direction = np.arange(360)
    lxx, lyy = env.game.tribe.left_wing()
    rxx, ryy = env.game.tribe.right_wing()
    lsum = np.sum(env.game.alpha / 2 + (1 - env.game.alpha) * env.game.prosperity(lxx, lyy))
    rsum = np.sum(env.game.alpha / 2 + (1 - env.game.alpha) * env.game.prosperity(rxx, ryy))
    return (lsum - rsum).max()


sgnf = np.vectorize(significent)
IX, IY = np.meshgrid(np.linspace(0, 255, 256), np.linspace(0, 255, 256))
IX = np.array(IX, dtype=np.int)
IY = np.array(IY, dtype=np.int)

plt.scatter(sgnf(IY, IX))

plt.show()