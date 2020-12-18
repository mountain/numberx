#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym_numberx

import gym
import time

from gym import wrappers, logger

from numx.baseline.rnd import rnd
from numx.baseline.grad import levy

policy = levy


if __name__ == '__main__':

    logger.set_level(logger.INFO)
    outdir = 'results'
    env = gym.make('numberx-serengeti-v0', mode='revealed')
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(int(time.time()))
    game = env.game
    game.policy = policy
    obs = env.reset()
    reward = 0
    done = False
    while True:
        action = game.act(obs, reward, done)
        ob, reward, done, _ = env.step(action)
        if done:
            break
