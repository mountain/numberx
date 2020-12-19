#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym_numberx

import gym
import time
import numpy as np

from gym import wrappers, logger

from numx.baseline.rnd import rnd
from numx.baseline.grad import swirl

policy = rnd


if __name__ == '__main__':

    logger.set_level(logger.INFO)

    env = gym.make('numberx-serengeti-v0', mode='revealed')
    env.seed(int(time.time()))
    game = env.game
    game.policy = policy

    results = []
    episode_count = 300
    for i in range(episode_count):
        reward = 0
        accumulated = 0
        ob = env.reset()
        done = False
        while True:
            action = game.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            accumulated += reward
            if done:
                break
        print(accumulated)
        results.append(accumulated)

    results = np.array(results)
    print(results.mean(), results.std())

    outdir = 'results'
    env = gym.make('numberx-serengeti-v0', mode='revealed')
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(int(time.time()))
    game = env.game
    game.policy = policy
    ob = env.reset()
    reward = 0
    done = False
    while True:
        action = game.act(ob, reward, done)
        ob, reward, done, _ = env.step(action)
        if done:
            break
