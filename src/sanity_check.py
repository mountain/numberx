#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym_numberx

import gym
import time
import argparse

from gym import wrappers, logger


parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=2000, help="number of epochs of training")
opt = parser.parse_args()


if __name__ == '__main__':

    logger.set_level(logger.INFO)

    env = gym.make('numberx-serengeti-v0', mode='revealed')

    outdir = 'results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(int(time.time()))
    game = env.game

    episode_count = opt.n
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = game.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
            env.render(mode='rgb_array')

    # Close the env and write monitor result info to disk
    env.close()
