# -*- coding: utf-8 -*-

import numpy as np

import gym

from gym import utils
from numx.serengeti import Serengeti

import logging

logger = logging.getLogger(__name__)


class NumXSerengetiEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, device=None):
        gym.Env.__init__(self)
        self.steps = 0
        self.game = Serengeti({}, device=device)
        self.game.add_step_handler(self)

        self.action_space = self.game.action_space()
        self.observation_space = self.state()

    def state(self):
        grayscale = np.array(self.game.state_space() * 255, dtype=np.uint8)
        grayscale = np.expand_dims(grayscale, axis=0)
        return np.concatenate([grayscale, grayscale, grayscale], axis=0)

    def step(self, action):
        reward = self.game.reward()
        state = self.state()
        episode_over = self.game.exit_condition() or self.game.force_condition()

        self.game.act(state, reward, episode_over)

        return state, reward, episode_over, {}

    def reset(self):
        self.game.reset()
        return self.state()

    def render(self, mode='rgb_array', close=False):
        state0 = self.state()
        state1 = np.rollaxis(state0, 1, 0)
        state2 = np.rollaxis(state1, 2, 1)
        return state2
