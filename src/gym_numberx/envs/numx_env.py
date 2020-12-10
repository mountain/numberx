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
        self.game = Serengeti({}, device=device)

        self.action_space = self.game.action_space()
        self.observation_space = self.game.state_space()

    def state(self):
        return self.observation_space

    def step(self, action):
        self.game.fire_step_event(action=self.game.action())

        reward = self.game.reward()

        _state = self.state()

        episode_over = self.game.exit_condition() or self.game.force_condition()

        return _state, reward, episode_over, {}

    def reset(self):
        self.game.reset()
        return self.state()

    def render(self, mode='rgb_array', close=False):
        grayscale = self.observation_space
        return np.concatenate([grayscale, grayscale, grayscale], axis=2)
