# -*- coding: utf-8 -*-

import numpy as np

import gym

from gym import utils
from numx.numberx import NumberXGame

import logging
logger = logging.getLogger(__name__)


class NumberxEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, device=None):
        gym.Env.__init__(self)
        self.game = NumberXGame(device=device)

        self.action_space = self.game.action_space()
        self.observation_space = self.game.state_space()

    def state(self):
        return self.observation_space

    def step(self, action):
        self.game.fire_step_event(action=self.game.action())

        reward = self.game.reward().detach()

        _state = self.state()

        episode_over = self.game.exit_condition() or self.game.force_condition()

        return _state, reward, episode_over, {}

    def reset(self):
        self.game.reset()
        return self.state()

    def render(self, mode='rgb_array', close=False):
        data = np.array(self.game.agent.blackboard * 255, dtype=np.uint8)
        grayscale = data.reshape(self.game.agent.height, self.game.agent.width, 1)
        view = np.concatenate([grayscale, grayscale, grayscale], axis=2)

        data = self.game.count_test.data
        arr1 = np.array(data, dtype=np.uint8)

        return np.concatenate([view, arr1], axis=1)
