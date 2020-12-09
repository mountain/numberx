# -*- coding: utf-8 -*-

from gym.envs.registration import register


register(
    id='numberx-v0',
    entry_point='gym_numberx.envs:NumberxEnv',
)
