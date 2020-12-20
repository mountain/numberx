#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import random

from numx.shaman import ACTIONS as shaman_actions
from numx.chief import ACTIONS as chief_actions


Action = collections.namedtuple('Action', ['chief', 'shaman_hl', 'shaman_hr', 'shaman_rl', 'shaman_rr'])


def rnd(obs, reward, done):
    return Action._make([
        (10 * (random.random() - 0.5), 10 * (random.random() - 0.5)),
        random.sample(shaman_actions, 1)[0],
        random.sample(shaman_actions, 1)[0],
        random.sample(shaman_actions, 1)[0],
        random.sample(shaman_actions, 1)[0],
    ])

