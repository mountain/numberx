#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import collections
import random

Action = collections.namedtuple('Action', ['chief', 'shaman_hl', 'shaman_hr', 'shaman_rl', 'shaman_rr'])

times = 0
last_reward, diff, last_diff = 0, 0, 0


def levy(obs, reward, done):
    global last_reward, diff, last_diff, times

    obs = obs[0, :, :] / 255
    h, w = obs.shape
    h, w = (w - 8) // 6, (w - 8) // 6
    lndhl, lndhr = obs[1+h*0:1+h*1, 1:1+w], obs[1+h*0:1+h*1, 2+w:2+w*2]
    lndrl, lndrr = obs[2+h*1:2+h*2, 1:1+w], obs[2+h*1:2+h*2, 2+w:2+w*2]
    brdhl, brdhr = obs[3+h*2:3+h*3, 1:1+w], obs[3+h*2:3+h*3, 2+w:2+w*2]
    brdrl, brdrr = obs[4+h*3:4+h*4, 1:1+w], obs[4+h*3:4+h*4, 2+w:2+w*2]

    brrhl = int(np.sum(lndhl))
    brrhr = int(np.sum(lndhr))
    brrrl = int(np.sum(lndrl))
    brrrr = int(np.sum(lndrr))
    hlaction, hraction, rlaction, rraction = [], [], [], []
    for _ in range(brrhl):
        hlaction.append(random.sample(['lf', 'rt', 'up', 'dn'], 1)[0])
    for _ in range(brrhr):
        hraction.append(random.sample(['lf', 'rt', 'up', 'dn'], 1)[0])
    for _ in range(brrrl):
        rlaction.append(random.sample(['lf', 'rt', 'up', 'dn'], 1)[0])
    for _ in range(brrrr):
        rraction.append(random.sample(['lf', 'rt', 'up', 'dn'], 1)[0])

    cnthl = int(np.sum(brdhl))
    cnthr = int(np.sum(brdhr))
    cntrl = int(np.sum(brdrl))
    cntrr = int(np.sum(brdrr))
    total = cnthl + cnthr + cntrl + cntrr
    dx = (cnthr + cntrr) - (cnthl + cntrl)
    dy = (cnthl + cnthr) - (cntrl + cntrr)
    ag = np.arctan2(dy, dx) * 180 / np.pi
    dr = np.sqrt(dx * dx + dy * dy)

    caction = ['sc', 'sc', 'sc', 'sc', 'sc', 'sc']
    if dy > 0:
        if ag > 90:
            for _ in range(int((ag - 90))):
                caction.append('lf')
        if ag < 90:
            for _ in range(int((90 - ag))):
                caction.append('rt')
        for _ in range(int(dr)):
            caction.append('gu')
    if dy < 0:
        if ag > -90:
            for _ in range(int((ag + 90))):
                caction.append('rt')
        if ag < -90:
            for _ in range(int((- ag - 90))):
                caction.append('lf')
        for _ in range(int(dr)):
            caction.append('gd')

    if last_diff < diff and last_diff > 0 and diff > 0:
        caction.append('sc')

    times = times + 1
    if times % 10 == 0:
        hlaction.append('rs')
        hraction.append('rs')
        rlaction.append('rs')
        rraction.append('rs')

    last_diff = diff
    diff = reward - last_reward
    last_reward = reward

    return Action._make([caction, hlaction, hraction, rlaction, rraction])
