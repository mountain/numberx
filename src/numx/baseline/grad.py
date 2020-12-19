#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import collections
import random

Action = collections.namedtuple('Action', ['chief', 'shaman_hl', 'shaman_hr', 'shaman_rl', 'shaman_rr'])


class Swirl:
    def __init__(self, ):
        self.reset()

    def reset(self):
        self.x = 0
        self.y = 0
        self.bounday = 0
        self.direction = 'up'
        self.action = ['rs']

    def step(self):
        self.action.append(self.direction)

        if self.direction == 'up':
            self.x -= 1
        elif self.direction == 'dn':
            self.x += 1
        elif self.direction == 'lf':
            self.y -= 1
        elif self.direction == 'rt':
            self.y += 1

        if (self.x < - self.bounday or self.x > self.bounday) or (self.y < - self.bounday or self.y > self.bounday):
            self.bounday += 1
            if self.direction == 'up':
                self.direction = 'rt'
            elif self.direction == 'dn':
                self.direction = 'lf'
            elif self.direction == 'lf':
                self.direction = 'up'
            elif self.direction == 'rt':
                self.direction = 'dn'

times = 0
wait_time = 10
rand_step = 1
hlaction, hraction, rlaction, rraction = Swirl(), Swirl(), Swirl(), Swirl()


def swirl(obs, reward, done):
    global times, wait_time, rand_step

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
    for _ in range(brrhl):
        hlaction.step()
    for _ in range(brrhr):
        hraction.step()
    for _ in range(brrrl):
        rlaction.step()
    for _ in range(brrrr):
        rraction.step()

    cnthl = int(np.sum(brdhl))
    cnthr = int(np.sum(brdhr))
    cntrl = int(np.sum(brdrl))
    cntrr = int(np.sum(brdrr))

    dxp = cntrr - cnthl
    dyp = cnthr - cntrl
    dx, dy = dyp + dxp, dyp - dxp
    dr = np.sqrt(dx * dx + dy * dy)

    if dr > 15.0:
        rand_step = 1.0
        wait_time -= 1
        if wait_time < 3:
            wait_time = 3
    elif dr != 0:
        wait_time += 1
        if wait_time > 20:
            wait_time = 20
        rand_step = rand_step * 2
        if rand_step > 1000:
            rand_step = 1000

    rndx, rndy = int(random.random() * rand_step - rand_step / 2), int(random.random() * rand_step - rand_step / 2)

    caction = (dx + rndx) / 10, (dy + rndy) / 10
    action = Action._make([caction, hlaction.action, hraction.action, rlaction.action, rraction.action])

    times = times + 1
    if times % wait_time == 0:
        hlaction.reset()
        hraction.reset()
        rlaction.reset()
        rraction.reset()

    return action

