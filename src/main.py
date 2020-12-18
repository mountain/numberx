# -*- coding: utf-8 -*-

import gym_numberx

import os
import gym
import numpy as np
import torch as th
import argparse
import tianshou as ts
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from pathlib import Path
from gym import wrappers, logger
from tianshou.utils.net.discrete import DQN
from tianshou.utils.net.common import Recurrent

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=256, help="number of epochs of training")
parser.add_argument("-g", type=str, default='0', help="index of gpu")
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
if cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.g
device = torch.device(int(opt.g) if torch.cuda.is_available() else "cpu")

logger.set_level(logger.INFO)
outdir = 'results'
model_path = Path(outdir)

env = gym.make('numberx-serengeti-v0', device=device)
env = wrappers.Monitor(env, directory=outdir, force=True)
env.seed(int(time.time()))
env.reset()

resize = T.Compose([
    T.ToPILImage(),
    T.ToTensor()
])


def get_screen(env, device):
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0).to(device)


init_screen = get_screen(env, device)
_, _, screen_height, screen_width = init_screen.shape
n_actions = [len(env.action_space)]


class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        h, w = state_shape[0], state_shape[1]
        h = w * 2
        self.h = h
        self.w = w

        self.dqn_lnd = DQN(1, h // 4, w // 2, 512, device=device)
        self.dqn_brd = DQN(1, h // 4, w // 2, 512, device=device)
        self.dqn_map = DQN(1, h // 2, w, 2048, device=device)
        self.recurr = Recurrent(1, 4096, action_shape, device=device)

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
            if cuda:
                obs = obs.cuda().to(device)

        lndl, lndr = obs[:, 0:1, 0:self.h // 4, 0:self.w // 2], obs[:, 0:1, 0:self.h // 4, self.w // 2:self.w]
        brdl, brdr = obs[:, 0:1, self.h // 4:self.h // 2, 0:self.w // 2], obs[:, 0:1, self.h // 4:self.h // 2, self.w // 2:self.w]
        map = obs[:, 0:1, self.h // 2:self.h, :]
        enc_lnd_l, _ = self.dqn_lnd(lndl, state=None)
        enc_lnd_r, _ = self.dqn_lnd(lndr, state=None)
        enc_brd_l, _ = self.dqn_brd(brdl, state=None)
        enc_brd_r, _ = self.dqn_brd(brdr, state=None)
        enc_map, _ = self.dqn_map(map, state=None)
        result, state = self.recurr(th.cat((enc_lnd_l, enc_lnd_r, enc_brd_l, enc_brd_r, enc_map), dim=1), state=state)
        return result, state


net = Net([screen_height, screen_width], n_actions)
if cuda:
    net = net.cuda().to(device)

optimizer = optim.Adam(net.parameters())
policy = ts.policy.DQNPolicy(net, optimizer, discount_factor=0.9, estimation_step=3, target_update_freq=320)

train_envs = ts.env.ShmemVectorEnv([lambda: gym.make('numberx-serengeti-v0') for _ in range(64)])
test_envs = ts.env.ShmemVectorEnv([lambda: gym.make('numberx-serengeti-v0') for _ in range(128)])

train_collector = ts.data.Collector(policy, train_envs, ts.data.ReplayBuffer(size=20000))
test_collector = ts.data.Collector(policy, test_envs)


def save(mean_rewards):
    torch.save(policy.state_dict(), model_path / f'perf_{mean_rewards}.chk')
    return mean_rewards


if __name__ == '__main__':
    logger.set_level(logger.INFO)

    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=1000, step_per_epoch=1000, collect_per_step=100,
        episode_per_test=100, batch_size=8,
        train_fn=lambda epoch, env_step: policy.set_eps(0.1),
        test_fn=lambda epoch, env_step: policy.set_eps(0.05),
        stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
        writer=None)
