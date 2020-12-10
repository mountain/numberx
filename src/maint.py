# -*- coding: utf-8 -*-

import gym_numberx

import os
import gym
import numpy as np
import argparse
import tianshou as ts

import torch
import torch.optim as optim
import torchvision.transforms as T

from pathlib import Path
from gym import wrappers, logger
from tianshou.utils.net.discrete import DQN

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
env.seed(0)
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
n_actions = len(env.action_space)

net = DQN(3, screen_height, screen_width, n_actions, device=device).to(device)
optimizer = optim.Adam(net.parameters())
policy = ts.policy.DQNPolicy(net, optimizer, discount_factor=0.9, estimation_step=3, target_update_freq=320)

train_envs = ts.env.DummyVectorEnv([lambda: gym.make('numberx-serengeti-v0') for _ in range(8)])
test_envs = ts.env.DummyVectorEnv([lambda: gym.make('numberx-serengeti-v0') for _ in range(32)])

train_collector = ts.data.Collector(policy, train_envs, ts.data.ReplayBuffer(size=20000))
test_collector = ts.data.Collector(policy, test_envs)


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

    print(f'training | duration: {result["duration"]}, best: {result["best_reward"]}')

    perf = result["best_reward"]
    dura = result["duration"]

    filepath = model_path / f'perf_{int(perf):010d}.duration_{int(dura):04d}.chk'
    torch.save(policy.state_dict(), filepath)
