import numpy as np
import torch as th

from numx.game import AbstractGame
from numx.agent import NNAgent
from numx.counting_test import CountTest


class NumberXGame(AbstractGame):
    def __init__(self, device=None):
        super(NumberXGame, self).__init__({})
        self.softmax = th.nn.Softmax()
        self.kld = th.nn.KLDivLoss()

        self.device = device

    def reset(self):
        super(NumberXGame, self).reset()

    def all_affordables(self):
        self.agent = NNAgent(self.ctx, 256, 256)
        self.count_test = CountTest(self.ctx, 256, 256, 3, 10)
        return self.agent, self.count_test

    def state_space(self):
        blkb = self.agent.blackboard.reshape(self.agent.height, self.agent.width, 1)
        blkb = np.concatenate((blkb, blkb, blkb), axis=2)
        ss = np.concatenate((blkb, self.count_test.data), axis=1)
        tns = th.tensor(ss, device=self.device).permute(2, 0, 1)
        return tns

    def state(self):
        s = np.concatenate((self.agent.blackboard, self.count_test.data), axis=2)
        return th.tensor(s, device=self.device)

    def reward(self):
        true_answer = th.tensor(self.count_test.answer().reshape(1, 3), device=self.device)
        agent_answer = self.softmax(self.agent.answer())
        return - self.kld(agent_answer, true_answer)

    def exit_condition(self):
        result = self.count_test.tests >= self.count_test.max_round
        if result:
            self.count_test.new_round()
        return result

    def force_condition(self):
        return False
