from numx.game import AbstractGame
from numx.agent import RandomAgent
from numx.counting_test import CountTest


class NumberXGame(AbstractGame):
    def __init__(self):
        super(NumberXGame, self).__init__({})

    def reset(self):
        super(NumberXGame, self).reset()

    def all_affordables(self):
        self.agent = RandomAgent(self.ctx, 200, 200)
        self.count_test = CountTest(self.ctx, 200, 200, 3, 10)
        return self.agent, self.count_test

    def available_actions(self):
        return ('idle',)

    def available_states(self):
        return ('idle',)

    def reward(self):
        true_answer = self.count_test.answer()
        agent_answer = self.agent.answer()
        if agent_answer == true_answer:
            return 1.0
        else:
            return 0.0

    def exit_condition(self):
        result = self.count_test.tests >= self.count_test.max_round
        if result:
            self.count_test.new_round()
        return result

    def force_condition(self):
        return False
