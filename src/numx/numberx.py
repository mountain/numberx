from numx.game import AbstractGame
from numx.blackboard import Blackboard


class NumberXGame(AbstractGame):
    def __init__(self):
        super(NumberXGame, self).__init__({})

    def reset(self):
        super(NumberXGame, self).reset()

    def all_affordables(self):
        self.blackboard = Blackboard(self.ctx, 200, 100)
        return self.blackboard,

    def available_actions(self):
        return ('idle',)

    def available_states(self):
        return ('idle',)