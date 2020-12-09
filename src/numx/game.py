import random

from numx.affordable import Affordable
from tianshou.data import Batch


class AbstractGame(Affordable):
    def __init__(self, ctx):
        self.ctx = ctx
        self.affordables = []
        self.actions_list = []
        self.states_list = []
        self.policy = None
        self.ctx['game'] = self
        self.step_handlers = []

        for sub in self.all_affordables():
            self.add_affordable(sub)

    def all_affordables(self):
        return ()

    def add_affordable(self, affordable):
        for sub in affordable.subaffordables():
            self.affordables.append(sub)
        self.affordables.append(affordable)

        self.ctx['game'].add_step_handler(affordable)
        affordable.add_change_handler(self.ctx['game'])
        for sub in affordable.subaffordables():
            self.ctx['game'].add_step_handler(sub)
            sub.add_change_handler(self.ctx['outer'])

        import itertools, collections

        fields = [a.name() for a in self.affordables]
        self.actions_list = [collections.namedtuple('Action', fields)._make(actions)
                             for actions in itertools.product(*[a.available_actions() for a in self.affordables])]

        holders = self.affordables
        fields = [a.name() for a in holders]
        self.states_list = [collections.namedtuple('State', fields)._make(states)
                            for states in itertools.product(*[h.available_states() for h in holders])]

    def add_step_handler(self, handler):
        if handler is not self and handler not in self.step_handlers:
            self.step_handlers.append(handler)

    def fire_step_event(self, **pwargs):
        for h in self.step_handlers:
            h.on_stepped(self, **pwargs)

    def action_space(self):
        return self.actions_list

    def state_space(self):
        return self.states_list

    def action(self):
        import collections

        fields = [a.name() for a in self.affordables]
        a = collections.namedtuple('Action', fields)._make([a.action() for a in self.affordables])
        return a

    def state(self):
        import collections

        holders = self.affordables + [self.ctx['agent'].eye]
        fields = [a.name() for a in holders]
        s = collections.namedtuple('State', fields)._make([a.state() for a in holders])
        return Batch(s)

    def act(self, observation, reward, done):
        if self.policy is None:
            action = random.sample(self.action_space(), 1)
            for a in self.affordables:
                a.act(action)
            return action
        else:
            action = self.policy(observation, reward, done)
            for a in self.affordables:
                a.act(action)
            return action

    def reward(self):
        return 0.0

    def reset(self):
        for a in self.affordables:
            a.reset()

    def exit_condition(self):
        return False

    def force_condition(self):
        return random.random() < 0.005


