import torch as th
import numpy as np
import random

from tianshou.data import Batch


NOOP = 'noop'
IDLE = 'idle'


def get_action(ctx, **pwargs):
    if 'action' in pwargs:
        action = pwargs['action']

    game = ctx['game']
    if type(action) == th.Tensor:
        action = action.item()
        action = game.action_space()[action]
    if type(action) == list:
        action = action[0]

    return action


class Affordable:
    def __init__(self, ctx, name):
        self._name = name
        self.ctx = ctx
        self.changed_handlers = []

        self._state = self.available_states()[0]
        self._action =  self.available_actions()[0]

    def name(self):
        return self._name

    def subaffordables(self):
        return ()

    def available_actions(self):
        return (NOOP, )

    def available_states(self):
        return (IDLE, )

    def action(self):
        return self._action

    def state(self):
        return self._state

    def reset(self):
        for sub in self.subaffordables():
            sub.reset()

    def on_stepped(self, src, **pwargs):
        pass

    def add_change_handler(self, handler):
        if handler not in self.changed_handlers:
            self.changed_handlers.append(handler)

    def fire_changed_event(self, **pwargs):
        for h in self.changed_handlers:
            h.on_changed(self, **pwargs)


class Blackboard(Affordable):
    def __init__(self, ctx, width, height):
        super(Blackboard, self).__init__(ctx, 'blackboard')
        self.width = width
        self.height = height
        self.data = np.zeros([height, width])

        self.pen_status = 'up'
        self.x = 0
        self.y = 0

    def available_actions(self):
        return (NOOP, 'pu', 'pd', 'up', 'dn', 'lf', 'rt')

    def available_states(self):
        return (IDLE, )

    def reset(self):
        super(Blackboard, self).reset()
        self.data = np.zeros([self.height, self.width])

    def act(self, action):
        action = get_action(self.ctx, action=action).blackboard
        if action == 'pu':
            self.pu()
        elif action == 'pd':
            self.pd()
        elif action == 'up':
            self.up()
        elif action == 'dn':
            self.dn()
        elif action == 'lf':
            self.lf()
        elif action == 'rt':
            self.rt()

    def pu(self):
        self.pen_status = 'up'

    def pd(self):
        self.pen_status = 'dn'

    def up(self):
        self.y -= 1
        if self.y < 0:
            self.y = 0

        if self.pen_status == 'dn':
            self.data[self.y, self.x] = 1.0

    def dn(self):
        self.y += 1
        if self.y > self.height - 1:
            self.y = self.height - 1

        if self.pen_status == 'dn':
            self.data[self.y, self.x] = 1.0

    def lf(self):
        self.x -= 1
        if self.x < 0:
            self.x = 0

        if self.pen_status == 'dn':
            self.data[self.y, self.x] = 1.0

    def rt(self):
        self.x += 1
        if self.x > self.width - 1:
            self.x = self.width - 1

        if self.pen_status == 'dn':
            self.data[self.y, self.x] = 1.0


class Agent:
    pass



class Questionary:
    pass


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
