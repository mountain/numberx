import torch as th


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

