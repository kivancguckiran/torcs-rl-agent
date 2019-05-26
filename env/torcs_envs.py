import numpy as np
import torch

from collections import deque

from env.gym_torcs import TorcsEnv
from gym import spaces


STEER = 0
ACCEL = 1
BRAKE = 2


class DefaultEnv(TorcsEnv):
    def __init__(self,
                 port=3101,
                 nstack=1,
                 reward_type='extra_github',
                 track='none',
                 state_filter=None,
                 action_filter=None,
                 client_mode=False):
        super().__init__(port,
                         path='/usr/local/share/games/torcs/config/raceman/quickrace.xml',
                         reward_type=reward_type,
                         client_mode=client_mode,
                         track=track)
        self.nstack = nstack
        self.stack_buffer = deque(maxlen=nstack)

        self.state_filter = state_filter
        if state_filter is not None:
            self.state_filter = np.tile(np.array(state_filter).reshape(-1, 1), self.observation_space.shape[0])
            self.state_filter_size = self.state_filter.shape[0]
            self.state_filter_buffer = deque(maxlen=self.state_filter_size)

        self.action_filter = action_filter
        if action_filter is not None:
            self.action_filter = np.tile(np.array(action_filter).reshape(-1, 1), self.action_space.shape[0])
            self.action_filter_size = self.action_filter.shape[0]
            self.action_filter_buffer = deque(maxlen=self.action_filter_size)

    @property
    def state_dim(self):
        return self.observation_space.shape[0] * self.nstack

    @property
    def action_dim(self):
        if isinstance(self.action_space, spaces.Discrete):
            return self.action_space.n
        return self.action_space.shape[0]

    def preprocess_action(self, action):
        if self.action_filter is not None:
            self.action_filter_buffer.append(action)
            while len(self.action_filter_buffer) < self.action_filter_size:
                self.action_filter_buffer.append(action)
            actions = np.array(self.action_filter_buffer)
            action = np.sum(np.multiply(actions, self.action_filter), axis=0) / sum(self.action_filter)

        return action

    def reset(self, relaunch=False, sampletrack=False, render=False):
        state = super().reset(relaunch, sampletrack, render)
        if self.nstack > 1:
            [self.stack_buffer.append(state) for i in range(self.nstack)]
            state = np.asarray(self.stack_buffer).flatten()
        return state

    def step(self, u):
        u = self.preprocess_action(u)

        next_state, reward, done, info = super().step(u)

        if self.state_filter is not None:
            self.state_filter_buffer.append(next_state)
            while len(self.state_filter_buffer) < self.state_filter_size:
                self.state_filter_buffer.append(next_state)
            states = np.array(self.state_filter_buffer)
            next_state = np.sum(np.multiply(states, self.state_filter), axis=0) / sum(self.state_filter)

        if self.nstack > 1:
            self.stack_buffer.append(next_state)
            next_state = np.asarray(self.stack_buffer).flatten()

        return next_state, reward, done, info

    def try_brake(self, action):
        return action


class ContinuousEnv(DefaultEnv):
    def __init__(self,
                 port=3101,
                 nstack=1,
                 reward_type='extra_github',
                 track='none',
                 state_filter=None,
                 action_filter=None,
                 client_mode=False):
        super().__init__(port, nstack, reward_type, track, state_filter, action_filter, client_mode)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

    def preprocess_action(self, u):
        act = np.zeros(3)

        act[STEER] = u[0]

        if u[1] > 0:
            act[ACCEL] = u[1]
            act[BRAKE] = -1
        else:
            act[ACCEL] = 0
            act[BRAKE] = (abs(u[1]) * 2) - 1

        return super().preprocess_action(act)

    def try_brake(self, u):
        u[1] = torch.rand(1) - 1
        return u


class DiscretizedEnv(DefaultEnv):
    def __init__(self,
                 port=3101,
                 nstack=1,
                 reward_type='extra_github',
                 track='none',
                 state_filter=None,
                 action_filter=None,
                 client_mode=False,
                 action_count=21):
        super().__init__(port, nstack, reward_type, track, state_filter, action_filter, client_mode)

        assert (action_count + 3) % 6 == 0

        self.action_space = spaces.Discrete(action_count)

        self.accelerate_actions = np.tile([1, 0, 0], action_count // 3)
        self.brake_actions = np.tile([-1, -1, 1], action_count // 3)
        self.steer_actions = np.repeat(np.linspace(-1, 1, action_count // 3), 3).flatten()

    def preprocess_action(self, u):
        act = np.zeros(3)

        act[ACCEL] = self.accelerate_actions[u]
        act[STEER] = self.steer_actions[u]
        act[BRAKE] = self.brake_actions[u]

        return super().preprocess_action(act)

    def try_brake(self, u):
        brake_actions = np.linspace(2, self.action_dim - 1, self.action_dim // 3)
        return int(np.random.choice(brake_actions))
