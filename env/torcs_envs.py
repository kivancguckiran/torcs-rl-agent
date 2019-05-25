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
                 reward_type='sigmoid',
                 track='none',
                 filter=None,
                 client_mode=False):
        super().__init__(port,
                         path='/usr/local/share/games/torcs/config/raceman/quickrace.xml',
                         reward_type=reward_type,
                         client_mode=client_mode,
                         track=track)
        self.nstack = nstack
        self.stack_buffer = deque(maxlen=nstack)
        self.filter = filter
        if filter is not None:
            self.filter = np.tile(np.array(filter).reshape(-1, 1), self.observation_space.shape[0])
            self.filter_size = self.filter.shape[0]
            self.filter_buffer = deque(maxlen=self.filter_size)

    @property
    def state_dim(self):
        return self.observation_space.shape[0] * self.nstack

    @property
    def action_dim(self):
        if isinstance(self.action_space, spaces.Discrete):
            return self.action_space.n
        return self.action_space.shape[0]

    def preprocess_action(self, u):
        return u

    def reset(self, relaunch=False, sampletrack=False, render=False):
        state = super().reset(relaunch, sampletrack, render)
        if self.nstack > 1:
            [self.stack_buffer.append(state) for i in range(self.nstack)]
            state = np.asarray(self.stack_buffer).flatten()
        return state

    def step(self, u):
        u = self.preprocess_action(u)

        next_state, reward, done, info = super().step(u)

        if self.filter is not None:
            self.filter_buffer.append(next_state)
            while len(self.filter_buffer) < self.filter_size:
                self.filter_buffer.append(next_state)
            prev_state = np.array(self.filter_buffer)
            next_state = np.sum(np.multiply(prev_state, self.filter), axis=0) / sum(self.filter)

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
                 reward_type='sigmoid',
                 track='none',
                 filter=None,
                 client_mode=False):
        super().__init__(port, nstack, reward_type, track, filter, client_mode)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

    def preprocess_action(self, u):
        env_u = np.zeros(3)

        env_u[STEER] = u[0]

        if u[1] > 0:
            env_u[ACCEL] = u[1]
            env_u[BRAKE] = -1
        else:
            env_u[ACCEL] = 0
            env_u[BRAKE] = (abs(u[1]) * 2) - 1

        return env_u

    def try_brake(self, u):
        u[1] = torch.rand(1) - 1
        return u


class DiscretizedEnv(DefaultEnv):
    def __init__(self,
                 port=3101,
                 nstack=1,
                 reward_type='sigmoid',
                 track='none',
                 filter=None,
                 action_count=21,
                 client_mode=False):
        super().__init__(port, nstack, reward_type, track, filter, client_mode)

        assert (action_count + 3) % 6 == 0

        self.action_space = spaces.Discrete(action_count)

        self.accelerate_actions = np.tile([1, 0, 0], action_count // 3)
        self.brake_actions = np.tile([-1, -1, 1], action_count // 3)
        self.steer_actions = np.repeat(np.linspace(-1, 1, action_count // 3), 3).flatten()

    def preprocess_action(self, u):
        env_u = np.zeros(3)

        env_u[ACCEL] = self.accelerate_actions[u]
        env_u[STEER] = self.steer_actions[u]
        env_u[BRAKE] = self.brake_actions[u]

        return env_u

    def try_brake(self, u):
        brake_actions = np.linspace(2, self.action_dim - 1, self.action_dim // 3)
        return int(np.random.choice(brake_actions))
