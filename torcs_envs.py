from gym_torcs import TorcsEnv
from gym import spaces
import numpy as np
from collections import deque
import torch


STEER = 0
ACCELERATE = 1
BRAKE = 2


class DefaultEnv(TorcsEnv):
    def __init__(self, port=3101, nstack=1, reward_type='no_trackpos', track='none'):
        super().__init__(port, path='/usr/local/share/games/torcs/config/raceman/quickrace.xml', reward_type=reward_type, track=track)
        self.nstack = nstack
        self.stack_buffer = deque(maxlen=nstack)

    @property
    def state_dim(self):
        return self.observation_space.shape[0] * self.nstack

    @property
    def action_dim(self):
        if isinstance(self.action_space, spaces.Discrete):
            return self.action_space.n
        return self.action_space.shape[0]

    def reset(self, relaunch=False, sampletrack=False, render=False):
        state = super().reset(relaunch, sampletrack, render)
        if self.nstack > 1:
            [self.stack_buffer.append(state) for i in range(self.nstack)]
            state = np.asarray(self.stack_buffer).flatten()
        return state

    def step(self, u):
        next_state, reward, done, info = super().step(u)
        if self.nstack > 1:
            self.stack_buffer.append(next_state)
            next_state = np.asarray(self.stack_buffer).flatten()
        return next_state, reward, done, info

    def try_brake(self, action):
        return action


class NoBrakeNoBackwardsEnv(DefaultEnv):
    def __init__(self, port=3101, nstack=1, reward_type='no_trackpos', track='none'):
        super().__init__(port, nstack, reward_type, track)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

    def step(self, u):
        env_u = u.copy()
        env_u[ACCELERATE] = (env_u[ACCELERATE] + 1) / 2
        return super().step(np.concatenate((env_u, [-1])))


class HalfBrakeNoBackwardsEnv(DefaultEnv):
    def step(self, u):
        env_u = u.copy()
        env_u[ACCELERATE] = (env_u[ACCELERATE] + 1) / 2
        env_u[BRAKE] = (env_u[BRAKE] - 1) / 2
        return super().step(env_u)


class NoBackwardsEnv(DefaultEnv):
    def step(self, u):
        env_u = u.copy()
        env_u[ACCELERATE] = (env_u[ACCELERATE] + 1) / 2
        return super().step(env_u)


class BitsPiecesEnv(DefaultEnv):
    def __init__(self, port=3101, nstack=1, reward_type='no_trackpos', track='none'):
        super().__init__(port, nstack, reward_type, track)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

    def step(self, u):
        env_u = np.zeros(3)

        env_u[STEER] = u[0]

        if u[1] > 0:
            env_u[ACCELERATE] = 1
            env_u[BRAKE] = -1
        else:
            env_u[ACCELERATE] = 0
            env_u[BRAKE] = 1

        return super().step(env_u)


class BitsPiecesContEnv(DefaultEnv):
    def __init__(self, port=3101, nstack=1, reward_type='no_trackpos', track='none'):
        super().__init__(port, nstack, reward_type, track)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

    def step(self, u):
        env_u = np.zeros(3)

        env_u[STEER] = u[0]

        if u[1] > 0:
            env_u[ACCELERATE] = u[1]
            env_u[BRAKE] = -1
        else:
            env_u[ACCELERATE] = 0
            env_u[BRAKE] = (abs(u[1]) * 2) - 1

        return super().step(env_u)

    def try_brake(self, u):
        u[1] = torch.rand(1) - 1
        return u


class DiscretizedEnv(DefaultEnv):
    def __init__(self, port=3101, nstack=1, reward_type='no_trackpos', track='none', action_count=21):
        super().__init__(port, nstack, reward_type, track)

        assert (action_count + 3) % 6 == 0

        self.action_space = spaces.Discrete(action_count)

        self.accelerate_actions = np.tile([1, 0, 0], action_count // 3)
        self.brake_actions = np.tile([-1, -1, 1], action_count // 3)
        self.steer_actions = np.repeat(np.linspace(-1, 1, action_count // 3), 3).flatten()

    def step(self, u):
        env_u = np.zeros(3)

        env_u[ACCELERATE] = self.accelerate_actions[u]
        env_u[STEER] = self.steer_actions[u]
        env_u[BRAKE] = self.brake_actions[u]

        return super().step(env_u)


class DiscretizedInriaEnv(DefaultEnv):
    def __init__(self, port=3101, nstack=1, reward_type='no_trackpos', track='none',
                 steer_count=9, accel_count=3, steer_brake_count=5):
        super().__init__(port, nstack, reward_type, track)

        assert steer_count % 2 == 1
        assert accel_count > 1
        assert steer_brake_count % 2 == 1

        self.n_steer_accel = steer_count * accel_count
        self.action_space = spaces.Discrete(steer_count * accel_count + steer_brake_count)

        self.accel_actions = np.zeros(self.action_space.n)
        self.steer_actions = np.zeros(self.action_space.n)
        self.brake_actions = np.ones(self.action_space.n) * -1.0

        self.accel_actions[:self.n_steer_accel] = np.tile(np.linspace(0, 1, accel_count), steer_count)
        self.steer_actions[:self.n_steer_accel] = np.repeat(np.linspace(-1, 1, steer_count), accel_count)
        self.steer_actions[self.n_steer_accel:] = np.linspace(-1, 1, steer_brake_count)
        self.brake_actions[self.n_steer_accel:] = np.ones(steer_brake_count)

    def step(self, u):
        env_u = np.zeros(3)

        env_u[ACCELERATE] = self.accel_actions[u]
        env_u[STEER] = self.steer_actions[u]
        env_u[BRAKE] = self.brake_actions[u]

        return super().step(env_u)

    def try_brake(self, u):
        return np.random.randint(self.n_steer_accel, self.action_space.n)


class DiscretizedOldEnv(DefaultEnv):
    def __init__(self, port=3101, nstack=1, reward_type='no_trackpos', track='none'):
        super().__init__(port, nstack, reward_type, track)
        self.action_space = spaces.Discrete(9)

    def step(self, u):
        env_u = np.zeros(3)

        if u == 0:
            # steer = 1, throttle = 1, brake = -1
            env_u[STEER] = 1
            env_u[ACCELERATE] = 1
            env_u[BRAKE] = -1
        elif u == 1:
            # steer = 0, throttle = 1, brake = -1
            env_u[STEER] = 0
            env_u[ACCELERATE] = 1
            env_u[BRAKE] = -1
        elif u == 2:
            # steer = -1, throttle = 1, brake = -1
            env_u[STEER] = -1
            env_u[ACCELERATE] = 1
            env_u[BRAKE] = -1
        elif u == 3:
            # steer = 1, throttle = 0, brake = -1
            env_u[STEER] = 1
            env_u[ACCELERATE] = 0
            env_u[BRAKE] = -1
        elif u == 4:
            # steer = 0, throttle = 0, brake = -1
            env_u[STEER] = 0
            env_u[ACCELERATE] = 0
            env_u[BRAKE] = -1
        elif u == 5:
            # steer = -1, throttle = 0, brake = -1
            env_u[STEER] = -1
            env_u[ACCELERATE] = 0
            env_u[BRAKE] = -1
        elif u == 6:
            # steer = 1, throttle = 0, brake = 1
            env_u[STEER] = 1
            env_u[ACCELERATE] = 0
            env_u[BRAKE] = 1
        elif u == 7:
            # steer = 0, throttle = 0, brake = 1
            env_u[STEER] = 0
            env_u[ACCELERATE] = 0
            env_u[BRAKE] = 1
        elif u == 8:
            # steer = -1, throttle = 0, brake = 1
            env_u[STEER] = -1
            env_u[ACCELERATE] = 0
            env_u[BRAKE] = 1

        return super().step(env_u)
