from gym_torcs import TorcsEnv
from gym import spaces
import numpy as np


STEER = 0
ACCELERATE = 1
BRAKE = 2


class DefaultEnv(TorcsEnv):
    def __init__(self, port=3101):
        super().__init__(port, '/usr/local/share/games/torcs/config/raceman/quickrace.xml')

    @property
    def state_dim(self):
        return self.observation_space.shape[0]

    @property
    def action_dim(self):
        return self.action_space.shape[0]

class NoBrakeNoBackwardsEnv(DefaultEnv):
    def __init__(self, port=3101):
        super().__init__(port)
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
    def __init__(self, port=3101):
        super().__init__(port)
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

class DiscretizedEnv(DefaultEnv):
    def __init__(self, port=3101):
        super().__init__(port)
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
        elif u == 7:
            # steer = -1, throttle = 0, brake = 1
            env_u[STEER] = -1
            env_u[ACCELERATE] = 0
            env_u[BRAKE] = 1

        return super().step(env_u)
