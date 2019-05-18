from gym_torcs import TorcsEnv
from gym import spaces
import numpy as np


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
        env_u[1] = (env_u[1] + 1) / 2
        return super().step(np.concatenate((env_u, [-1])))

class HalfBrakeNoBackwardsEnv(DefaultEnv):
    def step(self, u):
        env_u = u.copy()
        env_u[1] = (env_u[1] + 1) / 2
        env_u[2] = (env_u[2] - 1) / 2
        return super().step(env_u)

class NoBackwardsEnv(DefaultEnv):
    def step(self, u):
        env_u = u.copy()
        env_u[1] = (env_u[1] + 1) / 2
        return super().step(env_u)
