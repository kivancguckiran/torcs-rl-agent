from gym_torcs import TorcsEnv
import numpy as np


class TempEnv(TorcsEnv):
    def __init__(self, port=3101):
        super().__init__(port, '/usr/local/share/games/torcs/config/raceman/quickrace.xml')
        self.state_dim = 29
        self.action_dim = 2

    def step(self, u):
        env_u = u.copy()
        env_u[1] = (env_u[1] + 1) / 2
        return super().step(np.concatenate((env_u, [-1])))
