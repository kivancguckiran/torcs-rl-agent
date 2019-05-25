# -*- coding: utf-8 -*-
"""Abstract Agent used for all agents.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

from abc import ABC, abstractmethod
import argparse
import os
import subprocess
from typing import Tuple, Union

import gym
from gym.spaces import Discrete
from env.gym_torcs import TorcsEnv
import numpy as np
import torch

from env.torcs_envs import DefaultEnv


class Agent(ABC):
    """Abstract Agent used for all agents.

    Attributes:
        env (gym.Env): openAI Gym environment
        args (argparse.Namespace): arguments including hyperparameters and training settings
        env_name (str) : gym env name for logging
        sha (str): sha code of current git commit
        state_dim (int): dimension of states
        action_dim (int): dimension of actions
        is_discrete (bool): shows whether the action is discrete

    """

    def __init__(self, env: DefaultEnv, args: argparse.Namespace):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            args (argparse.Namespace): arguments including hyperparameters and training settings

        """
        self.args = args
        self.env = env

        if isinstance(env.action_space, Discrete):
            self.is_discrete = True
        else:
            self.is_discrete = False

        # for logging
        self.env_name = 'Torcs'
        self.sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])[:-1]
            .decode("ascii")
            .strip()
        )

        self.log_filename = self._init_log_file()

    @abstractmethod
    def select_action(self, state: np.ndarray) -> Union[torch.Tensor, np.ndarray]:
        pass

    @abstractmethod
    def step(
        self, action: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[np.ndarray, np.float64, bool]:
        pass

    @abstractmethod
    def update_model(self, *args) -> Tuple[torch.Tensor, ...]:
        pass

    @abstractmethod
    def load_params(self, *args):
        pass

    @abstractmethod
    def save_params(self, params: dict, n_episode: int):
        if not os.path.exists("./save"):
            os.mkdir("./save")

        save_name = self.env_name + "_" + self.args.algo + "_" + self.sha

        path = os.path.join("./save/" + save_name + "_ep_" + str(n_episode) + ".pt")
        torch.save(params, path)

        print("[INFO] Saved the model and optimizer to", path)

    @abstractmethod
    def write_log(self, *args):
        pass

    def _init_log_file(self):
        if not os.path.exists("./logs"):
            os.mkdir("./logs")

        logs_name = self.env_name + "_" + self.args.algo + "_" + self.sha

        return os.path.join("./logs/" + logs_name + ".txt")

    @abstractmethod
    def train(self):
        pass

    def interim_test(self):
        self.args.test = True

        print()
        print("===========")
        print("Start Test!")
        print("===========")

        self._test(interim_test=True)

        print("===========")
        print("Test done!")
        print("===========")
        print()

        self.args.test = False

    def test(self):
        """Test the agent."""

        self._test()

        # termination
        self.env.close()

    def _test(self, interim_test: bool = False):
        """Common test routine."""

        if interim_test:
            test_num = self.args.interim_test_num
        else:
            test_num = self.args.episode_num

        for i_episode in range(test_num):
            state = self.env.reset(relaunch=True, render=self.args.render, sampletrack=True)
            done = False
            score = 0
            step = 0
            speed = list()

            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward
                step += 1

                speed.append(self.env.last_speed)

            max_speed = 0 if speed is None else (max(speed))
            avg_speed = 0 if speed is None else (sum(speed) / len(speed))

            print(
                "[INFO] test %d\tstep: %d\ttotal score: %d\n" 
                "track name: %s\trace position: %d\tmax speed %.2f\tavg speed %.2f\n"
                % (
                    i_episode,
                    step,
                    score,
                    self.env.track_name,
                    self.env.last_obs['racePos'],
                    max_speed,
                    avg_speed
                )
            )


class AgentLSTM(ABC):
    """Abstract Agent used for all agents.

    Attributes:
        env (gym.Env): openAI Gym environment
        args (argparse.Namespace): arguments including hyperparameters and training settings
        env_name (str) : gym env name for logging
        sha (str): sha code of current git commit
        state_dim (int): dimension of states
        action_dim (int): dimension of actions
        is_discrete (bool): shows whether the action is discrete

    """

    def __init__(self, env: TorcsEnv, args: argparse.Namespace):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            args (argparse.Namespace): arguments including hyperparameters and training settings

        """
        self.args = args
        self.env = env

        if isinstance(env.action_space, Discrete):
            self.is_discrete = True
        else:
            self.is_discrete = False

        # for logging
        self.env_name = 'Torcs'
        self.sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])[:-1]
            .decode("ascii")
            .strip()
        )

        self.log_filename = self._init_log_file()

    @abstractmethod
    def select_action(self, state, hx, cx) -> Union[torch.Tensor, np.ndarray]:
        pass

    @abstractmethod
    def step(
        self, action: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[np.ndarray, np.float64, bool]:
        pass

    @abstractmethod
    def update_model(self, *args) -> Tuple[torch.Tensor, ...]:
        pass

    @abstractmethod
    def load_params(self, *args):
        pass

    @abstractmethod
    def save_params(self, params: dict, n_episode: int):
        if not os.path.exists("./save"):
            os.mkdir("./save")

        save_name = self.env_name + "_" + self.args.algo + "_" + self.sha

        path = os.path.join("./save/" + save_name + "_ep_" + str(n_episode) + ".pt")
        torch.save(params, path)

        print("[INFO] Saved the model and optimizer to", path)

    @abstractmethod
    def write_log(self, *args):
        pass

    def _init_log_file(self):
        if not os.path.exists("./logs"):
            os.mkdir("./logs")

        logs_name = self.env_name + "_" + self.args.algo + "_" + self.sha

        return os.path.join("./logs/" + logs_name + ".txt")

    @abstractmethod
    def train(self):
        pass

    def interim_test(self):
        self.args.test = True

        print()
        print("===========")
        print("Start Test!")
        print("===========")

        self._test(interim_test=True)

        print("===========")
        print("Test done!")
        print("===========")
        print()

        self.args.test = False

    def test(self):
        """Test the agent."""

        self._test()

        # termination
        self.env.close()

    def _test(self, interim_test: bool = False):
        """Common test routine."""

        if interim_test:
            test_num = self.args.interim_test_num
        else:
            test_num = self.args.episode_num

        for i_episode in range(test_num):
            state = self.env.reset(relaunch=True, render=self.args.render, sampletrack=True)
            hx, cx = self.actor.init_lstm_states(1)
            done = False
            score = 0
            step = 0
            speed = list()

            while not done:
                action, hx, cx = self.select_action(state, hx, cx)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward
                step += 1

                speed.append(self.env.last_speed)

            max_speed = 0 if speed is None else (max(speed))
            avg_speed = 0 if speed is None else (sum(speed) / len(speed))

            print(
                "[INFO] test %d\tstep: %d\ttotal score: %d\n" 
                "track name: %s\trace position: %d\tmax speed %.2f\tavg speed %.2f\n"
                % (
                    i_episode,
                    step,
                    score,
                    self.env.track_name,
                    self.env.last_obs['racePos'],
                    max_speed,
                    avg_speed
                )
            )

