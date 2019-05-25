# -*- coding: utf-8 -*-
"""MLP module for dqn algorithms

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
"""

import math
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.common.networks.mlp import MLP, init_layer_uniform, init_layer_xavier
from algorithms.dqn.linear import NoisyMLPHandler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DuelingMLP(MLP, NoisyMLPHandler):
    """Multilayer perceptron with dueling construction."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list,
        hidden_activation: Callable = F.relu,
        linear_layer: nn.Module = nn.Linear,
        init_fn: Callable = init_layer_xavier,
    ):
        """Initialization."""
        super(DuelingMLP, self).__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
            linear_layer=linear_layer,
            use_output_layer=False,
        )
        in_size = hidden_sizes[-1]

        # set advantage layer
        self.advantage_hidden_layer = self.linear_layer(in_size, in_size)
        self.advantage_layer = self.linear_layer(in_size, output_size)
        self.advantage_layer = init_fn(self.advantage_layer)

        # set value layer
        self.value_hidden_layer = self.linear_layer(in_size, in_size)
        self.value_layer = self.linear_layer(in_size, 1)
        self.value_layer = init_fn(self.value_layer)

    def _forward_dueling(self, x: torch.Tensor) -> torch.Tensor:
        adv_x = self.hidden_activation(self.advantage_hidden_layer(x))
        val_x = self.hidden_activation(self.value_hidden_layer(x))

        advantage = self.advantage_layer(adv_x)
        value = self.value_layer(val_x)
        advantage_mean = advantage.mean(dim=-1, keepdim=True)

        q = value + advantage - advantage_mean

        return q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = super(DuelingMLP, self).forward(x)
        x = self._forward_dueling(x)

        return x


class C51DuelingMLP(MLP, NoisyMLPHandler):
    """Multilayered perceptron for C51 with dueling construction."""

    def __init__(
        self,
        input_size: int,
        action_size: int,
        hidden_sizes: list,
        atom_size: int = 51,
        v_min: int = -10,
        v_max: int = 10,
        hidden_activation: Callable = F.relu,
        linear_layer: nn.Module = nn.Linear,
        init_fn: Callable = init_layer_xavier,
    ):
        """Initialization."""
        super(C51DuelingMLP, self).__init__(
            input_size=input_size,
            output_size=action_size,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
            linear_layer=linear_layer,
            use_output_layer=False,
        )
        in_size = hidden_sizes[-1]
        self.action_size = action_size
        self.atom_size = atom_size
        self.output_size = action_size * atom_size
        self.v_min, self.v_max = v_min, v_max

        # set advantage layer
        self.advantage_hidden_layer = self.linear_layer(in_size, in_size)
        self.advantage_layer = self.linear_layer(in_size, self.output_size)
        self.advantage_layer = init_fn(self.advantage_layer)

        # set value layer
        self.value_hidden_layer = self.linear_layer(in_size, in_size)
        self.value_layer = self.linear_layer(in_size, self.atom_size)
        self.value_layer = init_fn(self.value_layer)

    def forward_(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get distribution for atoms."""
        action_size, atom_size = self.action_size, self.atom_size

        x = super(C51DuelingMLP, self).forward(x)
        adv_x = self.hidden_activation(self.advantage_hidden_layer(x))
        val_x = self.hidden_activation(self.value_hidden_layer(x))

        advantage = self.advantage_layer(adv_x).view(-1, action_size, atom_size)
        value = self.value_layer(val_x).view(-1, 1, atom_size)
        advantage_mean = advantage.mean(dim=1, keepdim=True)

        q_atoms = value + advantage - advantage_mean
        dist = F.softmax(q_atoms, dim=2)

        support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(device)
        q = torch.sum(dist * support, dim=2)

        return dist, q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        _, q = self.forward_(x)

        return q
