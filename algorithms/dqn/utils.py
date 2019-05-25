# -*- coding: utf-8 -*-
"""Utility functions for DQN.

This module has DQN util functions.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

from typing import Tuple, Union

import torch
import torch.nn.functional as F

from algorithms.common.networks.mlp import MLP
from algorithms.dqn.networks import C51DuelingMLP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calculate_c51_loss(
    model: Union[C51DuelingMLP],
    target_model: Union[C51DuelingMLP],
    experiences: Tuple[torch.Tensor, ...],
    gamma: float,
    batch_size: int,
    v_min: int,
    v_max: int,
    atom_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return element-wise C51 loss and Q-values."""
    states, actions, rewards, next_states, dones = experiences[:5]
    support = torch.linspace(v_min, v_max, atom_size).to(device)
    delta_z = float(v_max - v_min) / (atom_size - 1)

    with torch.no_grad():
        # According to noisynet paper,
        # it resamples noisynet parameters on online network when using double q
        # but we don't because there is no remarkable difference in performance.
        next_actions = model.forward_(next_states)[1].argmax(1)

        next_dist = target_model.forward_(next_states)[0]
        next_dist = next_dist[range(batch_size), next_actions]

        t_z = rewards + (1 - dones) * gamma * support
        t_z = t_z.clamp(min=v_min, max=v_max)
        b = (t_z - v_min) / delta_z
        l = b.floor().long()  # noqa: E741
        u = b.ceil().long()

        offset = (
            torch.linspace(0, (batch_size - 1) * atom_size, batch_size)
            .long()
            .unsqueeze(1)
            .expand(batch_size, atom_size)
            .to(device)
        )

        proj_dist = torch.zeros(next_dist.size(), device=device)
        proj_dist.view(-1).index_add_(
            0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
        )
        proj_dist.view(-1).index_add_(
            0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
        )

    dist, q_values = model.forward_(states)
    log_p = torch.log(dist[range(batch_size), actions.long()])

    dq_loss_element_wise = -(proj_dist * log_p).sum(1)

    return dq_loss_element_wise, q_values


def calculate_dqn_loss(
    model: Union[MLP],
    target_model: Union[MLP],
    experiences: Tuple[torch.Tensor, ...],
    gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return element-wise dqn loss and Q-values."""
    states, actions, rewards, next_states, dones = experiences[:5]

    q_values = model(states)
    # According to noisynet paper,
    # it resamples noisynet parameters on online network when using double q
    # but we don't because there is no remarkable difference in performance.
    next_q_values = model(next_states)

    next_target_q_values = target_model(next_states)

    curr_q_value = q_values.gather(1, actions.long().unsqueeze(1))
    next_q_value = next_target_q_values.gather(  # Double DQN
        1, next_q_values.argmax(1).unsqueeze(1)
    )

    # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
    #       = r                       otherwise
    masks = 1 - dones
    target = rewards + gamma * next_q_value * masks
    target = target.to(device)

    # calculate dq loss
    dq_loss_element_wise = F.smooth_l1_loss(
        curr_q_value, target.detach(), reduction="none"
    )

    return dq_loss_element_wise, q_values
