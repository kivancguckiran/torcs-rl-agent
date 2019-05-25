import argparse

import numpy as np
import torch
import torch.optim as optim

from algorithms.common.networks.mlp import MLP, FlattenMLP, TanhGaussianDistParams
from algorithms.sac.agent import SACAgent

from env.torcs_envs import DefaultEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
hyper_params = {
    "GAMMA": 0.99,
    "TAU": 5e-3,
    "W_ENTROPY": 1e-3,
    "W_MEAN_REG": 0.0,
    "W_STD_REG": 0.0,
    "W_PRE_ACTIVATION_REG": 0.0,
    "LR_ACTOR": 3e-4,
    "LR_VF": 3e-4,
    "LR_QF1": 3e-4,
    "LR_QF2": 3e-4,
    "LR_ENTROPY": 3e-4,
    "POLICY_UPDATE_FREQ": 2,
    "BUFFER_SIZE": int(1e6),
    "BATCH_SIZE": 128,
    "AUTO_ENTROPY_TUNING": True,
    "WEIGHT_DECAY": 0.0,
    "INITIAL_RANDOM_ACTION": int(1e4),
    "PREFILL_BUFFER": int(1e4),
    "MULTIPLE_LEARN": 1,
    "BRAKE_ENABLE": True,
    "BRAKE_REGION": int(2e5),
    "BRAKE_DIST_MU": int(1e5),
    "BRAKE_DIST_SIGMA": int(3e4),
    "BRAKE_FACTOR": 1e-1
}


def init(env: DefaultEnv, args: argparse.Namespace):

    hidden_sizes_actor = [512, 256, 128]
    hidden_sizes_vf = [512, 256, 128]
    hidden_sizes_qf = [512, 256, 128]

    if args.load_from is not None:
        hyper_params["INITIAL_RANDOM_ACTION"] = 0

    # target entropy
    target_entropy = -np.prod((env.action_dim,)).item()  # heuristic

    # create actor
    actor = TanhGaussianDistParams(
        input_size=env.state_dim,
        output_size=env.action_dim,
        hidden_sizes=hidden_sizes_actor
    ).to(device)

    # create v_critic
    vf = MLP(
        input_size=env.state_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_vf).to(
        device
    )
    vf_target = MLP(
        input_size=env.state_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_vf
    ).to(device)
    vf_target.load_state_dict(vf.state_dict())

    # create q_critic
    qf_1 = FlattenMLP(
        input_size=env.state_dim + env.action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_qf
    ).to(device)
    qf_2 = FlattenMLP(
        input_size=env.state_dim + env.action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_qf
    ).to(device)

    # create optimizers
    actor_optim = optim.Adam(
        actor.parameters(),
        lr=hyper_params["LR_ACTOR"],
        weight_decay=hyper_params["WEIGHT_DECAY"],
    )
    vf_optim = optim.Adam(
        vf.parameters(),
        lr=hyper_params["LR_VF"],
        weight_decay=hyper_params["WEIGHT_DECAY"],
    )
    qf_1_optim = optim.Adam(
        qf_1.parameters(),
        lr=hyper_params["LR_QF1"],
        weight_decay=hyper_params["WEIGHT_DECAY"],
    )
    qf_2_optim = optim.Adam(
        qf_2.parameters(),
        lr=hyper_params["LR_QF2"],
        weight_decay=hyper_params["WEIGHT_DECAY"],
    )

    models = (actor, vf, vf_target, qf_1, qf_2)
    optims = (actor_optim, vf_optim, qf_1_optim, qf_2_optim)

    agent = SACAgent(env, args, hyper_params, models, optims, target_entropy)

    return agent
