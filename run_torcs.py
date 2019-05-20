# -*- coding: utf-8 -*-
"""Train or test algorithms on LunarLanderContinuous-v2.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse
import importlib

import torcs_envs as torcs


# configurations
parser = argparse.ArgumentParser(description="Pytorch RL algorithms")
parser.add_argument(
    "--seed", type=int, default=777, help="random seed for reproducibility")
parser.add_argument(
    "--algo", type=str, default="sac", help="choose an algorithm")
parser.add_argument(
    "--test", dest="test", action="store_true", help="test mode (no training)")
parser.add_argument(
    "--load-from", type=str, help="load the saved model and optimizer at the beginning")
parser.add_argument(
    "--on-render", dest="render", action="store_true", help="turn on rendering")
parser.add_argument(
    "--log", dest="log", action="store_true", help="turn on logging")
parser.add_argument(
    "--save-period", type=int, default=50, help="save model period")
parser.add_argument(
    "--episode-num", type=int, default=10000, help="total episode num")
parser.add_argument(
    "--max-episode-steps", type=int, default=10000, help="max episode step")
parser.add_argument(
    "--interim-test-num", type=int, default=1, help="interim test number")
parser.add_argument(
    "--relaunch-period", type=int, default=5, help="environment relaunch period")
parser.add_argument(
    "--test-period", type=int, default=100, help="test period")
parser.add_argument(
    "--num-stack", type=int, default=1, help="number of states to stack")
parser.add_argument(
    "--reward-type", type=str, default="no_trackpos", help="reward type")

parser.set_defaults(test=False)
parser.set_defaults(load_from=None)
parser.set_defaults(render=False)
parser.set_defaults(log=True)
args = parser.parse_args()


def main():
    if args.algo.startswith("dqn") or args.algo == "sac-discrete":
        env = torcs.DiscretizedEnv(nstack=args.num_stack, reward_type=args.reward_type, action_count=21)
        # env = torcs.DiscretizedOldEnv(nstack=args.num_stack, reward_type=args.reward_type)
    else:
        env = torcs.BitsPiecesContEnv(nstack=args.num_stack, reward_type=args.reward_type)

    # set a random seed
    # common_utils.set_random_seed(args.seed, env)

    # run
    module_path = "examples.torcs." + args.algo
    example = importlib.import_module(module_path)
    example.run(env, args, env.state_dim, env.action_dim)


if __name__ == "__main__":
    main()
