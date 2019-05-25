# -*- coding: utf-8 -*-
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
    "--load-from", type=str, help="load the saved model and optimizer at the beginning")
parser.add_argument(
    "--max-episode-steps", type=int, default=10000, help="max episode step")
parser.add_argument(
    "--test", dest="test", action="store_true", help="test mode (no training)")
parser.add_argument(
    "--use-filter", dest="filter", action="store_true", help="apply filter to observations")
parser.add_argument(
    "--port", dest="port", type=int, help="port")

parser.set_defaults(test=True)
parser.set_defaults(load_from=None)
parser.set_defaults(filter=False)
args = parser.parse_args()


def main():
    filter = None if not args.filter else [5., 2., 1.]  # example filter (recent to previous)

    if args.algo == "dqn9":
        env = torcs.DiscretizedOldEnv(nstack=1,
                                      filter=filter,
                                      client_mode=True,
                                      port=args.port)
    elif args.algo == "dqn21":
        env = torcs.DiscretizedEnv(nstack=1,
                                   filter=filter,
                                   action_count=21,
                                   client_mode=True,
                                   port=args.port)
    elif args.algo == "sac":
        env = torcs.BitsPiecesContEnv(nstack=4,
                                      filter=filter,
                                      client_mode=True,
                                      port=args.port)
    elif args.algo == "sac-lstm":
        env = torcs.BitsPiecesContEnv(nstack=1,
                                      filter=filter,
                                      client_mode=True,
                                      port=args.port)

    module = importlib.import_module("torcs." + args.algo)

    agent = module.init(env, args, env.state_dim, env.action_dim)

    state = env.reset()
    for i in range(args.max_episode_steps):
        action = module.action(agent, state)
        state, _, done, _ = env.step(action)
        if done:
            break
    env.close()


if __name__ == "__main__":
    main()
