import argparse
import importlib

from env import torcs_envs as torcs


parser = argparse.ArgumentParser(description="TORCS")
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
    filter_kernel = None if not args.filter else [5., 2., 1.]  # example filter (recent to previous)

    if args.algo == "dqn":
        env = torcs.DiscretizedEnv(nstack=1,
                                   filter=filter_kernel,
                                   action_count=21,
                                   client_mode=True,
                                   port=args.port)
    elif args.algo == "sac":
        env = torcs.ContinuousEnv(nstack=4,
                                  filter=filter_kernel,
                                  client_mode=True,
                                  port=args.port)
    elif args.algo == "sac-lstm":
        env = torcs.ContinuousEnv(nstack=1,
                                  filter=filter,
                                  client_mode=True,
                                  port=args.port)
    else:
        raise Exception("Invalid algorithm!")

    module = importlib.import_module("torcs." + args.algo)
    agent = module.init(env, args)

    state = env.reset()
    for i in range(args.max_episode_steps):
        u = agent.select_action(state)
        action = env.preprocess_action(u)
        state, _, done, _ = env.step(action)
        if done:
            break

    env.close()


if __name__ == "__main__":
    main()
