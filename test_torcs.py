import argparse
import importlib

from env import torcs_envs as torcs


parser = argparse.ArgumentParser(description="TORCS")
parser.add_argument(
    "--seed", type=int, default=777, help="random seed for reproducibility")
parser.add_argument(
    "--algo", type=str, default="sac-lstm", help="choose an algorithm")
parser.add_argument(
    "--load-from", type=str, help="load the saved model and optimizer at the beginning")
parser.add_argument(
    "--max-episode-steps", type=int, default=10000, help="max episode step")
parser.add_argument(
    "--test", dest="test", action="store_true", help="test mode (no training)")
parser.add_argument(
    "--track", type=str, default="none", help="track name")
parser.add_argument(
    "--use-state-filter", dest="state_filter", action="store_true", help="apply filter to observations")
parser.add_argument(
    "--use-action-filter", dest="action_filter", action="store_true", help="apply filter to actions")
parser.add_argument(
    "--port", dest="port", type=int, help="port")

parser.set_defaults(test=True)
parser.set_defaults(load_from=None)
parser.set_defaults(state_filter=False)
parser.set_defaults(action_filter=False)
args = parser.parse_args()


def main():
    state_filter = None if not args.state_filter else [1., 3., 10.]  # example filter (previous to recent)
    action_filter = None if not args.action_filter else [1., 3., 10.]

    if args.algo == "dqn":
        env = torcs.DiscretizedEnv(nstack=1,
                                   state_filter=state_filter,
                                   action_filter=None,
                                   action_count=21,
                                   client_mode=True,
                                   port=args.port)
    elif args.algo == "sac":
        env = torcs.ContinuousEnv(nstack=4,
                                  state_filter=state_filter,
                                  action_filter=action_filter,
                                  client_mode=True,
                                  port=args.port)
    elif args.algo == "sac-lstm":
        env = torcs.ContinuousEnv(nstack=1,
                                  state_filter=state_filter,
                                  action_filter=action_filter,
                                  client_mode=True,
                                  port=args.port)
    else:
        raise Exception("Invalid algorithm!")

    module = importlib.import_module("torcs." + args.algo)
    agent = module.init(env, args)

    state = env.reset()
    if args.algo == "sac-lstm":
        hx, cx = agent.actor.init_lstm_states(1)
    while True:
        if args.algo == "sac":
            action = agent.select_action(state)
        elif args.algo == "sac-lstm":
            action, hx, cx = agent.select_action(state, hx, cx)
        state, _, done, _ = env.step(action)
        if done:
            break

    env.close()


if __name__ == "__main__":
    main()
