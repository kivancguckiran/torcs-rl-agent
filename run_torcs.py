import argparse
import importlib

from env import torcs_envs as torcs


parser = argparse.ArgumentParser(description="TORCS")
parser.add_argument(
    "--seed", type=int, default=777, help="random seed for reproducibility")
parser.add_argument(
    "--algo", type=str, default="sac-lstm", help="choose an algorithm")
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
    "--relaunch-period", type=int, default=1, help="environment relaunch period")
parser.add_argument(
    "--test-period", type=int, default=100, help="test period")
parser.add_argument(
    "--num-stack", type=int, default=4, help="number of states to stack")
parser.add_argument(
    "--reward-type", type=str, default="extra_github", help="reward type")
parser.add_argument(
    "--track", type=str, default="none", help="track name")
parser.add_argument(
    "--use-state-filter", dest="state_filter", action="store_true", help="apply filter to observations")
parser.add_argument(
    "--use-action-filter", dest="action_filter", action="store_true", help="apply filter to actions")

parser.set_defaults(test=False)
parser.set_defaults(load_from=None)
parser.set_defaults(render=False)
parser.set_defaults(log=True)
parser.set_defaults(state_filter=False)
parser.set_defaults(action_filter=False)
args = parser.parse_args()


def main():
    state_filter = None if not args.state_filter else [1., 3., 10.]  # example filter (previous to recent)
    action_filter = None if not args.action_filter else [1., 3., 10.]

    if args.algo == "dqn":
        env = torcs.DiscretizedEnv(nstack=1,
                                   reward_type=args.reward_type,
                                   track=args.track,
                                   state_filter=state_filter,
                                   action_filter=None,
                                   action_count=21)
    elif args.algo == "sac":
        env = torcs.ContinuousEnv(nstack=4,
                                  reward_type=args.reward_type,
                                  track=args.track,
                                  state_filter=state_filter,
                                  action_filter=action_filter)
    elif args.algo == "sac-lstm":
        env = torcs.ContinuousEnv(nstack=1,
                                  reward_type=args.reward_type,
                                  track=args.track,
                                  state_filter=state_filter,
                                  action_filter=action_filter)
    else:
        raise Exception("Invalid algorithm!")

    module = importlib.import_module("torcs." + args.algo)
    agent = module.init(env, args)

    if args.test:
        agent.test()
    else:
        agent.train()


if __name__ == "__main__":
    main()
