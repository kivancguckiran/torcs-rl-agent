import argparse

from env.gym_torcs import TorcsEnv
from pesla.agent import PeslaAgent


parser = argparse.ArgumentParser(description="TORCS")
parser.add_argument(
    "--device", type=str, dest="device", help="device to work with")
parser.add_argument(
    "--load-from", type=str, help="load the model from")
parser.add_argument(
    "--port", type=int, dest="port", help="environment connection port")
parser.add_argument(
    "--client-mode", dest="client_mode", action="store_true", help="turn on client mode")
parser.add_argument(
    "--track", type=str, default="none", help="track name")

parser.set_defaults(device="cpu")
parser.set_defaults(load_from=None)
parser.set_defaults(port=3101)
parser.set_defaults(client_mode=False)
args = parser.parse_args()


def main():
    env = TorcsEnv(port=args.port,
                   path="/usr/local/share/games/torcs/config/raceman/quickrace.xml",
                   client_mode=args.client_mode,
                   track=args.track)

    # The agent must be recreated or reset_lstm() function must be called for every episode start
    agent = PeslaAgent(model_path=args.load_from, device=args.device)
    agent.reset_lstm()

    state = env.reset(relaunch=not args.client_mode,
                      sampletrack=not args.client_mode,
                      render=not args.client_mode)
    while True:
        action = agent.forward(state)  # accepts and returns numpy array - ready for environment !
        state, _, done, _ = env.step(action)
        if done:
            break

    env.close()


if __name__ == "__main__":
    main()
