import os
import torch
import torch.nn as nn
import numpy as np

from pesla.network import TanhGaussianDistParams


STEER = 0
ACCEL = 1
BRAKE = 2


class PeslaAgent(nn.Module):
    def __init__(self, model_path: str, device: str = 'cuda'):
        super().__init__()
        self._device = device
        self.hx, self.cx = None, None
        self.model = TanhGaussianDistParams(
            input_size=29,
            output_size=2,
            hidden_sizes=[512, 256, 128],
            lstm_layer_size=1,
        ).to(device)
        self.load_model(model_path)

    def load_model(self, model_path: str):
        if model_path is not None and os.path.exists(model_path):
            params = torch.load(model_path, map_location=self._device)
            self.model.load_state_dict(params['actor'])
            self.reset_lstm()
        else:
            raise Exception('[ERROR] the input path does not exist. -> %s' % model_path)

    def reset_lstm(self):
        self.hx, self.cx = self.model.init_lstm_states(1, self._device)

    def action(self, state: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            _, _, _, u, _, self.hx, self.cx = self.model(state, 1, 1, self.hx, self.cx)
        u = u.squeeze_(0).squeeze_(0).detach().cpu().numpy()

        action = np.zeros(3)
        action[STEER] = u[0]
        if u[1] > 0:
            action[ACCEL] = u[1]
            action[BRAKE] = -1
        else:
            action[ACCEL] = 0
            action[BRAKE] = (abs(u[1]) * 2) - 1

        return action

    def fallback(self):
        action = np.zeros(3)
        action[STEER] = 0
        action[ACCEL] = 0.5
        action[BRAKE] = -1
        return action

    def forward(self, state) -> np.ndarray:
        state = torch.FloatTensor(state).to(self._device)
        return self.action(state)
