"""Small action-conditioned state world model, predicting dx, dy and dyaw."""
import numpy as np
import torch
from torch import nn

from .common import features, target


class Dynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(9, 128), nn.ReLU(), nn.Linear(128, 128),
                                 nn.ReLU(), nn.Linear(128, 3))

    def forward(self, x):
        return self.net(x)


def arrays(records):
    return np.stack([features(r) for r in records]), np.stack([target(r) for r in records])


def load_predictor(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = Dynamics()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    xmean, xstd = checkpoint["xmean"].numpy(), checkpoint["xstd"].numpy()
    ymean, ystd = checkpoint["ymean"].numpy(), checkpoint["ystd"].numpy()

    def predict(state, actions):
        actions = np.atleast_2d(actions)
        inputs = np.c_[np.tile(state, (len(actions), 1)), actions].astype(np.float32)
        with torch.no_grad():
            outputs = model(torch.from_numpy((inputs - xmean) / xstd)).numpy()
        return outputs * ystd + ymean

    return predict
