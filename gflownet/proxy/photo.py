from typing import List, Union

import numpy as np
from torchtyping import TensorType
import torch
import torch.nn as nn
from tqdm import tqdm

from gflownet.proxy.base import Proxy
from nanophoto.get_trained_models import get_cpx_fields_unet_cnn_fompred
from gflownet.utils.photo.utils import RBF, rbf_parameters_to_designs


class PhotoUnetProxy(Proxy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_shape = (101, 91)

    def setup(self, env=None):
        self.model = get_cpx_fields_unet_cnn_fompred()
        rows, cols = self.image_shape
        self.grid_points = make_grid(rows, cols)
        self.device = next(self.model.parameters()).device

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        states = list_to_tensor(states)
        states = rbf_function(states, self.grid_points)
        states = torch.sigmoid(states)
        fompred = self.model(states.to(self.device))
        fompred = fompred.squeeze()

        return fompred


def list_to_tensor(state):
    centers = []
    weights = []
    widths = []

    for x in state:
        centers.append(x[0])
        weights.append(x[1])
        widths.append(x[2])
    centers = torch.stack(centers, dim=0)
    weights = torch.stack(weights, dim=0)
    widths = torch.stack(widths, dim=0)
    return weights, centers, widths


def rbf_function(params: torch.Tensor | tuple, coords):
    """
    Params are the parameters of gaussians
    input:
        params: torch.Tensor (batch_size, n_functions, 4) or tuple with weights,
    centers and widths respectively of shape (batch_size, n_functiona, ) and
    (1,), (2,) or (1,) respectively
        coords: torch.Tensor (n_cols, n_rows)
    output: torch.Tensor (batch_size, n_cols, n_rows)
    """

    weights, centers, widths = params_split(params)
    if weights.ndim == 2:
        # weights = weights.unsqueeze(1)
        centers = centers.unsqueeze(1)
        # widths = widths.unsqueeze(1)

    device = weights.device
    x, y = coords
    coords = torch.stack((x, y), dim=-1)

    z = broadcast_add(coords, -centers)
    sigma = torch.exp(widths)
    z = torch.exp(-(z[..., 0]**2 + z[..., 1]**2) / (2 * sigma**2))

    z = weights * z
    z = z.sum(-1)
    z = z.permute(2, 0, 1)
    return z


def broadcast_add(a, b):
    assert a.shape[-1] == b.shape[-1]
    sha = a.shape[:-1]
    shb = b.shape[:-1]
    shape = sha + shb + (a.shape[-1],)
    na = len(sha)
    nb = len(shb)
    for i in range(na):
        b = b.unsqueeze(0)
    for i in range(nb):
        a = a.unsqueeze(-2)
    a = a.expand(shape)
    b = b.expand(shape)
    return a + b


def params_split(params: torch.Tensor | tuple) -> tuple:
    """
    output: weights, centers, widths

    """
    if isinstance(params, torch.Tensor):
        weights = params[..., 0].clone().detach()
        centers = params[..., 1:3].clone().detach()
        widths = params[..., 3].clone().detach()
    elif isinstance(params, np.ndarray):
        weights = params[..., 0]
        centers = params[..., 1:3]
        widths = params[..., 3]
    elif isinstance(params, tuple):
        weights, centers, widths = params
    else:
        raise TypeError(type(params))
    return weights, centers, widths


def make_grid(rows, cols, device=torch.device("cpu")):
    r = torch.arange(rows, device=device)
    c = torch.arange(cols, device=device)
    r_grid, c_grid = torch.meshgrid(r, c, indexing='ij')
    return c_grid, r_grid


if __name__ == "__main__":
    m = PhotoUnetProxy(device=torch.device("cuda"), float_precision=16)
    print(m)
