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


def list_to_tensor(states):
    """Convert SetFlex states to tensors for the proxy.
    
    Each state is a list of [setflex_state, cubestack_dict] where:
    - setflex_state: [-1, 0, [stack1_data], [stack2_data]]
    - cubestack_dict: {idx: [tensor(...), tensor(...), tensor(...)], ...}
    
    The output should be shaped as (batch_size, n_functions, 4) for RBF parameters.
    """
    batch_tensors = []
    
    for state in states:
        # Extract the CubeStack dictionary (second element)
        cubestack_dict = state[1]
        
        # Convert CubeStack dictionary to flat features
        cube_features = []
        for idx in sorted(cubestack_dict.keys()):
            cube_tensors = cubestack_dict[idx]
            # Flatten all tensors for this cube
            for tensor in cube_tensors:
                cube_features.extend(tensor.cpu().numpy().flatten())
        
        # Convert to tensor
        feature_tensor = torch.tensor(cube_features, dtype=torch.float32)
        
        # Reshape to RBF parameters: (n_functions, 4) where 4 = [weight, center_x, center_y, width]
        # We'll group features into sets of 4
        n_features = len(feature_tensor)
        n_functions = max(1, n_features // 4)  # At least 1 function
        
        # Pad or truncate to get exactly n_functions * 4 features
        target_size = n_functions * 4
        if n_features < target_size:
            feature_tensor = torch.cat([
                feature_tensor, 
                torch.zeros(target_size - n_features)
            ])
        else:
            feature_tensor = feature_tensor[:target_size]
        
        # Reshape to (n_functions, 4)
        rbf_params = feature_tensor.view(n_functions, 4)
        batch_tensors.append(rbf_params)
    
    # Pad all to same number of functions for consistent batch shape
    max_functions = max(t.shape[0] for t in batch_tensors)
    padded_tensors = []
    for tensor in batch_tensors:
        if tensor.shape[0] < max_functions:
            padding = torch.zeros(max_functions - tensor.shape[0], 4)
            tensor = torch.cat([tensor, padding], dim=0)
        padded_tensors.append(tensor)
    
    return torch.stack(padded_tensors)


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
    
    # For 3D input (batch_size, n_functions, 4), we don't need to unsqueeze
    if weights.ndim == 2:  # This means we already have (batch_size, n_functions)
        # centers should be (batch_size, n_functions, 2)
        # We don't need to unsqueeze centers for this case
        pass
    else:
        # For other cases, might need different handling
        centers = centers.unsqueeze(1)

    device = weights.device
    x, y = coords
    coords = torch.stack((x, y), dim=-1)

    z = broadcast_add(coords, -centers)
    sigma = torch.exp(widths)
    z = torch.exp(-(z[..., 0]**2 + z[..., 1]**2) / (2 * sigma**2))

    # Add dimensions for proper broadcasting
    weights_expanded = weights.unsqueeze(0).unsqueeze(0)  # (1, 1, batch_size, n_functions)
    
    z = weights_expanded * z
    z = z.sum(-1)  # Sum over n_functions
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
