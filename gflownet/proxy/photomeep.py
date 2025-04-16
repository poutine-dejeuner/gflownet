from typing import List, Union

from torchtyping import TensorType
import torch
import torch.nn as nn
from tqdm import tqdm

from gflownet.proxy.base import Proxy
from nanophoto.unet_fompred import get_unet_fompred

class PhotoMeepProxy(Proxy):
    from nanophoto.meep_compute_fom import compute_FOM_parallel
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self, env=None):
        if env:
            self.height = env.height
            self.width = env.width

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:

        designs = rbf_parameters_to_design(states)
        fom = compute_FOM_parallel(designs)
        return fom
