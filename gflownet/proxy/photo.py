from typing import List, Union

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

    def setup(self, env=None):
        self.model = get_cpx_fields_unet_cnn_fompred()
        # if env:
        #     self.height = env.height
        #     self.width = env.width

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        designs = rbf_parameters_to_designs(states)
        fompred = self.model(states)

        return fompred


if __name__ == "__main__":
    m = PhotoUnetProxy(device=torch.device("cuda"), float_precision=16)
    print(m)
    
