from typing import Dict, List, Optional, Tuple, Union

import torch
from torchtyping import TensorType
from tqdm import tqdm

from gflownet.envs.stack import Stack
from gflownet.envs.cube import ContinuousCube
from gflownet.envs.set import SetFlex

from gflownet.utils.common import set_device

from icecream import ic, install
install()


class Photo(SetFlex):
    def __init__(self, device=torch.device("cpu"), **kwargs):
        # self.device = set_device(kwargs["device"])
        self.device = set_device(device)
        max_elements = 32
        envs_unique = [CubeStack(device=self.device, **kwargs)]
        # subenvs = [CubeStack(**kwargs)]
        super().__init__(max_elements=max_elements, envs_unique=envs_unique,
               do_random_subenvs=True, device=self.device, **kwargs)
               # subenvs=subenvs, do_random_subenvs=True, **kwargs)

class CubeStack(Stack):
    def __init__(self, **kwargs):
        self.dim_seq = (2,1,1)
        subenvs = (ContinuousCube(n_dim=n, **kwargs) for n in self.dim_seq)
        # fixed_distr_params = None, le probleme viens de ca. il faudrait
        # enlever ca des kwargs
        kwargs.pop("fixed_distr_params")
        kwargs.pop("random_distr_params")
        self.device = set_device(kwargs["device"])
        super().__init__(subenvs=tuple(subenvs), **kwargs)

def test__get_action_space(): 
    "constructs the list of possible actions (tuples)"
    env = Photo()
    act_sp = env.get_action_space()

def get_rand_action():
    import random
    a,b,c,d = [random.random() for i in range(4)]
    action = [(0, a, b, 0),
              (1, c, 0, 0),
              (2, d, 0, 0)]
    action = action[0] + action[1] + action[2]
    return action

def test__step():
    "given an action, update the environment's state"

    env = Photo()
    action = get_rand_action()
    state = env.step(action)

def test__get_parents():
    """obtains the list of parents of a given state, and their corresponding
    actions"""
    env = Photo()
    action = get_rand_action()
    state = env.step(action)
    states = env.get_parents()

def test__get_mask_invalid_actions_forward():
    "determines the invalid actions from a given state"
    env = Photo()
    env.get_mask_invalid_actions_forward(torch.rand(8,4))

# def test__state_converisons()
#     "(states2proxy(), states2policy(), states2readable(), readable2state()"
#     env.Photo()

def test__cube_stack_init():
    env = CubeStack(device=torch.device("cuda"))
    ic(env)


if __name__ == "__main__":

    test__cube_stack_init()
    # test__get_action_space()
    # test__step()
    # test__get_parents()
    # test__get_mask_invalid_actions_forward()


