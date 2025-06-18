import numpy as np
from photo import *
from icecream import ic

def test__get_action_space():
    env = Photo()
    actsp = env.get_action_space()
    ic(actsp)


def test__step():
    env = Photo()
    action = tuple(np.random.rand(env.action_dim + 1))
    state, _, _ = env.step(action)
    ic(state)

# test__get_action_space()
test__step()
