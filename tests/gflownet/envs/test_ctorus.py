import common
import numpy as np
import pytest
import torch

from gflownet.envs.ctorus import ContinuousTorus
from gflownet.utils.common import tbool, tfloat


@pytest.fixture
def env():
    return ContinuousTorus(n_dim=2, length_traj=3)


@pytest.mark.parametrize(
    "action_space",
    [
        [
            (0.0, 0.0),
            (np.inf, np.inf),
        ],
    ],
)
def test__get_action_space__returns_expected(env, action_space):
    assert set(action_space) == set(env.action_space)


@pytest.mark.parametrize(
    "state, done, is_backward, action_expected",
    [
        ([0.0, 0.0, 3.0], False, False, (np.inf, np.inf)),
        ([0.0, 0.0, 3.0], True, False, (np.inf, np.inf)),
        ([0.0, 0.0, 3.0], True, True, (np.inf, np.inf)),
        ([1.37, 2.49, 3.0], False, False, (np.inf, np.inf)),
        ([1.37, 2.49, 3.0], True, False, (np.inf, np.inf)),
        ([1.37, 2.49, 3.0], True, True, (np.inf, np.inf)),
        ([0.0, 0.0, 1.0], False, True, (0.0, 0.0)),
        ([1.37, 2.49, 1.0], False, True, (1.37, 2.49)),
    ],
)
def test__sample_actions_batch__special_cases(
    env, state, done, is_backward, action_expected
):
    """
    Test a few of all (known...) special cases, both forward and backward.
    """
    env.set_state(state, done=done)
    if is_backward:
        mask = torch.unsqueeze(
            tbool(env.get_mask_invalid_actions_backward(), device=env.device), 0
        )
    else:
        mask = torch.unsqueeze(
            tbool(env.get_mask_invalid_actions_forward(), device=env.device), 0
        )
    random_policy = torch.unsqueeze(env.random_policy_output, 0)
    action_sampled = env.sample_actions_batch(
        random_policy,
        mask,
        [state],
        is_backward,
    )[0][0]
    assert all(np.isclose(action_sampled, action_expected))


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "state, done, is_backward, action_special",
    [
        ([0.0, 0.0, 2.0], False, False, (np.inf, np.inf)),
        ([0.0, 0.0, 3.0], False, True, (np.inf, np.inf)),
        ([1.37, 2.49, 2.0], False, False, (np.inf, np.inf)),
        ([1.37, 2.49, 2.0], False, True, (np.inf, np.inf)),
        ([0.0, 0.0, 2.0], False, True, (0.0, 0.0)),
        ([1.37, 2.49, 2.0], False, True, (1.37, 2.49)),
        ([1.37, 2.49, 1.0], False, False, (1.37, 2.49)),
    ],
)
def test__sample_actions_batch__not_special_cases(
    env, state, done, is_backward, action_special
):
    """
    Test a few seemingly special cases, both forward and backward, and check that the
    special action is not sampled. Some of the tests may fail once in a blue moon if at
    all.
    """
    env.set_state(state, done=done)
    if is_backward:
        mask = torch.unsqueeze(
            tbool(env.get_mask_invalid_actions_backward(), device=env.device), 0
        )
    else:
        mask = torch.unsqueeze(
            tbool(env.get_mask_invalid_actions_forward(), device=env.device), 0
        )
    random_policy = torch.unsqueeze(env.random_policy_output, 0)
    action_sampled = env.sample_actions_batch(
        random_policy,
        mask,
        [state],
        is_backward,
    )[0][0]
    assert action_sampled != action_special


class TestContinuousTorusBasic(common.BaseTestsContinuous):
    @pytest.fixture(autouse=True)
    def setup(self, env):
        self.env = env
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__state2readable__is_reversible": 20,
            "test__gflownet_minimal_runs": 3,
        }
        self.n_states = {
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
            "test__gflownet_minimal_runs": 10,
        }
