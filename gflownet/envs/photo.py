from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import copy, tlong

from icecream import ic

"""
un nouvel env GFlowNetEnv 
1. __init__(): defines attributes, EOS action, source state

2. get_action_space(): constructs the list of possible actions (tuples)

3. step(action): given an action, update the environment's state

4. get_parents(state): obtains the list of parents of a given state, and their
corresponding actions

5. get_mask_invalid_actions_forward(state): determines the invalid actions from
a given state

6. state converisons (states2proxy(), states2policy(), states2readable(),
readable2state()

"""

class Photo(GFlowNetEnv):
    def __init__(
        self,
        max_length: int = 50,
        n_dim: int = 6,
        **kwargs,
    ):
        # Main attributes
        self.n_dim = n_dim
        self.pad_token = torch.zeros(n_dim)
        self.max_length = max_length
        self.eos_idx = -1
        self.pad_idx = 0
        # Dictionaries
        self.source = torch.zeros(n_dim)
        # End-of-sequence action
        self.eos = (self.eos_idx,)
        # Base class init
        super().__init__(**kwargs)

    def get_action_space(self):
        """
        The action space is continuous, thus not defined as such here.

        The actions contained in the action space are "representatives"

        The actions are tuples of length n_dim + 1, where the value at position d
        indicates the increment of dimension d, and the value at position -1 indicates
        whether the action is from or to source (1), or 0 otherwise.

        EOS is indicated by np.inf for all dimensions.

        The action space consists of the EOS actions and two representatives:
        - Generic increment action, not from or to source: (0, 0, ..., 0, 0)
        - Generic increment action, from or to source: (0, 0, ..., 0, 1)
        - EOS: (inf, inf, ..., inf, inf)
        """
        actions_dim = self.n_dim + 1
        self.eos = tuple([np.inf] * actions_dim)
        self.representative_no_source = tuple([0.0] * self.n_dim) + (0,)
        self.representative_source = tuple([0.0] * self.n_dim) + (1,)
        return [self.representative_no_source, self.representative_source, self.eos]

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List[int]] = None,
        done: Optional[bool] = None,
    ) -> List[bool]:
        """
        Returns a list of length the action space with values:
            - True if the forward action is invalid from the current state.
            - False otherwise.

        Args
        ----
        state : tensor
            Input state. If None, self.state is used.

        done : bool
            Whether the trajectory is done. If None, self.done is used.

        Returns
        -------
        A list of boolean values.
        """
        return [True]*len(state)
        

    def get_parents(
        self,
        state: Optional[List[int]] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        """
        Determines all parents and actions that lead to state.

        The GFlowNet graph is a tree and there is only one parent per state.

        Args
        ----
        state : tensor
            Input state. If None, self.state is used.

        done : bool
            Whether the trajectory is done. If None, self.done is used.

        action : None
            Ignored

        returns
        -------
        parents : list
            list of parents in state format. this environment has a single parent per
            state.

        actions : list
            list of actions that lead to state for each parent in parents. this
            environment has a single parent per state.
        """
        if state == self.source:
            return [], []
        else:
            if len(state) == 1:
                return [self.source()], [state]

            parents = []
            actions = []
            for single_action in state:
                subset = state.copy()
                subset.remove(single_action)
                parents.append(subset)
                actions.append(single_action)
            return parents, actions

    def step(
        self, action: Tuple[int], skip_mask_check: bool = False
    ) -> [List[int], Tuple[int], bool]:
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action to be executed. An action is a tuple int values indicating the
            dimensions to increment by 1.

        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if the
            action is valid.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : tuple
            Action executed

        valid : bool
            False, if the action is not allowed for the current state.
        """
        # Generic pre-step checks
        do_step, self.state, action = self._pre_step(
            action, skip_mask_check or self.skip_mask_check
        )
        if not do_step:
            return self.state, action, False
        valid = True
        self.n_actions += 1
        # If action is EOS, set done to True and return state as is
        if action == self.eos:
            self.done = True
            return self.state, action, valid
        # Update state
        self.state.add(action[0])
        return self.state, action, valid

    def _get_max_trajectory_length(self) -> int:
        """
        Returns the maximum trajectory length of the environment.

        The maximum trajectory lenght is the maximum sequence length (self.max_length)
        plus one (EOS action).
        """
        return self.max_length + 1

    def states2tensor(
        self, states: Union[List[List[int]], List[TensorType["max_length"]]]
    ) -> TensorType["batch", "state_dim"]:
        states = list(states)
        ic(len(states), states[0].shape)
        if len(states) == 1:
            return states[0]
        else:
            states = torch.concatenate(states, dim=0)
            return states

    def states2proxy(
        self, states: Union[List[List[int]], List[TensorType["max_length"]]]
    ) -> TensorType["batch", "state_dim"]:
        """
        Prepares a batch of states in "environment format" for a proxy: the batch is
        simply converted into a tensor of indices.

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            list of tensors.

        Returns
        -------
        A list containing all the states in the batch, represented themselves as lists.
        """
        states = self.states2tensor(states)
        return tlong(states, device=self.device)

    def states2policy(
        self, states: Union[List[List[int]], List[TensorType["max_length"]]]
    ) -> TensorType["batch", "policy_input_dim"]:
        """
        Prepares a batch of states in "environment format" for the policy model: states
        are one-hot encoded.

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            list of tensors.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        states = self.states2tensor(states)
        states = tlong(states, device=self.device)
        return states

    def get_uniform_terminating_states(
        self, n_states: int, seed: int = None
    ) -> List[List[int]]:
        """
        Constructs a batch of n states uniformly sampled in the sample space of the
        environment.

        Args
        ----
        n_states : int
            The number of states to sample.

        seed : int
            Random seed.
        """
        n_letters = len(self.letters)
        n_per_length = tlong(
            [n_letters**length for length in range(1, self.max_length + 1)],
            device=self.device,
        )
        lengths = Categorical(
            logits=n_per_length.repeat(n_states, 1)).sample() + 1
        samples = torch.randint(
            low=1, high=n_letters + 1, size=(n_states, self.max_length)
        )
        for idx, length in enumerate(lengths):
            samples[idx, length:] = 0
        return samples.tolist()

    def _get_seq_length(self, state: List[int] = None):
            return len(state)

    def print_states(self, state):
        import matplotlib.pyplot as plt
        from gflownet.utils.photo.utils import rbf_parameters_to_designs



