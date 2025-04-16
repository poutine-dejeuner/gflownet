import pytest
import torch

from gflownet.envs.photo import Photo 
from gflownet.proxy.photo import PhotoUnetProxy


@pytest.fixture()
def proxy():
    return PhotoUnetProxy(device="cpu", float_precision=32)


@pytest.fixture
def env():
    return Photo(max_length=7, device="cpu")


@pytest.mark.parametrize(
    "samples, scores_expected",
    [
        (
            [
                ["C", "A", "T", "0", "0", "0", "0"],
                ["D", "O", "G", "0", "0", "0", "0"],
                ["B", "I", "R", "D", "0", "0", "0"],
                ["F", "R", "I", "E", "N", "D", "S"],
            ],
            [3 + 1 + 1, 2 + 1 + 2, 3 + 1 + 1 + 2, 4 + 1 + 1 + 1 + 1 + 2 + 1],
        ),
    ],
)
def test__scrabble_scorer__returns_expected_scores_list_input_list_tokens(
    env, proxy, samples, scores_expected
):
    proxy.setup(env)
    scores = proxy(samples)
    assert scores.tolist() == scores_expected
