import common
import numpy as np
import pytest
import torch

from gflownet.envs.photo import Photo
from gflownet.utils.photo.utils import rbf_parameters_to_designs

@pytest.mark.parametrize(
"state",
    [
        (
            set(torch.rand(4), torch.rand(4))
        ),
    ]
)
def test__states_convert_to_images_of_correct_size(state):
    image = rbf_parameters_to_designs(state)
    plt.imshow(image.numpy())
    plt.axis('off')
    plt.savefig('test_
