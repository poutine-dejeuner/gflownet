import torch
import torch.nn as nn


class RBF(nn.Module):
    def __init__(self, weights, centers, log_widths, bias):
        """
        weights: (num_rbf)
        centers: (num_rbf, 2)
        log_widths: (num_rbf)
        bias: (num_rbf)

        """
        super().__init__()
        self.weights = nn.Parameter(weights)
        self.centers = nn.Parameter(centers)
        self.log_widths = nn.Parameter(log_widths)
        self.image_shape = (101, 91)
        # self.bias = nn.Parameter(bias)
        rows, cols = self.image_shape
        r = torch.arange(rows, dtype=torch.float32)
        c = torch.arange(cols, dtype=torch.float32)
        r_grid, c_grid = torch.meshgrid(r, c, indexing='ij')
        coords = (c_grid, r_grid)
        self.grid = coords

    def forward(self, grid):
        # grid should have shape (num_fun, height, width, 2)
        # where the first dimension is the batch (matching num_fun)
        # and the last dimension is (x, y) coordinates

        # N = self.weights.shape[0]  # num_fun (also batch size here)
        # M = self.weights.shape[1]  # num_rbf
        # H = grid.shape[1]        # height
        # W = grid.shape[2]        # width

        # Expand grid and centers for broadcasting
        grid = grid.unsqueeze(1)      # (N, 1, H, W, 2)
        centers = self.centers.unsqueeze(2).unsqueeze(3)  # (N, M, 1, 1, 2)
        weights = self.weights.unsqueeze(2).unsqueeze(3)  # (N, M, 1, 1)
        # bias = self.bias.unsqueeze(2)                   # (N, 1, 1)
        widths = torch.exp(self.log_widths)              # (N, M)
        denominator = 2 * \
            (widths ** 2).unsqueeze(2).unsqueeze(3)  # (N, M, 1, 1)

        # Calculate squared Euclidean distances
        distances_sq = torch.sum((grid - centers) ** 2, dim=4)  # (N, M, H, W)

        # Calculate RBF values
        rbf_values = torch.exp(-distances_sq / denominator)  # (N, M, H, W)

        # Weight the RBF values and sum over the RBFs
        weighted_rbf = weights * rbf_values               # (N, M, H, W)
        output = torch.sum(weighted_rbf, dim=1)  # + bias   # (N, H, W)

        return output


def rbf_parameters_to_designs(states_tensor):
    weights = states_tensor[:, :, 0]
    centers = states_tensor[:, :, 1:3]
    log_widths = states_tensor[:, :, 3]

    rbf = RBF(weights, centers, log_widths)
    designs = rbf()
    designs = designs > 1/2
    return designs
