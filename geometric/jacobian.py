import torch
import torch.nn as nn
from .config import GeometricConfig

class JacobianExtractor(nn.Module):
    def __init__(self, config: GeometricConfig):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts first-order spatial gradients (Jacobian approximation) along height and width.
        Input: Tensor of shape (Batch, Channels, Height, Width)
        Output: Tuple of gradient tensors (grad_y, grad_x)
        """
        grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
        grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
        return grad_y, grad_x
