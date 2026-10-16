import torch
import torch.nn as nn
from .config import GeometricConfig

class HessianExtractor(nn.Module):
    def __init__(self, config: GeometricConfig):
        super().__init__()
        self.config = config

    def forward(self, grad_y: torch.Tensor, grad_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts second-order differential curvature (Hessian components) from gradients.
        Inputs: grad_y, grad_x from JacobianExtractor
        Output: Tuple of second-order partial derivative tensors (hxx, hyy, hxy)
        """
        hxx = grad_x[:, :, :, 1:] - grad_x[:, :, :, :-1]
        hyy = grad_y[:, :, 1:, :] - grad_y[:, :, :-1, :]
        hxy = grad_y[:, :, :, 1:] - grad_y[:, :, :, :-1]
        return hxx, hyy, hxy
