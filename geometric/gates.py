import torch
import torch.nn as nn
from .config import GeometricConfig
from .jacobian import JacobianExtractor
from .hessian import HessianExtractor

class GeometricGateRouter(nn.Module):
    def __init__(self, config: GeometricConfig):
        super().__init__()
        self.config = config
        self.jacobian = JacobianExtractor(config)
        self.hessian = HessianExtractor(config)
        
        # Distribution gate weights for rotational chunk partitioning
        self.dist_gate = nn.Parameter(torch.ones(config.chunk_rotations))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Routes feature tensors through the rotational distribution gate, 
        extracts Jacobian gradients and Hessian curvature components.
        """
        # 1. Distribution Gate: Partition input channels across rotational chunks
        chunks = torch.chunk(x, chunks=self.config.chunk_rotations, dim=1)
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            gate_weight = torch.sigmoid(self.dist_gate[i])
            processed_chunks.append(chunk * gate_weight)

        # Recombine gated rotational chunks
        recombined = torch.cat(processed_chunks, dim=1)
        
        # 2. Extract First-Order Jacobian Gradients
        grad_y, grad_x = self.jacobian(x)
        
        # 3. Extract Second-Order Hessian Curvature
        hxx, hyy, hxy = self.hessian(grad_y, grad_x)
        
        return recombined, (hxx, hyy, hxy)
