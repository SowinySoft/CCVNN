from dataclasses import dataclass

@dataclass
class GeometricConfig:
    in_channels: int = 64
    spatial_dims: int = 2
    enable_mixed_precision: bool = True
    epsilon: float = 1e-5
    chunk_rotations: int = 4
