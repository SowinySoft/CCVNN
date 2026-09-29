import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalFusionGate(nn.Module):
    """
    Gated Multi-Factor Fusion module combining 2D spatial embeddings
    with the 1D V_cum telemetry vector.
    """
    def __init__(self, spatial_dim: int = 256, telemetry_dim: int = 8, fused_dim: int = 128):
        super().__init__()
        self.spatial_proj = nn.Linear(spatial_dim, fused_dim)
        self.telemetry_proj = nn.Linear(telemetry_dim, fused_dim)
        
        # Gating network
        self.gate_net = nn.Sequential(
            nn.Linear(spatial_dim + telemetry_dim, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid()
        )
        
        # Output classification head for joint decision
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4) # Output: [Nominal, Minor, Critical, Hardware_Fault]
        )

    def forward(self, z_vis: torch.Tensor, v_cum: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        z_vis: [B, 256] 2D spatial feature embedding
        v_cum: [B, 8]   1D V_cum telemetry state vector
        """
        concat_features = torch.cat([z_vis, v_cum], dim=-1)
        
        # Calculate dynamic gate values
        gate = self.gate_net(concat_features)
        
        # Projected representations
        h_vis = self.spatial_proj(z_vis)
        h_tel = self.telemetry_proj(v_cum)
        
        # Gated fusion
        fused_embedding = gate * torch.tanh(h_vis) + (1.0 - gate) * torch.tanh(h_tel)
        
        logits = self.classifier(fused_embedding)
        return logits, gate

class DualModeVisionEngine(nn.Module):
    """
    Dual-Mode Vision Processing Model integrating 2D feature extraction
    and cross-modal telemetry alignment.
    """
    def __init__(self):
        super().__init__()
        # Backbone for 2D Feature Extraction
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # B, 32, H/2, W/2
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # B, 64, H/4, W/4
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.spatial_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU()
        )
        
        self.fusion = CrossModalFusionGate(spatial_dim=256, telemetry_dim=8, fused_dim=128)

    def forward(self, frame: torch.Tensor, v_cum: torch.Tensor):
        # Extract 2D Spatial Features
        feat_map = self.backbone(frame)
        z_vis = self.spatial_head(feat_map)
        
        # Fuse with 1D Telemetry Vector
        logits, gate_weights = self.fusion(z_vis, v_cum)
        return logits, gate_weights, z_vis