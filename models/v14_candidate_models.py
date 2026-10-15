import torch
import torch.nn as nn

class AConC(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.p1 = nn.Parameter(torch.ones(1, channels))
        self.p2 = nn.Parameter(torch.zeros(1, channels))
        self.beta = nn.Parameter(torch.ones(1, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.p1 - self.p2
        return d * x * torch.sigmoid(self.beta * d * x) + self.p2 * x

class ModelA_HardSwish(nn.Module):
    def __init__(self, in_features=14, hidden_dim=64, num_classes=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Hardswish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Hardswish(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class ModelB_ACon(nn.Module):
    def __init__(self, in_features=14, hidden_dim=64, num_classes=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.act1 = AConC(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = AConC(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)

class ModelC_Hybrid(nn.Module):
    def __init__(self, in_features=14, hidden_dim=64, num_classes=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.act1 = AConC(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.Hardswish()
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)

def export_candidates_to_onnx():
    dummy_input = torch.randn(1, 14, dtype=torch.float32)
    models = {
        "model_a_hardswish.onnx": ModelA_HardSwish(),
        "model_b_acon.onnx": ModelB_ACon(),
        "model_c_hybrid.onnx": ModelC_Hybrid()
    }
    for filename, model in models.items():
        model.eval()
        torch.onnx.export(
            model,
            dummy_input,
            f"build/{filename}",
            input_names=["input_vector"],
            output_names=["output_prediction"],
            dynamic_axes={"input_vector": {0: "batch_size"}, "output_prediction": {0: "batch_size"}},
            opset_version=14
        )
        print(f"[Export] Generated build/{filename}")

if __name__ == "__main__":
    export_candidates_to_onnx()
