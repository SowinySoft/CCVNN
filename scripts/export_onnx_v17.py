import torch
import torch.nn as nn

class CCVNN_V17_Backbone(nn.Module):
    def __init__(self, input_dim=17, hidden_dim1=64, hidden_dim2=64, hidden_dim3=32, output_dim=1):
        super(CCVNN_V17_Backbone, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.head = nn.Linear(hidden_dim3, output_dim)
        
        # HardSwish Activation function
        self.act = nn.Hardswish()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return torch.sigmoid(self.head(x))

if __name__ == "__main__":
    model = CCVNN_V17_Backbone(input_dim=17)
    model.eval()
    
    dummy_input = torch.randn(1, 17, dtype=torch.float32)
    onnx_file = "ccvnn_v17_backbone.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['v_input_17'],
        output_names=['hazard_prob'],
        dynamic_axes={'v_input_17': {0: 'batch_size'}, 'hazard_prob': {0: 'batch_size'}}
    )
    
    print(f"Successfully exported CCVNN V17 Backbone ONNX model to: {onnx_file}")