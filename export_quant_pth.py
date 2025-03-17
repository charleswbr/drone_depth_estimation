import torch
import torch.quantization

# Set the quantization engine (qnnpack is typically used on Apple Silicon/ARM CPUs)
torch.backends.quantized.engine = 'qnnpack'

# ---------------------------
# 1. Load the Checkpoint
# ---------------------------
model_path = "checkpoints/depth_anything_v2_metric_hypersim_vits.pth"
device = torch.device("cpu")  # Use CPU for quantization and scripting
checkpoint = torch.load(model_path, map_location=device)

# Extract the state dictionary.
if isinstance(checkpoint, dict):
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

# ---------------------------
# 2. Instantiate the Model
# ---------------------------
# Adjust the import to match your repository’s structure.
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

# Use the configuration for the "small absolute metric" variant.
model_cfg = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}

# Pass the config as keyword arguments (unpacking the dictionary).
model = DepthAnythingV2(**model_cfg)

# Load the state dict and set to evaluation mode.
model.load_state_dict(state_dict)
model.eval()
model = model.to("cpu")

# ---------------------------
# 3. Apply Dynamic Quantization
# ---------------------------
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# ---------------------------
# 4. Wrap the Model for Scripting
# ---------------------------
# TorchScript cannot handle variable arguments with keyword defaults,
# so we wrap the model in a module with a fixed forward signature.
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # Call the underlying model with a fixed parameter: is_training=False
        return self.model.forward(x, is_training=False)

wrapper_model = ModelWrapper(quantized_model)

# ---------------------------
# 5. Convert to TorchScript and Save
# ---------------------------
scripted_model = torch.jit.script(wrapper_model)
output_script_path = "depth_anything_v2_quantized.pt"
torch.jit.save(scripted_model, output_script_path)
print("Quantized TorchScript model saved to:", output_script_path)
