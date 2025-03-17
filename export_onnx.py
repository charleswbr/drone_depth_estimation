import torch
import numpy as np
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

# ---------------------------
# 1. Load the Checkpoint
# ---------------------------
model_path = "checkpoints/depth_anything_v2_metric_hypersim_vits.pth"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
checkpoint = torch.load(model_path, map_location=device)

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
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2  # Adjust import as needed

model_cfg = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
model = DepthAnythingV2(**model_cfg)
model.load_state_dict(state_dict)
model.eval()

# Move model to CPU for export.
model_cpu = model.to("cpu")

# ---------------------------
# 3. Export to ONNX
# ---------------------------
dummy_input = torch.randn(1, 3, 224, 224)
onnx_model_path = "depth_anything_v2.onnx"

torch.onnx.export(
    model_cpu,
    dummy_input,
    onnx_model_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=12
)
print("Model successfully exported to ONNX format:", onnx_model_path)

# ---------------------------
# 4. Create a Calibration Data Reader for Static Quantization
# ---------------------------
# A simple dummy calibration data reader. For better results, use a small set of real images.
class DummyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, input_name, num_samples=10, input_size=(3, 224, 224)):
        self.input_name = input_name
        self.num_samples = num_samples
        self.input_size = input_size
        self.data = self._generate_data()
    
    def _generate_data(self):
        data = []
        for _ in range(self.num_samples):
            sample = np.random.rand(1, *self.input_size).astype(np.float32)
            data.append({self.input_name: sample})
        return data
    
    def get_next(self):
        if len(self.data) > 0:
            return self.data.pop(0)
        return None

import onnxruntime
session = onnxruntime.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name
calibration_data_reader = DummyCalibrationDataReader(input_name)

# ---------------------------
# 5. Apply Static Quantization with Calibration Data
# ---------------------------
quantized_model_path = "depth_anything_v2_quant.onnx"
quantize_static(
    onnx_model_path,
    quantized_model_path,
    calibration_data_reader,
    quant_format=QuantFormat.QLinearOps,  # Use QLinearConv for convolutions.
    weight_type=QuantType.QInt8
)
print("Static quantization completed. Quantized model saved as:", quantized_model_path)