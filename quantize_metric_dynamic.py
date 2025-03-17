from onnxruntime.quantization import quantize_dynamic, QuantType
import os

# Input and output model paths
input_model_path = "depth_anything_v2_small_metric.onnx"
output_model_path = "depth_anything_v2_small_metric_int8.onnx"

print("Starting dynamic quantization of METRIC model...")
# Perform dynamic quantization - no calibration data needed
quantize_dynamic(
    input_model_path, 
    output_model_path,
    weight_type=QuantType.QInt8
)

print("Dynamic quantization complete!")

# Compare model sizes
original_size = os.path.getsize(input_model_path) / (1024 * 1024)
quantized_size = os.path.getsize(output_model_path) / (1024 * 1024)
print(f"Original model size: {original_size:.2f} MB")
print(f"Quantized model size: {quantized_size:.2f} MB")
print(f"Size reduction: {(1 - quantized_size/original_size) * 100:.2f}%")