# Drone depth estimation

Built during Entrepreneur First's Builders Retreat in early 2025. The goal was monocular depth estimation fast enough to run on a drone, as a first step towards obstacle avoidance and SLAM. It didn't really work end to end, but it was my first steps into robotics, SLAM and quantization, and I learnt a ton: running a metric depth model on a live camera feed, exporting it to ONNX, and quantizing a ViT-based model to INT8.

This repo is a copy of [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) with my scripts added at the root. Code comments are in French.

## What I built on top of Depth Anything V2

All scripts use the small metric model (`vits` encoder, Hypersim indoor weights, max depth 20 m).

- `depth_webcam.py`: real-time webcam demo in PyTorch. Runs inference on a background thread (CUDA, Apple MPS or CPU), shows the camera feed next to a colour-mapped depth map with a distance colour bar, the metric depth at the image centre and a live FPS counter. Optional mixed precision on CUDA, grayscale output and video recording.
- `export_onnx.py`: exports the model to ONNX (224x224 input, opset 12), then applies ONNX Runtime static INT8 quantization calibrated on random inputs.
- `quantize_metric_dynamic.py`: ONNX Runtime dynamic INT8 quantization (weights only, no calibration data) and prints the size before and after.
- `export_quant_pth.py`: PyTorch dynamic INT8 quantization of the `Linear` layers (`qnnpack` engine, for ARM / Apple Silicon CPUs), saved as TorchScript.
- `depth_webcam_quant.py`: the same webcam demo running the ONNX model through ONNX Runtime (CPU, or CUDA when available), at 224x224 input.
- `cuda-check.py`: prints the Python, PyTorch, CUDA and GPU setup, times a 1000x1000 matrix multiply on the GPU and lists likely causes when CUDA is missing.

Rough edges, left as they were: `depth_webcam_quant.py` loads the float export `checkpoints/depth_anything_v2.onnx` rather than a quantized file; `quantize_metric_dynamic.py` expects `depth_anything_v2_small_metric.onnx`, so the export has to be renamed first; `export_quant_pth.py` calls `forward(x, is_training=False)`, which the upstream model's `forward` does not accept; and the static quantization uses random calibration data, not real images.

## Running it

```bash
pip install -r requirements.txt onnxruntime
mkdir -p checkpoints
```

Download the small metric Hypersim checkpoint from upstream's Hugging Face page into `checkpoints/` (the `.pth` and `.onnx` files committed here are my local copies; prefer the upstream weights):

```bash
curl -L -o checkpoints/depth_anything_v2_metric_hypersim_vits.pth \
  "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth?download=true"
```

Other sizes and the outdoor (VKITTI) weights are listed in [`metric_depth/README.md`](metric_depth/README.md).

Webcam demo (press `q` to quit):

```bash
python depth_webcam.py --camera-id 0              # add --mixed-precision on CUDA, --save-video to record
```

Export and quantization, run from the repo root:

```bash
python export_onnx.py                  # writes depth_anything_v2.onnx and depth_anything_v2_quant.onnx
cp depth_anything_v2.onnx depth_anything_v2_small_metric.onnx
python quantize_metric_dynamic.py      # writes depth_anything_v2_small_metric_int8.onnx
python export_quant_pth.py             # writes depth_anything_v2_quantized.pt
```

ONNX webcam demo (reads `checkpoints/depth_anything_v2.onnx`):

```bash
cp depth_anything_v2.onnx checkpoints/
python depth_webcam_quant.py
```

`python cuda-check.py` checks the GPU setup.

## Credit

Built on Depth Anything V2 by Yang et al. (HKU / TikTok), Apache-2.0: [upstream repo](https://github.com/DepthAnything/Depth-Anything-V2), [paper](https://arxiv.org/abs/2406.09414). The upstream licence is kept in [`LICENSE`](LICENSE), and the original README is in [`README.upstream.md`](README.upstream.md).
