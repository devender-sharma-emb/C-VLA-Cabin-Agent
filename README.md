# C-VLA: Hierarchical Vision-Language-Action Architecture for In-Cabin Agentic AI

A hierarchical two-brain architecture for energy-efficient driver monitoring on automotive edge hardware.

## Results
- System 1: 92.8% accuracy, 7.9ms latency, 126.6 FPS on Jetson Orin Nano Super
- System 2: 307ms reasoning latency on-device
- 65% LLM invocation reduction vs single-model baseline
- 97% latency reduction vs single-model baseline

## Hardware
- Development: MacBook Pro M3 (18GB, MPS)
- Deployment: NVIDIA Jetson Orin Nano Super (8GB, JetPack 6.2, CUDA 12.6)

## Dataset
- DMD (Driver Monitoring Dataset) — 6,405 labeled frames
- 20 in-house recorded behavioral clips

## Files
- cabin_agent_v4.py — Full pipeline: camera + ViT gate + Llama reasoning
- train_cnn_jetson.py — System 1 CNN training on Jetson CUDA
- evaluate_cnn_jetson.py — Evaluation with precision/recall/F1
- benchmark_jetson.py — Latency benchmarking on Jetson
- convert_dmd.py — DMD dataset label conversion

## Paper
See paper/cvla_paper.pdf for the full research paper.

## Citation
If you use this work please cite:
Sharma, D. (2026). C-VLA: A Hierarchical Vision-Language-Action Architecture
for Energy-Efficient In-Cabin Agentic AI on Automotive Edge Hardware.
