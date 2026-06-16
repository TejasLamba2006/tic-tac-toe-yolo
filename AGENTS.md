# AGENTS.md — STM32MP257 Tic-Tac-Toe Vision (high-signal)

## Entry points to use
- Main pipeline: `python main.py --camera <src> --weights <model>`
  - Note: root `main.py` just forwards to `src.main.main()`.
- OpenCV board-inset visualizer: `python detect.py --weights <model> --camera <src> [--npu]`

## Model formats + what the agent must match with `--npu`
- `.pt` → Ultralytics/PyTorch (CPU/GPU dev); `--npu` is typically for non-`.pt` runtime use.
- `.onnx` → ONNX Runtime + `VSINPUExecutionProvider` (NPU)
- `.tflite` → TFLite + `libvx_delegate` (NPU, INT8)
- `.nb` → `stai_mpu_network` (optimal NPU path)
- `--npu` is the flag that enables NPU acceleration inside `YoloInference` in `src/main.py`/`src/ai/yolo_inference.py`.

## Required deploy command order (build → validate)
- Export ONNX (opset 12):
  `python src/stm32/export_to_onnx.py --weights runs/detect/train/weights/best.pt --output build/best.onnx --opset 12`
- Export TFLite:
  `python src/stm32/export_to_tflite.py --source build/best.onnx --output build/best.tflite`
- INT8 quantize:
  `python scripts/tflite_quant.py` (uses `scripts/config_quant.yaml`)
- Validate ONNX:
  `python3 /usr/local/x-linux-ai/bin/ort-vsinpu-ep-example/ort-vsinpu-ep-example.py build/best.onnx`
- Validate TFLite:
  `python3 /usr/local/x-linux-ai/bin/tflite-vx-delegate-example/tflite-vx-delegate-example.py build/best.tflite`

## STM32 camera gotcha (can silently break streaming)
- On V4L2/DCMIPP `/dev/videoX`: pass `--frame-width 640 --frame-height 480`.
- `src/main.py` explicitly warns that setting resolution on `/dev/videoX` can stop the stream silently.

## NPU memory gotcha
- `src/main.py` probes `/proc/meminfo` `CmaFree` before/after NPU model load.
- If CMA free drops too far, suspect CMA exhaustion during NPU load.

## Tests (known failure mode to expect)
- `python -m pytest src/tests/ -v`
- `src/tests/test_pipeline.py` imports `from src.ai.move_selector import recommend_move`.
  - If `src/ai/move_selector.py` is missing, pytest will fail immediately on import.
  
## Debugging: where to look first
- Orchestration/UI + runtime flags: `src/main.py`
- Camera capture + configuration: `src/vision/camera.py`
- Board detection + warp: `src/vision/board_detector.py`, `src/vision/perspective.py`
- Board cell state + smoothing: `src/vision/board_state.py`
- Inference backend selection: `src/ai/yolo_inference.py`
