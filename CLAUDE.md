# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Two related but independent subsystems living in one repo:

1. **Tic-Tac-Toe vision app** (`src/`, `detect.py`, `main.py`) — real-time camera capture → board detection/warp → YOLO inference → game-state overlay. Runs on a dev PC (PyTorch/CPU) or deployed to an **STM32MP257F-DK** NPU (OpenSTLinux/X-LINUX-AI).
2. **ML training/deployment pipeline** (`pipeline/`, `webui/`) — a staged pipeline that takes a YOLO dataset from `dataset_validation` through `training` → `export` → `quantization` → `stedge_compile` (ST Edge AI Developer Cloud) → `deployment` to the board, plus a FastAPI dashboard (`webui/`) to drive it.

These two subsystems share the exported/quantized model artifacts but have separate entry points and don't import from each other.

## Commands

### Environment setup
```bash
uv python install 3.12          # TensorFlow/quantization requires 3.12, not system 3.14+
uv venv --python 3.12
uv pip install -e ".[all,stedge]"   # training + quantization + stedge SDK
uv pip install -e ".[ui]"           # FastAPI dashboard only
```
Optional dependency groups (`pyproject.toml`): `training`, `quantization`, `stedge`, `ui`, `all`. The base install only gets `pyyaml`, `opencv-python`, `numpy` — most work requires an extra.

### Vision app (tic-tac-toe)
```bash
python main.py --camera <src> --weights <model>     # forwards to src.main.main()
python detect.py --weights yolov8n.pt --camera 0    # PC/PyTorch, board-inset visualizer
python detect.py --weights model.nb --camera auto --npu   # STM32MP257 NPU
```
On STM32 `/dev/videoX` sources, always pass `--frame-width 640 --frame-height 480` explicitly — setting resolution on V4L2/DCMIPP after open can silently kill the stream (see `src/vision/camera.py`).

### ML pipeline
```bash
python run_pipeline.py                          # full pipeline
python run_pipeline.py --stage training          # single stage
python run_pipeline.py --from-stage export        # resume from a stage
python run_pipeline.py --dry-run                  # validate config + print plan, no run
python run_pipeline.py --list-stages              # list stage names
python run_pipeline.py --config my_config.yaml    # non-default config
```
Equivalent: `python -m pipeline` or the installed `pipeline` console script (`pipeline.cli:main`).

Credentials for the ST Edge AI cloud stage come from env vars, never hardcode: `STEDGE_USERNAME`/`STEDGE_PASSWORD` (or `STM32AI_USERNAME`/`STM32AI_PASSWORD`).

### Web dashboard
```bash
python run_ui.py                  # http://127.0.0.1:8000
python run_ui.py --port 8080
```
No auth — do not bind `--host 0.0.0.0` outside a trusted network; it can execute the pipeline subprocess.

### Tests
```bash
python -m pytest src/tests/ -v      # vision app tests
python -m pytest pipeline/ -v       # pipeline tests
python -m pytest src/tests/test_board_state.py::test_name -v   # single test
```
`pytest.ini_options` in `pyproject.toml` sets `testpaths = ["pipeline", "src/tests"]`, so plain `python -m pytest` also works from repo root. Known failure mode: `src/tests/test_pipeline.py` imports `from src.ai.move_selector import recommend_move` — if `move_selector.py` doesn't exist yet, that import fails the whole test file.

### Manual model export/deploy chain (order matters)
```bash
python src/stm32/export_to_onnx.py --weights runs/detect/train/weights/best.pt --output build/best.onnx --opset 12
python src/stm32/export_to_tflite.py --source build/best.onnx --output build/best.tflite
python scripts/tflite_quant.py                      # uses scripts/config_quant.yaml, INT8 per-tensor
python3 /usr/local/x-linux-ai/bin/ort-vsinpu-ep-example/ort-vsinpu-ep-example.py build/best.onnx
python3 /usr/local/x-linux-ai/bin/tflite-vx-delegate-example/tflite-vx-delegate-example.py build/best.tflite
```
This is exactly what the pipeline's `export`/`quantization`/`stedge_compile`/`verification` stages automate — use the manual chain only for local debugging of one step.

## Architecture

### Vision app data flow (per frame)
```
YoloInference -> analyze_frame(frame, detector, board_detector, board_estimator, ...) -> _build_board_inset(analysis, inset_size) -> cv2.imshow(...)
```
Key modules:
- `src/main.py` — orchestration, CLI flags, CMA memory probing before/after NPU model load (`_read_cma_free_kb`, reads `/proc/meminfo`, always `None` on Windows)
- `src/ai/yolo_inference.py::YoloInference` — backend-agnostic inference wrapper; **selects backend by file extension**: `.pt` → Ultralytics/PyTorch, `.onnx` → ONNX Runtime + `VSINPUExecutionProvider`, `.tflite` → TFLite + `libvx_delegate.so`, `.nb` → `stai_mpu_network`. All non-Ultralytics backends share one preprocessing path: resize → BGR→RGB → normalize/cast → NCHW transpose → batch dim.
- `src/vision/board_detector.py::BoardDetector` + `src/vision/perspective.py` — board geometry detection and perspective warp to canonical view
- `src/vision/board_state.py::BoardStateEstimator` — turns detections into cell occupancy state
- `src/vision/stability.py::BoardGeometryTracker` — smooths/stabilizes board geometry across frames
- `src/vision/camera.py` — capture open/configure (the V4L2 resolution gotcha lives here)

`--npu` is the flag that enables NPU acceleration inside `YoloInference`; it's meaningful for `.onnx`/`.tflite`/`.nb`, not `.pt`.

### Pipeline architecture
`pipeline/cli.py::_build_stages()` instantiates all `Stage` subclasses in a fixed execution order (dataset_validation → augmentation → training → evaluation → export_onnx → quantization → stedge_compile → deployment → verification → benchmark → report), then `pipeline/core/runner.py::PipelineRunner.run_all()` walks them.

- `pipeline/core/stage.py::Stage` — abstract base every stage subclasses. `execute()` is the orchestration wrapper all stages get for free: `validate_inputs()` → `should_skip()` (resume support) → `run()` (implemented per-stage) → returns a `StageResult` (status/message/duration/artifacts/metrics). Only `run()` needs implementing in new stages.
- `pipeline/core/context.py::PipelineContext` — the mutable state threaded through every stage. Stages read/write fields like `best_weights`, `onnx_model`, `tflite_model`, `nb_model`, `training_metrics` here so downstream stages can find upstream outputs. `stage_artifacts_dir(name)` gives each stage its own subdir under `artifacts_dir`.
- `pipeline/core/manifest.py` — persists stage results (`ctx.record_result`) so runs are resumable via `--from-stage`.
- `pipeline/config/loader.py` — config precedence is env vars > `.env` file > `config.yaml` > built-in defaults. `pipeline/config/schema.py::PipelineConfig` is the typed config; `config.validate()` runs before any stage executes.
- `pipeline/stedge_wrapper.py` — wraps the vendored `stm32ai_dc` SDK (ST's official REST client for the ST Edge AI Developer Cloud) for the `stedge_compile` stage. The SDK is vendored under `vendor/common/stm32ai_dc` (copied from `STMicroelectronics/stm32ai-modelzoo-services`, commit `0f6210ed`) because upstream lacks a `pyproject.toml` and the whole repo has Windows-incompatible path lengths. It ships with the project install via `[tool.setuptools.packages.find]` (importable as `common.stm32ai_dc`); the `[stedge]` extra only declares its runtime deps (`requests`, `marshmallow`, `tqdm`).

### Web dashboard architecture
`webui/app.py` is intentionally a thin FastAPI router: dataset inspection/config generation lives in `webui/config_builder.py`, subprocess execution and log streaming (over WebSocket) live in `webui/process_manager.py`, and preflight/results views are `webui/preflight.py` / `webui/results.py`. `app.py` only wires HTTP/WebSocket routes to those modules — don't put business logic directly in route handlers.

Note: `templates.TemplateResponse(request, "name.html", {})` requires `request` as the first positional arg on current Starlette — the older `TemplateResponse("name.html", {"request": request})` form was removed and raises a confusing "unhashable dict" error.

### Config files at repo root
Multiple dataset-specific YAML configs exist at root (`data.yaml`, `board.yaml`, `computer_accessories.yaml`) plus generated ones under `generated_configs/` — these are pipeline dataset configs for different training runs, not app config. `pipeline/config/config.yaml` is the actual pipeline run config consumed by `run_pipeline.py`.

## Gotchas worth knowing before touching related code

- **NPU memory**: `src/main.py` checks `/proc/meminfo` `CmaFree` before/after NPU model load; if CMA free drops too far, suspect CMA exhaustion, not a model bug.
- **Camera resolution on STM32**: setting resolution on `/dev/videoX` (V4L2/DCMIPP) can silently stop the stream — always pass `--frame-width`/`--frame-height` up front rather than reconfiguring after open.
- **Python version split**: system Python may be 3.14+, but TensorFlow/quantization work needs the `.venv` on Python 3.12 — check which interpreter is active before debugging quantization stage failures.
- **`stm32ai_dc` SDK is vendored, not pip-installed from PyPI** — it lives at `vendor/common/stm32ai_dc` (copied from ST's `stm32ai-modelzoo-services` repo at commit `0f6210ed`), ships with the project install, and imports as `common.stm32ai_dc`. The `[stedge]` extra only adds its runtime deps (`requests`, `marshmallow`, `tqdm`).
