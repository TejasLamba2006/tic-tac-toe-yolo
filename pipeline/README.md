# ST Edge AI Pipeline — From Model to MPU

> *"We wanted to go from a PyTorch model to a binary running on an STM32MP257F without losing our sanity. This is how we did it."*

## The Story

Picture this: you've trained a YOLO model to detect tic-tac-toe boards. It runs great on your PC. Now you need it on an STM32MP1 — a Cortex-A7 application processor running Linux. Not a microcontroller. Not a simple "compile and flash" situation. A full MPU with an NPU, CMA memory pools, and Yocto-based OS.

The ST Edge AI Developer Cloud can optimize your model for this hardware, but the web UI is manual, slow, and error-prone for CI/CD. So we built a pipeline that automates the entire journey: train → export → quantize → optimize → deploy.

Under the hood, we use the official `stm32ai_dc` Python SDK — ST's own REST API wrapper. No reinventing the wheel. No raw `requests` calls. Just clean, typed Python that does exactly what the cloud UI does, but in a script.

## The Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  YOLO Training   │────▶│  ONNX Export     │────▶│  TFLite INT8    │
│  (Ultralytics)   │     │  (opset 12)      │     │  Quantization   │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Deploy to Board │◀────│  NBG Generation  │◀────│  ST Edge AI DC  │
│  (SSH/SCP)       │     │  (Cloud Optimize)│     │  (Upload + Auth)│
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## How It Works (The Journey)

### 1. Upload — Your model lands in the ST cloud

The pipeline takes your quantized `.tflite` file and uploads it to the Developer Cloud. Authentication happens via ST's SSO (my.st.com), the same way you'd log in through the browser. Tokens are cached locally so you don't re-authenticate on every run.

### 2. Analyze — The cloud dissects your model

Before optimizing, the cloud tells you exactly what you're working with: memory footprint, MACC count, layer breakdown. Think of it as an X-ray of your neural network.

### 3. Optimize (NBG) — This is where the magic happens

For MPU targets (STM32MP1/STM32MP257), the cloud generates a **Neural Binary Graph** — a serialized representation optimized for the Cortex-A7 + Cortex-M4 architecture. This is what the X-LINUX-AI runtime loads on the board. The NBG knows about CMA pools, NPU offloading, and memory layouts that a generic `.tflite` doesn't.

### 4. Download — You get a neatly wrapped package

The optimized `.nb` file downloads back to your machine. Drop it into your Yocto build or SCP it directly to the board.

### 5. Deploy — Drop it into OpenSTLinux

The final stage uses SSH/SCP to push the model to the board's filesystem. If you've configured a verification command, it runs automatically to confirm the model loads and infers correctly.

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[stedge]"

# 2. Set credentials (never hardcode these)
export STEDGE_USERNAME="you@example.com"
export STEDGE_PASSWORD="your_password"

# 3. Run the full pipeline
python run_pipeline.py --config pipeline/config/config.yaml

# 4. Or just the ST Edge AI compilation stage
python run_pipeline.py --stage stedge_compile

# 5. Or use the standalone example
python examples/optimize_for_mpu.py --model build/best.tflite
```

## When Things Go Sideways

**"Invalid credentials"** — Double-check `STEDGE_USERNAME` and `STEDGE_PASSWORD`. The SDK also supports `STM32AI_USERNAME`/`STM32AI_PASSWORD` if you prefer the official naming.

**"stm32ai_dc not installed"** — Install it:
```bash
pip install stm32ai_dc@git+https://github.com/STMicroelectronics/stm32ai-modelzoo-services.git#subdirectory=common/stm32ai_dc
```

**NBG generation times out** — The default timeout is 600 seconds. Complex models may need more. Set `--timeout 900` or higher. Also check if the cloud service is under heavy load.

**"Model not found" after upload** — The SDK references models by filename, not path. Make sure the model file name doesn't contain special characters.

**Proxy issues** — If you're behind a corporate proxy:
```bash
export http_proxy="http://proxy:port"
export https_proxy="http://proxy:port"
export NO_SSL_VERIFY="1"  # Only if proxy requires it
```

## Pipeline Stages

| # | Stage | What It Does |
|---|-------|-------------|
| 1 | `dataset_validation` | Checks images, labels, class distribution |
| 2 | `augmentation` | Brightness, contrast, blur, noise transforms |
| 3 | `training` | YOLOv8 training via Ultralytics |
| 4 | `evaluation` | mAP, precision, recall, confusion matrix |
| 5 | `export` | best.pt → ONNX (opset 12) |
| 6 | `quantization` | ONNX → TFLite INT8 |
| 7 | `stedge_compile` | TFLite → NBG via ST Edge AI DC Cloud |
| 8 | `deployment` | SCP model to board |
| 9 | `verification` | Confirm model loads on-board |
| 10 | `benchmark` | Latency, FPS, memory metrics |
| 11 | `report` | JSON + HTML + Markdown summary |

## Roadmap

- **Cloud benchmarking** — Run inference on real STM32 boards via the cloud benchmark API
- **Validation datasets** — Upload input/output datasets for accuracy verification
- **CI/CD integration** — GitHub Actions workflow for automated retraining on data changes
- **Multi-board support** — Auto-detect board capabilities and optimize accordingly
- **Web dashboard** — Visualize pipeline runs, compare model versions, track accuracy drift

---

*Built for the STM32MP257F-DK with OpenSTLinux and X-LINUX-AI. Designed to work with any STM32MP1/MP2 target.*
