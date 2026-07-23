# STM32 YOLO Pipeline

Train a YOLO object detection model and deploy it to an STM32MP257F-DK board through an automated pipeline. The pipeline handles everything from dataset validation to quantized model export, with optional cloud compilation via ST Edge AI Developer Cloud.

## What this does

You give it a YOLO-formatted dataset (images + label files) and it:

1. Validates your dataset (checks images, labels, class distribution)
2. Augments images (brightness, contrast, blur, noise)
3. Trains a YOLOv8n model
4. Evaluates precision, recall, mAP
5. Exports to ONNX
6. Quantizes to INT8 TFLite
7. Compiles to NBG via ST Edge AI cloud (optional)
8. Deploys to your STM32 board via SSH (optional)

Each step saves its output. If the pipeline crashes mid-run, it picks up where it left off.

## Prerequisites

### Windows

- **Python 3.12** — the pipeline does not work with Python 3.13 or 3.14. TensorFlow and the quantization stage require 3.12 specifically.
- **uv** (package manager) — install from https://docs.astral.sh/uv/getting-started/installation/
- **Git** — install from https://git-scm.com/download/win
- A CUDA-capable GPU is optional but makes training 10-50x faster. The pipeline falls back to CPU automatically.

### Linux (Ubuntu/Debian)

```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv python3-pip git
```

For GPU support, install CUDA toolkit from https://developer.nvidia.com/cuda-downloads

### Clone the repo

```bash
git clone https://github.com/TejasLamba2006/STM32M257f-dk-x-linux-ai.git
cd STM32M257f-dk-x-linux-ai
```

## Setup

### Step 1: Create a virtual environment

**Windows:**
```bash
uv python install 3.12
uv venv --python 3.12
```

**Linux:**
```bash
uv python install 3.12
uv venv --python 3.12
```

If you don't have `uv`, install it first:

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2: Activate the environment

**Windows (cmd):**
```cmd
.venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Linux:**
```bash
source .venv/bin/activate
```

You should see `(.venv)` at the start of your terminal prompt after this.

### Step 3: Install dependencies

```bash
uv pip install -e ".[all]"
```

This installs everything: training, quantization, the web dashboard, and the ST Edge AI SDK.

If you only need training and local stages (no cloud compilation, no web dashboard):

```bash
uv pip install -e ".[training]"
```

### Step 4: Verify the install

```bash
python run_pipeline.py --list-stages
```

You should see 11 stages listed. If you get `ModuleNotFoundError: No module named 'cv2'`, your virtual environment is not activated — go back to Step 2.

## Your dataset

The pipeline expects a YOLO-formatted dataset with this structure:

```
dataset4/
  train/
    images/     # .jpg, .png, etc.
    labels/     # .txt files with YOLO annotations
  val/
    images/
    labels/
  test/
    images/
    labels/
```

Each label file has one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized (0.0 to 1.0). If you trained with Ultralytics before, your dataset already follows this format.

If you have a flat folder of images with labels and need to split into train/val/test, edit `pipeline/config/config.yaml`:

```yaml
dataset:
  path: your_dataset_folder
  split_enabled: true
  split_ratios:
    train: 0.8
    val: 0.1
    test: 0.1
```

## Running the pipeline

### Full pipeline

```bash
python run_pipeline.py
```

This runs all 11 stages in order. It takes a while on CPU (30+ minutes for training). On GPU, training alone is 1-5 minutes for small datasets.

### Run a single stage

```bash
python run_pipeline.py --stage training
```

### Resume from a specific stage

```bash
python run_pipeline.py --from-stage export
```

Skips everything before `export` and continues from there.

### Dry run (no actual work, just checks)

```bash
python run_pipeline.py --dry-run
```

### List available stages

```bash
python run_pipeline.py --list-stages
```

## Configuration

All settings live in `pipeline/config/config.yaml`. The main things you will want to change:

```yaml
# Your dataset location
dataset:
  path: dataset4

# Training settings
training:
  model: yolov8n.pt    # base model (yolov8n.pt is small and fast)
  epochs: 100           # more epochs = better but slower
  batch: 4              # lower if you run out of GPU memory
  imgsz: 320            # input image size
  device: 0             # 0 = first GPU, 'cpu' = no GPU

# ST Edge AI cloud (optional)
stedge:
  enabled: false        # set to true when you have credentials
```

Credentials for the cloud stage go in environment variables, never in the config file:

```bash
# Windows (cmd)
set STEDGE_USERNAME=your_email@example.com
set STEDGE_PASSWORD=your_password

# Windows (PowerShell)
$env:STEDGE_USERNAME="your_email@example.com"
$env:STEDGE_PASSWORD="your_password"

# Linux
export STEDGE_USERNAME="your_email@example.com"
export STEDGE_PASSWORD="your_password"
```

## Web dashboard

A browser-based UI for running the pipeline and inspecting results:

```bash
python run_ui.py
```

Then open http://127.0.0.1:8000 in your browser. No login required. Do not expose this to the internet.

## Project structure

```
.
├── run_pipeline.py          # entry point for the pipeline
├── run_ui.py                # entry point for the web dashboard
├── pipeline/
│   ├── cli.py               # command-line interface
│   ├── core/
│   │   ├── runner.py        # orchestrates stage execution
│   │   ├── stage.py         # base class for all stages
│   │   ├── context.py       # shared state between stages
│   │   └── manifest.py      # tracks what ran (for resume)
│   ├── config/
│   │   ├── config.yaml      # main config file
│   │   └── schema.py        # config validation
│   └── stages/
│       ├── dataset_validation.py
│       ├── augmentation.py
│       ├── training.py
│       ├── evaluation.py
│       ├── export.py
│       ├── quantization.py
│       ├── stedge_compile.py
│       ├── deployment.py
│       ├── verification.py
│       ├── benchmark.py
│       └── report.py
├── webui/                   # FastAPI dashboard
├── scripts/                 # standalone utilities
├── stm32ai-modelzoo-services/  # vendored ST SDK
└── dataset4/                # example dataset
```

## Troubleshooting

### "No module named 'cv2'"

Your virtual environment is not activated, or opencv is not installed. Run:

```bash
uv pip install -e ".[all]"
```

Make sure you see `(.venv)` in your terminal prompt.

### "No module named 'numpy'" or numpy import errors

This happens when numpy was installed for a different Python version. Fix:

```bash
uv pip install --force-reinstall numpy
```

### Training fails with "No module named 'ultralytics'"

Install the training extra:

```bash
uv pip install -e ".[training]"
```

### TensorFlow quantization fails on Python 3.13/3.14

TensorFlow only supports Python 3.12 and below. You must use Python 3.12:

```bash
uv python install 3.12
uv venv --python 3.12
uv pip install -e ".[all]"
```

### Verification stage fails when deployment is disabled

This is a known issue. The verification stage requires `deployment.host` and `deployment.username` to be set, even when deployment is disabled. To work around it, either:

1. Set deployment config in `config.yaml`:
   ```yaml
   deployment:
     enabled: true
     host: your_board_ip
     username: your_board_user
   ```

2. Or run stages individually:
   ```bash
   python run_pipeline.py --from-stage training --from-stage quantization
   ```

### GPU out of memory

Lower the batch size in `config.yaml`:

```training:
  batch: 2   # or even 1
```

### Pipeline says "skip" for a stage you want to rerun

The pipeline remembers completed stages. Delete the artifacts for that stage:

```bash
# Windows
rmdir /s /q artifacts\training

# Linux
rm -rf artifacts/training
```

### ST Edge AI cloud: "Invalid credentials"

Double-check your `STEDGE_USERNAME` and `STEDGE_PASSWORD` environment variables. The SDK also accepts `STM32AI_USERNAME`/`STM32AI_PASSWORD`.

### ONNX export fails

Make sure the training stage completed and `artifacts/training/weights/best.pt` exists.

## License

See the repository for license information.
