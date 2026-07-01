# PIPELINE.md

A step-by-step guide to training a YOLOv8 model on your own dataset and deploying it to an STM32MP257F board via the NPU. Works for any object detection dataset — cats, cars, circuit boards, doesn't matter.

---

## Table of contents

1. [What this pipeline does](#what-this-pipeline-does)
2. [What you need](#what-you-need)
3. [Install dependencies](#install-dependencies)
4. [How the pipeline works](#how-the-pipeline-works)
5. [Preparing your dataset](#preparing-your-dataset)
6. [Creating your config file](#creating-your-config-file)
7. [Running the pipeline](#running-the-pipeline)
8. [What each stage does](#what-each-stage-does)
9. [The scripts directory](#the-scripts-directory)
10. [Running inference on your PC](#running-inference-on-your-pc)
11. [Deploying to the STM32 board](#deploying-to-the-stm32-board)
12. [Configuration reference](#configuration-reference)
13. [Troubleshooting](#troubleshooting)
14. [FAQ](#faq)

---

## What this pipeline does

You point it at a folder of images with bounding-box labels. It trains a YOLOv8n model on your GPU, exports it to ONNX, quantizes it to INT8 TFLite, and optionally compiles it for the STM32MP257F NPU via ST's cloud service.

One command runs everything. If a stage already finished, it gets skipped. If something fails, you fix it and run the same command again — it picks up where it left off.

End results:
- `best.pt` — PyTorch weights for running on your PC
- `best.onnx` — ONNX model, works across frameworks
- `best_int8.tflite` — Quantized TFLite for the STM32 NPU
- `best.nb` — ST's Neural Binary Graph (if you run the cloud compilation stage)

---

## What you need

**Hardware:**
- A PC with an NVIDIA GPU (RTX 2060 or better recommended). Training on CPU is possible but painfully slow.
- Optionally: an STM32MP257F-DK board for on-device deployment.

**Software:**
- Python 3.11 (3.11 is what we tested with)
- Git
- A terminal

**No ML experience required.** If you can open a terminal and copy-paste commands, you can run this pipeline.

---

## Install dependencies

Open a terminal in the project root and run these commands in order:

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install PyTorch with CUDA support (for GPU training)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install the rest
pip install ultralytics opencv-python numpy pillow \
    onnx onnxruntime onnx2tf onnx-graphsurgeon sng4onnx \
    ai-edge-litert tensorflow tf_keras \
    pyyaml
```

Verify CUDA is working:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Should print `True`. If it says `False`, reinstall PyTorch using the CUDA-specific index URL from the PyTorch website.

---

## How the pipeline works

```
Your images + labels
        |
        v
  [1] Validate dataset     -- checks images open, labels are valid YOLO format
        |
        v
  [2] Augment              -- brightness, contrast, blur, noise (optional)
        |
        v
  [3] Train                -- YOLOv8n on your GPU
        |
        v
  [4] Evaluate             -- mAP, precision, recall on test split
        |
        v
  [5] Export to ONNX       -- universal model format
        |
        v
  [6] Quantize to TFLite   -- INT8 quantization for edge deployment
        |
        v
  [7] Compile for NPU      -- ST cloud generates optimized .nb file
        |
        v
  [8] Deploy to board      -- SCP the model to your STM32
        |
        v
  [9] Verify               -- confirm it loads and runs on the board
        |
        v
  [10] Benchmark            -- measure latency and FPS on the board
        |
        v
  [11] Report               -- summary of everything
```

You can stop at any stage. If you just want a `.pt` model for your PC, stages 1-4 are enough. For the board, you need through stage 7.

---

## Preparing your dataset

The pipeline expects a specific folder structure. Here's how to set it up from scratch.

### Folder structure

```
your_dataset/
    train/
        images/         -- training images (.jpg, .png, .bmp)
        labels/         -- one .txt file per image
    val/
        images/         -- validation images
        labels/
    test/
        images/         -- test images (used for final evaluation)
        labels/
```

### Label format (YOLO bounding boxes)

Each image has a matching `.txt` file with the same name. For example, `photo1.jpg` has `photo1.txt`. Each line in the `.txt` file represents one object:

```
<class_id> <center_x> <center_y> <width> <height>
```

All four coordinates are normalized to 0-1 relative to the image dimensions. Example:

```
0 0.5 0.3 0.2 0.4
2 0.1 0.7 0.15 0.25
```

This means: class 0 at 50% x, 30% y (20% wide, 40% tall) and class 2 at 10% x, 70% y (15% wide, 25% tall).

An empty `.txt` file is fine — it just means no objects in that image.

### The data.yaml file

Create a `data.yaml` file (Ultralytics format) in your dataset folder:

```yaml
train: ./train/images
val: ./val/images
test: ./test/images

nc: 3
names: ['cat', 'dog', 'bird']
```

Replace the class names and count with your own. Paths are relative to the `data.yaml` location.

### Splitting your data

If your images are in a flat folder (no train/val/test split), use the split script:

```bash
python scripts/split_dataset.py --input your_dataset/ --ratios 0.8 0.1 0.1
```

This creates the train/val/test folders automatically.

### Labeling tools

You need bounding-box labels. Options:

- **[CVAT](https://cvat.ai/)** — Free, web-based, supports YOLO export
- **[Label Studio](https://labelstud.io/)** — Free, web-based, supports YOLO export
- **[Roboflow](https://roboflow.com/)** — Has a free tier, handles labels and augmentation
- **Auto-labeling with Grounding DINO** — Zero-shot detection, no manual work. Script: `scripts/auto_label_dino.py`

### Example datasets in this repo

Two datasets are included for demonstration:

**Boards (4 classes):** `dataset_boards/` — detects Arduino Uno Q, STM32MP257F-DK, Raspberry Pi Model B, STM32MP157F-DK. 223 images, manually split 80/10/10, augmented to 1561 images.

**Accessories (10 classes):** `dataset_accessories/` — detects Ac, Chair, Cpu, Digital board, Fire Extinguisher, Keyboard, Light, Monitor, Mouse, Projector. 2584 images from the LabEquipVis dataset, pre-split by the authors.

---

## Creating your config file

Copy one of the existing configs and modify it. The config is a YAML file that tells the pipeline where everything is and how to train.

Here's a minimal config for your own dataset:

```yaml
# my_dataset.yaml

artifacts_dir: artifacts_my_dataset

logging:
  level: INFO
  log_dir: artifacts_my_dataset/logs
  log_to_console: true

dataset:
  path: your_dataset           # folder containing train/val/test
  train_images: train/images
  train_labels: train/labels
  val_images: val/images
  val_labels: val/labels
  test_images: test/images
  test_labels: test/labels
  image_extensions: [".jpg", ".jpeg", ".png", ".bmp"]
  split_enabled: false         # set true if you need auto-splitting

augmentation:
  enabled: true                # false if your dataset is already augmented
  output_dir: augmented
  brightness_factors: [0.6, 1.4]
  contrast_factors: [0.7, 1.4]
  gaussian_blur: true
  gaussian_blur_kernel: 5
  noise_stddev: 10.0
  rotate_enabled: false
  rotate_max_degrees: 5.0
  perspective_enabled: false
  perspective_max_shift: 15
  copy_labels: true

training:
  model: yolov8n.pt            # pretrained weights to start from
  data_yaml: your_dataset/data.yaml   # <-- IMPORTANT: path to your data.yaml
  epochs: 50
  batch: 4
  imgsz: 320
  workers: 0
  patience: 15
  optimizer: auto
  lr0: 0.01
  lrf: 0.01
  momentum: 0.937
  weight_decay: 0.0005
  device: 0                    # GPU index, or "cpu"
  seed: 0
  deterministic: true
  amp: true
  cache: false
  plots: true
  save: true
  exist_ok: true
  resume: false
  project: null
  name: null

export:
  opset: 12
  imgsz: 320
  simplify: false
  output_dir: export

quantization:
  enabled: true
  quantization_type: per_channel
  quantization_input_type: uint8
  quantization_output_type: float
  input_shape: [320, 320, 3]
  rescaling_scale: 255.0
  rescaling_offset: 0.0
  fake: false
  calib_dataset_path: ""
  output_dir: quantization
  model_name: my_model
  use_case: od_my_dataset

stedge:
  enabled: false               # true if you have ST cloud credentials
  target: "STM32MP257F-DK"
  optimization: ""
  board: ""
  runtime: ""
  output_dir: compilation
  username: ""
  password: ""
  timeout: 600

deployment:
  enabled: false               # true if deploying via SSH
  host: ""
  port: 22
  username: ""
  password: ""
  ssh_key: ""
  remote_model_directory: ""
  remote_working_directory: ""
  inference_command: ""
  verification_command: ""
  benchmark_command: ""

benchmark:
  enabled: false
  iterations: 100
  warmup_iterations: 10
  output_dir: benchmark

report:
  enabled: true
  formats: [json, html, markdown]
  output_dir: reports
```

**The most important line:** `training.data_yaml`. This points to the Ultralytics-format YAML that defines your classes. It must be correct or training won't start.

---

## Running the pipeline

```bash
# Full pipeline (all stages)
python run_pipeline.py --config my_dataset.yaml

# Single stage
python run_pipeline.py --config my_dataset.yaml --stage training

# Resume from a specific stage (skip everything before it)
python run_pipeline.py --config my_dataset.yaml --from-stage export_onnx

# Dry run (validate config, don't run anything)
python run_pipeline.py --config my_dataset.yaml --dry-run

# List all available stages
python run_pipeline.py --config my_dataset.yaml --list-stages
```

### What you see

The pipeline prints colored output to your terminal:
- **Cyan bold** "Starting training" — stage is running
- **Yellow** "Skipping: ..." — stage already done, being skipped
- **Green** "success (43.1s)" — stage finished successfully
- **Red** text — something went wrong

### Resume behavior

If a stage already produced its output, the pipeline skips it. To re-run a stage, delete its artifacts folder:

```bash
# Force retraining
rm -rf artifacts_my_dataset/training

# Force re-export
rm -rf artifacts_my_dataset/export
```

---

## What each stage does

| # | Stage | What happens | When to skip |
|---|-------|-------------|-------------|
| 1 | `dataset_validation` | Scans images and labels. Checks that images open, labels exist, bounding boxes are valid YOLO format (normalized 0-1). Writes a report. | Never — always run first |
| 2 | `augmentation` | Applies brightness, contrast, blur, and noise transforms to expand your dataset. Copies labels alongside. | If your dataset is already large enough or already augmented |
| 3 | `training` | Trains YOLOv8n from a pretrained checkpoint on your GPU. Saves `best.pt` and `last.pt`. | Never — this is the core stage |
| 4 | `evaluation` | Runs the trained model on the test split. Computes mAP, precision, recall. Generates confusion matrix plots. | If you don't care about metrics |
| 5 | `export_onnx` | Converts `best.pt` to ONNX format (opset 12). | If you only want the `.pt` file |
| 6 | `quantization` | Converts ONNX to TFLite INT8 via onnx2tf. Uses a representative dataset for calibration. | If you're not deploying to edge |
| 7 | `stedge_compile` | Uploads the TFLite model to ST Edge AI Developer Cloud. Downloads the optimized `.nb` file. Needs `stm32ai_dc` and ST account credentials. | If you're not using an STM32 NPU |
| 8 | `deployment` | SCPs the model to the board via SSH. | If you're not deploying remotely |
| 9 | `verification` | Runs a quick inference on the board to confirm the model loads. | If you're not deploying |
| 10 | `benchmark` | Measures inference latency, FPS, and memory on the board. | If you're not deploying |
| 11 | `report` | Aggregates everything into JSON, HTML, and Markdown reports. | Optional |

---

## The scripts directory

`scripts/` contains standalone tools. Run them independently — they don't need the pipeline framework.

### Inference and testing

**`scripts/inference.py`** — Live camera inference. Pick a model and camera, see detections in real time. Supports `.pt`, `.onnx`, `.tflite`, and `.nb` formats. See [Running inference](#running-inference-on-your-pc) below.

### Dataset preparation

**`scripts/split_dataset.py`** — Splits a flat `images/`+`labels/` folder into train/val/test.

```bash
python scripts/split_dataset.py --input my_flat_dataset/ --ratios 0.8 0.1 0.1
```

**`scripts/capture.py`** — Captures images from a webcam and saves them. Useful for building a training dataset.

**`scripts/augment_dataset.py`** — Standalone augmentation. The pipeline has its own augmentation stage, but this script works independently on flat datasets.

### Labeling

**`scripts/auto_label_dino.py`** — Zero-shot auto-labeling using Grounding DINO. Give it a text prompt like "fire extinguisher" and it finds and boxes matching objects in your images. No manual annotation needed.

```bash
python scripts/auto_label_dino.py --input images/ --prompt "fire extinguisher" --output labels/
```

**`scripts/pseudo_labelling.py`** — Uses a pretrained model to generate labels for unlabeled images, then lets you review and correct them.

### Model conversion

**`scripts/tflite_quant.py`** — Standalone TFLite INT8 quantization. The pipeline's quantization stage wraps similar logic.

**`scripts/export_saved.py`** — Exports a model to TensorFlow SavedModel format (intermediate step for TFLite conversion).

**`scripts/diagnose_yolo_tflite.py`** — Debug tool for TFLite models. Checks tensor shapes, quantization parameters, and runs a test inference.

---

## Running inference on your PC

Once training finishes, test your model on a live camera feed:

```bash
# Interactive — asks you to pick camera and model
python scripts/inference.py

# Specify directly
python scripts/inference.py --model artifacts_my_dataset/training/weights/best.pt --camera 0

# ONNX model
python scripts/inference.py --model artifacts_my_dataset/export/best.onnx --camera 0

# Higher confidence threshold
python scripts/inference.py --model best.pt --camera 0 --conf 0.5
```

### What it does

1. Scans for cameras (indices 0-9) and lets you pick one.
2. Finds model files (`.pt`, `.onnx`, `.tflite`, `.nb`) in `artifacts*/`.
3. Loads the model with the right backend.
4. Opens the camera and runs detection in a loop.
5. Draws bounding boxes with class labels and confidence.
6. Shows an FPS counter.

Press `Q` or `ESC` to quit.

### Supported model formats

| Format | Backend | Where it runs |
|--------|---------|--------------|
| `.pt` | Ultralytics | PC (GPU) |
| `.onnx` | onnxruntime | PC or STM32 with NPU (`--npu`) |
| `.tflite` | tflite_runtime | STM32 with `libvx_delegate` (`--npu`) |
| `.nb` | stai_mpu_network | STM32 NPU (optimal) |

---

## Deploying to the STM32 board

This part requires ST Edge AI Developer Cloud credentials and the `stm32ai_dc` package.

### 1. Set credentials (never hardcode them in config files)

```bash
# Windows
set STEDGE_USERNAME=you@example.com
set STEDGE_PASSWORD=your_password

# Mac/Linux
export STEDGE_USERNAME="you@example.com"
export STEDGE_PASSWORD="your_password"
```

### 2. Install stm32ai_dc

```bash
pip install -e ".[stedge]"
```

### 3. Enable the stages in your config

Set `stedge.enabled: true` and `deployment.enabled: true`. Fill in your board's SSH details for deployment.

### 4. Run the pipeline from the compilation stage onward

```bash
python run_pipeline.py --config my_dataset.yaml --from-stage stedge_compile
```

### What the cloud does

- Uploads your `.tflite` model to ST's servers
- Analyzes it (memory footprint, MACC count, layer breakdown)
- Generates a Neural Binary Graph (`.nb`) optimized for the Cortex-A7 + Cortex-M4 NPU
- Downloads the `.nb` file back to your machine

The `.nb` file is what actually runs on the board. It knows about CMA memory pools and NPU offloading that a generic `.tflite` doesn't.

---

## Configuration reference

### Dataset section

| Field | What it does | Example |
|-------|-------------|---------|
| `path` | Root folder of your dataset (relative to project root) | `my_dataset` |
| `train_images` | Subfolder for training images | `train/images` |
| `train_labels` | Subfolder for training labels | `train/labels` |
| `val_images` | Subfolder for validation images | `val/images` |
| `val_labels` | Subfolder for validation labels | `val/labels` |
| `test_images` | Subfolder for test images | `test/images` |
| `test_labels` | Subfolder for test labels | `test/labels` |
| `image_extensions` | Which image file types to look for | `[".jpg", ".png"]` |
| `split_enabled` | Auto-split a flat dataset into train/val/test | `false` |
| `split_ratios` | Ratio for auto-splitting | `train: 0.8, val: 0.1, test: 0.1` |

### Training section

| Field | What it does | Default |
|-------|-------------|---------|
| `model` | Pretrained weights to start from | `yolov8n.pt` |
| `data_yaml` | Path to Ultralytics data YAML | `data.yaml` |
| `epochs` | Number of training epochs | `50` |
| `batch` | Batch size (lower if GPU runs out of memory) | `4` |
| `imgsz` | Input image size | `320` |
| `device` | GPU index (`0`) or `cpu` | `0` |
| `patience` | Early stopping patience | `15` |
| `lr0` | Initial learning rate | `0.01` |
| `amp` | Automatic mixed precision | `true` |

### Quantization section

| Field | What it does | Default |
|-------|-------------|---------|
| `enabled` | Run quantization stage | `true` |
| `quantization_type` | `per_channel` or `per_tensor` | `per_channel` |
| `quantization_input_type` | Input data type: `uint8`, `int8`, `float` | `uint8` |
| `quantization_output_type` | Output data type | `float` |

### ST Edge AI section

| Field | What it does | Default |
|-------|-------------|---------|
| `enabled` | Run cloud compilation | `false` |
| `target` | Target board | `STM32MP257F-DK` |
| `timeout` | Cloud operation timeout (seconds) | `600` |

Credentials should come from environment variables (`STEDGE_USERNAME`, `STEDGE_PASSWORD`), not from the config file.

---

## Troubleshooting

### "CUDA out of memory"
Reduce `batch` in the config (try 2 or 1). Close other GPU-using apps.

### Training is slow
Check that CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`. If `False`, reinstall PyTorch with CUDA support.

### "Data YAML not found"
The `data_yaml` path is relative to the project root. If your dataset is at `my_dataset/data.yaml`, set `data_yaml: my_dataset/data.yaml` in the config.

### "Label file is empty" warnings
Some images have no objects. The pipeline warns but continues. This is normal.

### Quantization fails with onnx2tf
Make sure `onnx2tf` and `onnx-graphsurgeon` are installed. Try setting `simplify: true` in the export config.

### stedge_compile fails
- Verify credentials: `echo $STEDGE_USERNAME`
- The `stm32ai_dc` package must be installed from the vendored copy
- Cloud operations can take 5-10 minutes. Increase `timeout` if needed.

### Pipeline skips a stage
The output already exists. Delete the stage's folder in `artifacts*/` to force a re-run.

### "No image directories found"
Your dataset folder structure doesn't match the config. Check that the paths in `dataset` section actually exist.

### Images are the wrong size
No problem — YOLO resizes to `imgsz` automatically. But extremely small images (<32px) may not detect well.

---

## FAQ

**Do I need to know ML to use this?**
No. Prepare your dataset in the right folder structure, create a config file (copy an existing one and change the paths), and run the command.

**What if I have a classification dataset, not detection?**
This pipeline is for object detection (bounding boxes). Classification datasets need a different approach.

**How do I label my images?**
Use CVAT, Label Studio, or Roboflow. Export in YOLO format. Or use `scripts/auto_label_dino.py` for zero-shot auto-labeling with text prompts.

**Can I use a larger model?**
Yes. Change `model: yolov8n.pt` to `yolov8s.pt`, `yolov8m.pt`, etc. Larger models are more accurate but slower. The `n` in yolov8n stands for "nano" (smallest).

**How many images do I need?**
Depends on the task. 100-200 images per class can work for simple objects. Complex scenes with many similar objects need 500+. More is always better.

**Can I skip the ST cloud stage?**
Yes. Set `stedge: enabled: false`. The pipeline stops at quantization. You still get `.pt`, `.onnx`, and `.tflite` files.

**What's the difference between `.pt`, `.onnx`, `.tflite`, and `.nb`?**
- `.pt` — PyTorch weights. Works on your PC with GPU.
- `.onnx` — Universal format. Works across frameworks and runtimes.
- `.tflite` — TensorFlow Lite. The INT8 quantized version runs on edge devices.
- `.nb` — ST's Neural Binary Graph. Specifically optimized for the STM32MP257F NPU.

**How long does training take?**
Depends on dataset size and GPU. On an RTX 4060: ~15 minutes for 2500 images at 50 epochs. ~2 minutes for 200 images.

**Can I run this on CPU?**
Yes. Set `device: cpu` in the config. It'll be 10-50x slower depending on your CPU.

**The pipeline crashed midway. What now?**
Run the same command again. It skips completed stages and picks up where it stopped.

**How do I add a new class to an existing dataset?**
Add images of the new class, label them with the next class ID, update `names` in the data.yaml, and retrain.

**Can I change the image size?**
Yes. Set `imgsz` in both `training` and `export` sections. Smaller = faster but less accurate. 320 is a good balance for edge. 640 is the YOLO default.

**My GPU has 4GB VRAM. Will that work?**
Yes, with `batch: 2` or `batch: 1` and `imgsz: 320`. Close other GPU apps during training.

---

## File structure reference

```
project/
├── run_pipeline.py                  # Entry point
├── main.py                          # Application entry (for runtime, not training)
├── board.yaml                       # Example config: board detection
├── computer_accessories.yaml        # Example config: accessories detection
├── my_dataset.yaml                  # Your config goes here
│
├── your_dataset/                    # Your dataset
│   ├── data.yaml                    # Ultralytics class definitions
│   ├── train/images/ + labels/
│   ├── val/images/ + labels/
│   └── test/images/ + labels/
│
├── artifacts_my_dataset/            # Pipeline outputs
│   ├── dataset_validation/
│   ├── training/weights/best.pt
│   ├── evaluation/
│   ├── export/best.onnx
│   ├── quantization/best_int8.tflite
│   ├── compilation/best.nb
│   └── reports/
│
├── pipeline/                        # The pipeline framework
│   ├── cli.py                       # CLI parser
│   ├── config/                      # Config loading and validation
│   ├── core/                        # Runner, stage base class, context
│   ├── stages/                      # All 11 pipeline stages
│   └── utils/                       # Logging, path helpers
│
├── scripts/                         # Standalone tools
│   ├── inference.py                 # Live camera inference
│   ├── train.py                     # Standalone training
│   ├── split_dataset.py             # Split flat data into train/val/test
│   ├── auto_label_dino.py           # Zero-shot auto-labeling
│   ├── augment_dataset.py           # Dataset augmentation
│   ├── capture.py                   # Camera image capture
│   ├── tflite_quant.py              # Standalone TFLite quantization
│   └── ...
│
└── src/                             # Runtime application code
    ├── ai/yolo_inference.py         # Unified inference backend
    ├── vision/                      # Camera, detection, perspective
    └── gui/                         # Tkinter GUI
```
