# PIPELINE.md

A practical guide to the ML pipeline — from raw images to a model running on an STM32MP257F board.

---

## Table of contents

1. [What this pipeline does](#what-this-pipeline-does)
2. [What you need before starting](#what-you-need-before-starting)
3. [Install dependencies](#install-dependencies)
4. [The two datasets](#the-two-datasets)
5. [Running the pipeline](#running-the-pipeline)
6. [What each stage does](#what-each-stage-does)
7. [The scripts directory](#the-scripts-directory)
8. [Running inference](#running-inference)
9. [Configuration files explained](#configuration-files-explained)
10. [Troubleshooting](#troubleshooting)
11. [FAQ](#faq)

---

## What this pipeline does

You give it a folder of images with bounding-box labels. It trains a YOLOv8n model, exports it to ONNX, quantizes it to INT8 TFLite, and optionally compiles it for the STM32MP257F NPU via ST's cloud. At the end you get a `.pt` file for your PC, an `.onnx` file, a `.tflite` file, and (if you run the cloud stage) a `.nb` file that runs on the board.

The pipeline handles all of this in one command. You point it at a config YAML and it runs every stage in order, skipping anything that's already done.

---

## What you need before starting

**Hardware:**
- A PC with an NVIDIA GPU (RTX series works well). The pipeline trains on GPU.
- Optionally: an STM32MP257F-DK board for on-device deployment.

**Software:**
- Python 3.11 (3.11 is what we tested with; 3.12 works for TensorFlow but we had issues)
- Git
- CUDA 12.8+ (for GPU training — the install instructions below handle this)

**No prior ML experience needed.** If you can open a terminal and run commands, you can use this pipeline.

---

## Install dependencies

Open a terminal and run these in order:

```bash
# Clone the repo (if you haven't already)
git clone <repo-url>
cd STM32M257f-dk-x-linux-ai

# Create a virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate

# Install PyTorch with CUDA support (for GPU training)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install everything else
pip install ultralytics opencv-python numpy pillow \
    onnx onnxruntime onnx2tf onnx-graphsurgeon sng4onnx \
    ai-edge-litert tensorflow tf_keras \
    pyyaml kaggle
```

If you have an RTX 4060 or similar, this gives you GPU-accelerated training. You can verify CUDA works:

```python
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

---

## The two datasets

This project works with two separate datasets. They use different class lists, so they have separate configs.

### 1. Board detection (4 classes)

Detects four development boards: Arduino Uno Q, STM32MP257F-DK, Raspberry Pi Model B, STM32MP157F-DK.

**Dataset location:** `dataset_boards/`

**Structure:**
```
dataset_boards/
  train/
    images/     (.jpg files)
    labels/     (.txt files, one per image)
  val/
    images/
    labels/
  test/
    images/
    labels/
```

**Label format (YOLO):** Each `.txt` file contains zero or more lines, one per object:
```
<class_id> <center_x> <center_y> <width> <height>
```
All values are normalized to 0-1 relative to image dimensions. For example:
```
0 0.5 0.5 0.3 0.4
```
means class 0 (Arduino Uno Q), centered at 50% x 50% y, 30% wide, 40% tall.

**How we built it:**
- Captured images with a camera, then used Grounding DINO (zero-shot object detection) to auto-label them. The script is `scripts/auto_label_dino.py`.
- Manually corrected some labels.
- Split into train/val/test with an 80/10/10 ratio using `scripts/split_dataset.py`.

**Class mapping:**
| ID | Class |
|----|-------|
| 0 | arduino_uno_q |
| 1 | stm32mp257f_dk |
| 2 | raspberry_pi_model_b |
| 3 | stm32mp157f_dk |

**Class distribution (223 images):**
- Class 0 (Arduino): 66 boxes
- Class 1 (STM32MP257F): 56 boxes
- Class 2 (Raspberry Pi): 71 boxes
- Class 3 (STM32MP157F): 30 boxes

After augmentation, the dataset grew to 1561 images.

### 2. Computer accessories (10 classes)

Detects office/lab equipment: Ac, Chair, Cpu, Digital board, Fire Extinguisher, Keyboard, Light, Monitor, Mouse, Projector.

**Dataset location:** `dataset_accessories/`

**Source:** LabEquipVis dataset, downloaded from Kaggle (`kaggle datasets download -d tmoore/lab-equipvis`). The dataset comes pre-split by the authors, so no manual splitting is needed.

**Structure:** Same YOLO layout as boards, but the split directories are named `train/`, `valid/`, `test/` (not `val/`).

```
dataset_accessories/
  data.yaml          (tells YOLO where images are)
  train/
    images/          (1809 images)
    labels/
  valid/
    images/          (516 images)
    labels/
  test/
    images/          (259 images)
    labels/
```

**Class mapping:**
| ID | Class |
|----|-------|
| 0 | Ac |
| 1 | Chair |
| 2 | Cpu |
| 3 | Digital board |
| 4 | Fire Extinguisher |
| 5 | Keyboard |
| 6 | Light |
| 7 | Monitor |
| 8 | Mouse |
| 9 | Projector |

**Class distribution (train split, 2584 total images):**
- Monitor: 13,382 boxes (dominant class)
- Chair: 6,768
- Mouse: 5,695
- Cpu: 2,850
- Keyboard: 2,506
- Light: 863
- Fire Extinguisher: 522
- Digital board: 242
- Ac: 215
- Projector: 153

**Important:** This dataset already includes 3x augmented copies from the authors. The pipeline's augmentation stage is disabled for this dataset to avoid double-augmenting.

**The `data.yaml` fix:** The original LabEquipVis `data.yaml` had paths like `../train/images` (designed for a different folder layout). We fixed them to `./train/images` so the pipeline finds them correctly.

---

## Running the pipeline

### Boards pipeline

```bash
# Full pipeline
python run_pipeline.py --config board.yaml

# Just training
python run_pipeline.py --config board.yaml --stage training

# Resume from export (skip everything before it)
python run_pipeline.py --config board.yaml --from-stage export_onnx

# Dry run (validate config, don't actually run)
python run_pipeline.py --config board.yaml --dry-run

# List all stages
python run_pipeline.py --config board.yaml --list-stages
```

### Accessories pipeline

```bash
# Full pipeline
python run_pipeline.py --config computer_accessories.yaml

# Single stage
python run_pipeline.py --config computer_accessories.yaml --stage training
```

### What you'll see

The pipeline prints colored output to your terminal:
- **Cyan bold** "Starting training" — stage is running
- **Yellow** "Skipping: ..." — stage was skipped (already done)
- **Green** "success (43.1s)" — stage finished
- **Red** — something went wrong

Each stage writes its results to `artifacts/<stage_name>/` (for boards) or `artifacts_accessories/<stage_name>/` (for accessories).

### Resume behavior

If training already ran and `best.pt` exists, the pipeline skips training automatically. Delete the `artifacts*/training/` folder to retrain from scratch.

---

## What each stage does

| # | Stage | What happens | Output |
|---|-------|-------------|--------|
| 1 | `dataset_validation` | Scans every image and label file. Checks that images open, labels exist, bounding boxes are valid YOLO format (0-1 range). Writes a report. | `validation_report.json` |
| 2 | `augmentation` | Applies brightness, contrast, blur, and noise transforms. Copies labels alongside. Disabled in the accessories config. | Augmented images in `augmented/` |
| 3 | `training` | Trains YOLOv8n from a pretrained checkpoint. Runs on GPU. Saves `best.pt` and `last.pt`. | `weights/best.pt`, `training_metrics.json` |
| 4 | `evaluation` | Runs the trained model on the test split. Computes mAP, precision, recall. Generates confusion matrix plots. | `evaluation_report.json`, confusion matrix PNGs |
| 5 | `export_onnx` | Exports `best.pt` to ONNX format (opset 12). | `best.onnx` |
| 6 | `quantization` | Converts ONNX to TFLite INT8 via onnx2tf. Uses a representative dataset for calibration. | `best_float32.tflite`, `best_int8.tflite` |
| 7 | `stedge_compile` | Uploads the TFLite model to ST Edge AI Developer Cloud. Downloads the optimized `.nb` file for the NPU. Requires `stm32ai_dc` package and ST account credentials. | `best.nb` |
| 8 | `deployment` | SCPs the model to the board over SSH. | Model file on the remote board |
| 9 | `verification` | Runs a quick inference on the board to confirm the model loads. | Pass/fail result |
| 10 | `benchmark` | Measures inference latency, FPS, and memory usage on the board. | `benchmark_results.json` |
| 11 | `report` | Aggregates everything into JSON, HTML, and Markdown reports. | Reports in `reports/` |

---

## The scripts directory

`scripts/` contains standalone utilities. You can run any of them independently — they don't depend on the pipeline framework.

### Key scripts

**`scripts/inference.py`** — Interactive inference. See the [Running inference](#running-inference) section below.

**`scripts/train.py`** — Standalone training script. Loads a config YAML and runs Ultralytics training. Useful if you want to tweak training params without the full pipeline.

**`scripts/augment_dataset.py`** — Runs augmentations on a flat dataset. The pipeline has its own augmentation stage, but this script works independently.

**`scripts/split_dataset.py`** — Splits a flat `images/`+`labels/` dataset into `train/`+`val/`+`test/` folders. We used this for the boards dataset.

**`scripts/auto_label_dino.py`** — Auto-labels images using Grounding DINO (zero-shot detection). You give it a text prompt like "Arduino Uno board" and it finds and boxes matching objects. This is how we labeled the boards dataset without manually drawing boxes.

**`scripts/capture.py`** — Captures images from a camera and saves them to a folder. Used to build the training dataset.

**`scripts/tflite_quant.py`** — Standalone quantization. Converts a YOLO model to TFLite INT8. The pipeline's quantization stage wraps similar logic.

**`scripts/download_lab_equipvis.py`** — Attempts to download the LabEquipVis dataset from Mendeley. We ended up downloading manually from Kaggle instead (the Mendeley API redirected oddly).

**`scripts/diagnose_yolo_tflite.py`** — Debug tool for TFLite models. Checks tensor shapes, quantization parameters, and runs a test inference.

---

## Running inference

Once you have a trained model, use `scripts/inference.py` to test it on a live camera feed.

### Basic usage

```bash
# Interactive mode — asks you to pick a camera and model
python scripts/inference.py

# Specify model and camera directly
python scripts/inference.py --model artifacts/training/weights/best.pt --camera 0

# Use an ONNX model
python scripts/inference.py --model artifacts/export/best.onnx --camera 0

# Use NPU acceleration (on STM32 board with NPU support)
python scripts/inference.py --model path/to/model.nb --camera 0 --npu

# Adjust confidence threshold
python scripts/inference.py --model best.pt --camera 0 --conf 0.5
```

### What it does

1. Probes cameras 0-9 and lets you pick one that works.
2. Scans `artifacts/` for `.pt`, `.onnx`, `.tflite`, and `.nb` model files.
3. Loads the model (dispatches to the right backend automatically).
4. Opens the camera feed and runs detection in a loop.
5. Draws bounding boxes with class labels and confidence scores.
6. Shows an FPS counter overlay.

Press `Q` or `ESC` to quit.

### Supported model formats

| Format | Backend | Notes |
|--------|---------|-------|
| `.pt` | Ultralytics | Works on PC with GPU |
| `.onnx` | onnxruntime | Works on PC. Use `--npu` for VSINPU on STM32 |
| `.tflite` | tflite_runtime | Use `--npu` for libvx_delegate on STM32 |
| `.nb` | stai_mpu_network | Optimal NPU path for STM32 |

---

## Configuration files explained

The pipeline uses YAML config files. Two exist in the repo:

| Config | Dataset | Classes | Use case |
|--------|---------|---------|----------|
| `board.yaml` | `dataset_boards/` | 4 board types | Development board detection |
| `computer_accessories.yaml` | `dataset_accessories/` | 10 office items | Computer accessories detection |

### Key sections

**`artifacts_dir`** — Where all pipeline outputs go. Change this to keep runs separate.

**`dataset`** — Points to your data. The `path` field is relative to the project root. Split subdirectories follow. Set `split_enabled: true` if your data is in a flat folder and needs splitting.

**`augmentation`** — Controls brightness, contrast, blur, noise transforms. Set `enabled: false` if your dataset already has augmentations (like the accessories dataset).

**`training`** — YOLO hyperparameters. `data_yaml` is the critical path: it points to the Ultralytics-format YAML that defines class names and image paths. This is relative to the project root.

**`export`** — ONNX export settings. Opset 12 works broadly.

**`quantization`** — TFLite INT8 quantization. Uses per-channel quantization with uint8 input and float output.

**`stedge`** — ST Edge AI cloud compilation. Needs your ST account credentials. Set `enabled: false` to skip this stage.

**`deployment`** — SSH-based deployment to the board. Needs the board's IP, SSH key or password.

### Passing a custom config

```bash
python run_pipeline.py --config path/to/your_config.yaml
```

---

## Troubleshooting

### "CUDA out of memory"
Reduce `batch` in the config (try 2 or 1). Close other GPU-using apps.

### "Data YAML not found"
Check the `data_yaml` path in your config. It's relative to the project root, not the config file. For the accessories dataset it should be `dataset_accessories/data.yaml`.

### Training is slow
Make sure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`. If it says `False`, reinstall PyTorch with the CUDA index URL shown in the install section.

### "Label file is empty"
Some images in the LabEquipVis dataset have empty label files (no objects annotated). The pipeline warns about these but continues fine.

### Quantization fails with onnx2tf
Make sure `onnx2tf` and `onnx-graphsurgeon` are installed. Sometimes the ONNX model needs simplification first — set `simplify: true` in the export config.

### stedge_compile fails
- Check credentials: `echo $STEDGE_USERNAME` and `echo $STEDGE_PASSWORD`
- The `stm32ai_dc` package must be installed. It's not on PyPI — it comes from ST's GitHub repo.
- Cloud operations can be slow (5-10 minutes). The default timeout is 600 seconds.

### Pipeline skips a stage
The pipeline skips stages whose output already exists. Delete the stage's folder in `artifacts*/` to force a re-run. For example, to retrain: `rm -rf artifacts_accessories/training`.

---

## FAQ

**Do I need to know ML to use this?**
No. Run the install commands, put your images in the right folder, and run the pipeline command. The configs are already set up.

**What if I have my own dataset?**
Put images in `images/` and labels in `labels/` under a dataset folder. Labels must be YOLO format (class_id cx cy w h, normalized 0-1). Set `split_enabled: true` in the config if you don't have train/val/test splits yet. Update the class names in the YAML.

**How do I label my own images?**
Use [Label Studio](https://labelstud.io/) or [CVAT](https://cvat.ai/). Export in YOLO format. Or use `scripts/auto_label_dino.py` for zero-shot auto-labeling.

**Can I skip the ST cloud stage?**
Yes. Set `stedge: enabled: false` in the config. The pipeline will stop at quantization.

**What's the difference between `.pt`, `.onnx`, and `.tflite`?**
- `.pt` — PyTorch weights. Works on your PC with GPU.
- `.onnx` — Universal format. Works across frameworks. Good for deployment.
- `.tflite` — TensorFlow Lite. INT8 quantized version runs on the STM32 NPU.
- `.nb` — ST's Neural Binary Graph. Optimized for the STM32MP257F NPU specifically.

**How long does training take?**
On an RTX 4060 laptop: ~15 minutes for 50 epochs on the accessories dataset (2584 images). The boards dataset (223 images) trains in about 2 minutes.

**Can I run this on CPU?**
Yes, but it'll be slow. Remove `device: 0` from the config or set it to `device: cpu`.

**What Python version do I need?**
Python 3.11 is recommended. Python 3.12 works but has some TensorFlow compatibility quirks.

**Do I need an NVIDIA GPU?**
For training, yes — or at least a lot of patience. The quantization and export stages work on CPU. Inference works on CPU too.

**The pipeline crashed midway. What now?**
Just run the same command again. The pipeline skips stages that already completed. It picks up where it left off.

**How do I add a new class?**
Add images of the new class, label them with the new class ID, update `names` in the data YAML, and retrain. The number of classes is determined by the data YAML, not the config.

**Can I use a different YOLO model (not YOLOv8n)?**
Yes. Change `model: yolov8n.pt` to `yolov8s.pt`, `yolov8m.pt`, etc. in the training config. Larger models are more accurate but slower.

---

## File structure quick reference

```
STM32M257f-dk-x-linux-ai/
├── run_pipeline.py              # Entry point — run this
├── main.py                      # Application entry (for runtime, not training)
├── board.yaml                   # Config for board detection
├── computer_accessories.yaml    # Config for accessories detection
├── data.yaml                    # Ultrainties YAML for boards (4 classes)
├── yolov8n.pt                   # Pretrained YOLOv8n weights
│
├── dataset_boards/              # Board detection dataset
├── dataset_accessories/         # Accessories dataset (LabEquipVis)
│   └── data.yaml               # Ultralytics YAML for accessories (10 classes)
│
├── artifacts/                   # Pipeline outputs for boards
├── artifacts_accessories/       # Pipeline outputs for accessories
│
├── pipeline/                    # The pipeline framework
│   ├── cli.py                   # CLI parser (--config, --stage, etc.)
│   ├── config/                  # Config loading and validation
│   ├── core/                    # Runner, stage base class, context
│   ├── stages/                  # All 11 pipeline stages
│   └── utils/                   # Logging, path helpers
│
├── scripts/                     # Standalone utilities
│   ├── inference.py             # Live camera inference
│   ├── train.py                 # Standalone training
│   ├── split_dataset.py         # Split flat dataset into train/val/test
│   ├── auto_label_dino.py       # Zero-shot auto-labeling with DINO
│   ├── augment_dataset.py       # Dataset augmentation
│   ├── capture.py               # Camera image capture
│   ├── tflite_quant.py          # Standalone TFLite quantization
│   └── ...
│
└── src/                         # Runtime application code
    ├── main.py                  # App logic (board detection + game)
    ├── ai/yolo_inference.py     # Unified inference backend
    ├── vision/                  # Camera, board detection, perspective
    └── gui/                     # Tkinter GUI
```
