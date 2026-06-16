# Tic-Tac-Toe (detect.py)

This repo’s real-time preview + inference loop is driven by **`detect.py`**.

It:

1. Reads frames from a camera
2. Detects/warps the tic-tac-toe board to a canonical view
3. Runs YOLO inference on the warped board
4. Draws the 3×3 grid + YOLO detection boxes and shows it in an OpenCV window

---

## Architecture (what `detect.py` calls)

`detect.py` is a thin runner around the core pipeline:

- **Model**: `src/ai/yolo_inference.py::YoloInference`
- **Board geometry / warp**: `src/vision/board_detector.py::BoardDetector`
- **Board state estimator**: `src/vision/board_state.py::BoardStateEstimator`
- **Stability tracking**: `src/vision/stability.py::BoardGeometryTracker`
- **Warp + overlay**: `src/main.py::analyze_frame()` and `src/main.py::_build_board_inset()`

In code form (per frame):

```text
YoloInference  ->  analyze_frame(frame, detector, board_detector, board_estimator, ...)
                               |
                               v
                      _build_board_inset(analysis, inset_size)
                               |
                               v
                          cv2.imshow(...)
```

---

## How to run `detect.py`

### 1) PC / PyTorch (no NPU)

Use a `.pt` model and a camera index:

```bash
python detect.py --weights yolov8n.pt --camera 0
```

### 2) STM32MP257 NPU (example: `.nb`)

Use a `.nb` model and enable `--npu`:

```bash
python3 detect.py \
  --weights tictactoe_yolov8_quant_pc_uf_od_tictactoe_1.nb \
  --camera auto \
  --npu
```

---

## Command-line options (match `detect.py --help`)

- `--weights` : Model weights path (`.pt / .onnx / .tflite / .nb`)
  - default: `yolov8n.pt`
- `--camera` : Camera source (`index`, `/dev/videoX`, or `auto`)
  - default: `auto`
- `--width` : Capture width
  - default: `640`
- `--height` : Capture height
  - default: `480`
- `--fps` : Requested frame rate
  - default: `30`
- `--conf` : YOLO confidence threshold
  - default: `0.50`
- `--iou` : NMS IoU threshold
  - default: `0.45`
- `--image-size` : YOLO input resolution
  - default: `320`
- `--inset-size` : Size of the displayed board inset window (pixels)
  - default: `480`
- `--npu` : Enable STM32 NPU acceleration

---

## Controls

In the preview window:

- Press **Q** or **ESC** to quit

## Model Overview

| Property | Value |
|:---------|:------|
| Base model | YOLOv8n (nano) |
| Task | Object detection |
| Input size | 320 × 320 |
| Classes | 3 (`empty`, `red_ball`, `yellow_ball`) |
| Training epochs | 100 |
| Batch size | 4 |
| Confidence threshold | 0.70 |
| NMS IoU threshold | 0.45 |
| Quantization | INT8 per-tensor |
| Target hardware | STM32MP257 NPU |

---

## Classes

| ID | Label | Description |
|:---|:------|:------------|
| 0 | `empty` | Board cell with no piece |
| 1 | `red_ball` | Board cell occupied by a red piece |
| 2 | `yellow_ball` | Board cell occupied by a yellow piece |

Defined in `data.yaml`:

```yaml
names:
  0: empty
  1: red_ball
  2: yellow_ball
```

---

## Architecture

The model is **YOLOv8n** — the nano variant of Ultralytics YOLOv8. It takes a 320×320 RGB image as input and outputs bounding boxes with class scores for each detected piece on the board.

The `YoloInference` class in `src/ai/yolo_inference.py` is a backend-agnostic wrapper. It selects the execution backend at runtime based on the model file extension.

### Inference Backend Selection

```mermaid
graph TD
    subgraph "src/ai/yolo_inference.py"
        INIT["YoloInference.__init__"]
        TFL["_init_tflite()"]
        ONX["_init_onnx()"]
        NB["_init_nb()"]
        ULT["_init_ultralytics()"]
    end

    INIT -->|".tflite"| TFL
    INIT -->|".onnx"| ONX
    INIT -->|".nb"| NB
    INIT -->|".pt"| ULT

    subgraph "Hardware Acceleration"
        TFL -->|"libvx_delegate.so"| NPU["STM32 NPU"]
        ONX -->|"VSINPUExecutionProvider"| NPU
        NB -->|"stai_mpu_network"| NPU
    end
```

### Preprocessing Pipeline

All non-Ultralytics backends share the same preprocessing path:

1. Resize frame to model input size (`net_w × net_h`) using bilinear interpolation
2. Convert BGR → RGB
3. Normalize pixel values to `[0.0, 1.0]` by dividing by 255 (float32 models) or cast to the quantized dtype (INT8 models)
4. Transpose to NCHW if the model expects channel-first layout
5. Add batch dimension → shape `[1, C, H, W]` or `[1, H, W, C]`

Post-processing decodes the raw output tensor, applies a confidence threshold, then runs NMS with IoU threshold 0.45.

### NPU Memory Safety Flow

```mermaid
flowchart TD
    START["App Start"] --> MEM["_read_cma_free_kb()"]
    MEM -->|"less than 100MB"| WARN["Log Warning: CMA Low"]
    MEM -->|"greater than 100MB"| LOAD["Load NPU Model"]
    LOAD --> BACKEND{"Backend Check"}
    BACKEND -->|".nb / .tflite"| VX["VX Delegate / stai_mpu"]
    BACKEND -->|".onnx"| ORT["VSINPU Execution Provider"]
```

---

## Training

Training uses the Ultralytics YOLOv8 framework on a custom dataset of physical Tic-Tac-Toe board images.

```bash
python scripts/train.py
```

`scripts/train.py` runs:

```python
model = YOLO("yolov8n.pt")
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=320,
    batch=4,
    workers=0,
    format="onnx",
    opset=12,
)
```

Trained weights are saved to `runs/detect/train/weights/best.pt`.

---

## Quantization

The model is quantized to **INT8 per-tensor** using the ST Edge AI toolchain for deployment on the STM32MP257 NPU.

Configuration (`scripts/config_quant.yaml`):

| Parameter | Value |
|:----------|:------|
| Quantization type | `per_tensor` |
| Input type | `int8` |
| Output type | `int8` |
| Input shape | `[320, 320, 3]` |
| Rescaling scale | 255 |
| Rescaling offset | 0 |

```bash
python scripts/tflite_quant.py
```

Quantized models are saved to `quantized_models/`.

---

## Export Pipeline

```mermaid
flowchart LR
    PT["best.pt"] --> ONNX["best.onnx"]
    ONNX --> TFLite["best.tflite"]
    TFLite --> NB["best.nb (ST Edge AI)"]
    NB --> Board["STM32MP257 NPU"]
```

**Step 1 — Export to ONNX (opset 12):**

```bash
python src/stm32/export_to_onnx.py \
    --weights runs/detect/train/weights/best.pt \
    --output build/best.onnx
```

**Step 2 — Export to TFLite:**

```bash
python src/stm32/export_to_tflite.py \
    --source build/best.onnx \
    --output build/best.tflite
```

**Step 3 — Validate on STM32MP257:**

```bash
python3 /usr/local/x-linux-ai/bin/ort-vsinpu-ep-example/ort-vsinpu-ep-example.py build/best.onnx
python3 /usr/local/x-linux-ai/bin/tflite-vx-delegate-example/tflite-vx-delegate-example.py build/best.tflite
```

The `.nb` Network Binary format (produced by ST Edge AI packaging) is the optimal final format for the STM32MP257 NPU.

---

## Inference Backends

| Format | Backend | Target | Notes |
|:-------|:--------|:-------|:------|
| `.pt` | Ultralytics (PyTorch) | Windows / Linux CPU | Development |
| `.onnx` | ONNX Runtime + `VSINPUExecutionProvider` | STM32MP257 NPU | Handoff format |
| `.tflite` | TFLite Runtime + `libvx_delegate.so` | STM32MP257 NPU | Quantized INT8 |
| `.nb` | `stai_mpu_network` | STM32MP257 NPU | Optimal NPU path |

---

## Configuration Flags

| Flag | Description | Default |
|:-----|:------------|:--------|
| `--weights` | Path to `.pt`, `.onnx`, `.tflite`, or `.nb` file | `runs/detect/train/weights/best.pt` |
| `--npu` | Enable NPU acceleration (VX delegate / stai_mpu) | `False` |
| `--confidence-threshold` | Minimum YOLO confidence for detection | `0.70` |
| `--image-size` | Input image size for inference | `640` |
