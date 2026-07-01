"""
Auto-label board images for YOLOv8 training using Grounding DINO (zero-shot detection).

WHAT THIS DOES
--------------
1. Runs a pretrained open-vocabulary detector (Grounding DINO) on every image
   in an input folder, using a text prompt describing the object (e.g. "circuit board").
2. Takes the highest-confidence detection box per image (assumes one board per image).
3. Converts the box to YOLO format: class_id x_center y_center width height (all normalized 0-1).
4. Writes one .txt label file per image into an output "labels" folder (YOLO dataset convention).
5. Saves a copy of each image with the box drawn on it into a "preview" folder, so you can
   quickly scroll through and sanity-check the auto-labels before training.

REQUIREMENTS (run on a machine with internet access + ideally a GPU)
----------------------------------------------------------------------
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # or cpu build
pip install transformers pillow opencv-python numpy

USAGE
-----
python auto_label.py \
    --images_dir ./dataset/stm32mp157f-dk/images \
    --output_dir ./dataset/stm32mp157f-dk \
    --class_id 0 \
    --class_name stm32mp157f-dk \
    --prompt "circuit board."

Repeat once per board, changing --images_dir, --class_id (0,1,2,3), --class_name, and
optionally the --prompt if a different phrase detects that board better
(e.g. "single board computer." or "electronic development board.").

After doing this for all 4 boards, merge everything into one dataset folder structure:

dataset/
  images/
    train/  (all images from all 4 boards, split ~80%)
    val/    (remaining ~20%)
  labels/
    train/
    val/

and write a data.yaml:

  path: ./dataset
  train: images/train
  val: images/val
  names:
    0: stm32mp157f-dk
    1: arduino_uno_q
    2: stm32mp257f-dk
    3: raspberry_pi_4

Then train with:
  yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

MODEL_ID = "IDEA-Research/grounding-dino-tiny"  # small + fast; use "-base" for higher accuracy


def load_model(device):
    print(f"Loading {MODEL_ID} on {device} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
    model.eval()
    return processor, model


def detect_best_box(image_pil, prompt, processor, model, device,
                     box_threshold=0.25, text_threshold=0.2):
    """Returns the single highest-confidence box as (x0, y0, x1, y1) in pixel coords, or None."""
    inputs = processor(images=image_pil, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    try:
        # newer transformers versions use `threshold` instead of `box_threshold`
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image_pil.size[::-1]],  # (height, width)
        )[0]
    except TypeError:
        # older transformers versions use `box_threshold`
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image_pil.size[::-1]],
        )[0]

    if len(results["boxes"]) == 0:
        return None, None

    # take the highest scoring box (assumes one board per image)
    best_idx = int(torch.argmax(results["scores"]))
    box = results["boxes"][best_idx].tolist()
    score = float(results["scores"][best_idx])
    return box, score


def xyxy_to_yolo(box, img_w, img_h):
    x0, y0, x1, y1 = box
    x0, x1 = max(0, x0), min(img_w, x1)
    y0, y1 = max(0, y0), min(img_h, y1)
    xc = (x0 + x1) / 2 / img_w
    yc = (y0 + y1) / 2 / img_h
    w = (x1 - x0) / img_w
    h = (y1 - y0) / img_h
    return xc, yc, w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="Folder containing input images")
    ap.add_argument("--output_dir", required=True, help="Where to write labels/ and preview/")
    ap.add_argument("--class_id", type=int, required=True, help="YOLO class id for this board")
    ap.add_argument("--class_name", required=True, help="Human-readable class name (for logging)")
    ap.add_argument("--prompt", default="circuit board.",
                     help='Text prompt, must end with a period, e.g. "circuit board."')
    ap.add_argument("--box_threshold", type=float, default=0.25)
    ap.add_argument("--text_threshold", type=float, default=0.2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_model(device)

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.output_dir) / "labels"
    preview_dir = Path(args.output_dir) / "preview"
    labels_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)
    print(f"Found {len(image_paths)} images in {images_dir}")

    missed = []
    for i, img_path in enumerate(image_paths, 1):
        image_pil = Image.open(img_path).convert("RGB")
        img_w, img_h = image_pil.size

        box, score = detect_best_box(
            image_pil, args.prompt, processor, model, device,
            args.box_threshold, args.text_threshold,
        )

        if box is None:
            print(f"[{i}/{len(image_paths)}] {img_path.name}: NO DETECTION (skipped)")
            missed.append(img_path.name)
            continue

        xc, yc, w, h = xyxy_to_yolo(box, img_w, img_h)
        label_path = labels_dir / (img_path.stem + ".txt")
        with open(label_path, "w") as f:
            f.write(f"{args.class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        # draw preview
        cv_img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        x0, y0, x1, y1 = [int(v) for v in box]
        cv2.rectangle(cv_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(cv_img, f"{args.class_name} {score:.2f}", (x0, max(0, y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(str(preview_dir / img_path.name), cv_img)

        print(f"[{i}/{len(image_paths)}] {img_path.name}: score={score:.2f} box={[round(v) for v in box]}")

    print("\nDone.")
    print(f"Labels written to: {labels_dir}")
    print(f"Preview images (with boxes drawn) written to: {preview_dir}")
    if missed:
        print(f"\n{len(missed)} image(s) had NO detection and need manual labeling:")
        for m in missed:
            print(f"  - {m}")


if __name__ == "__main__":
    main()