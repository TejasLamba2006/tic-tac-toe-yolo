from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = "runs/detect/train/weights/best.pt"

IMAGE_DIR = "dataset_cropped"

OUTPUT_LABELS = "pseudo_labels/labels"
OUTPUT_DEBUG = "pseudo_labels/debug"

CONF = 0.25

Path(OUTPUT_LABELS).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DEBUG).mkdir(parents=True, exist_ok=True)

# Your class mapping
CLASS_NAMES = {
    0: "empty",
    1: "red_ball",
    2: "yellow_ball"
}

CLASS_IDS = {
    "empty": 0,
    "red_ball": 1,
    "yellow_ball": 2
}


# =====================================================
# HSV YELLOW DETECTOR
# =====================================================

def contains_yellow(region):
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([15, 60, 60])
    upper_yellow = np.array([45, 255, 255])

    mask = cv2.inRange(
        hsv,
        lower_yellow,
        upper_yellow
    )

    yellow_pixels = cv2.countNonZero(mask)

    area = region.shape[0] * region.shape[1]

    ratio = yellow_pixels / max(area, 1)

    return ratio > 0.12


# =====================================================
# YOLO LABEL WRITER
# =====================================================

def save_yolo_label(filepath, detections):
    with open(filepath, "w") as f:
        for cls_id, x, y, w, h in detections:
            f.write(
                f"{cls_id} "
                f"{x:.6f} "
                f"{y:.6f} "
                f"{w:.6f} "
                f"{h:.6f}\n"
            )


# =====================================================
# MAIN
# =====================================================

model = YOLO(MODEL_PATH)

images = []

for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
    images.extend(Path(IMAGE_DIR).glob(ext))

print(f"\nFound {len(images)} images\n")

for image_path in images:

    img = cv2.imread(str(image_path))

    h_img, w_img = img.shape[:2]

    results = model.predict(
        source=img,
        conf=CONF,
        verbose=False
    )

    result = results[0]

    corrected = []

    debug = img.copy()

    for box in result.boxes:

        cls_id = int(box.cls.item())

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        class_name = CLASS_NAMES[cls_id]

        roi = img[
            max(0, y1):min(h_img, y2),
            max(0, x1):min(w_img, x2)
        ]

        # =========================================
        # FIX EMPTY -> YELLOW
        # =========================================

        if class_name == "empty":

            if roi.size > 0 and contains_yellow(roi):
                cls_id = CLASS_IDS["yellow_ball"]
                class_name = "yellow_ball"

        # =========================================

        cx = ((x1 + x2) / 2) / w_img
        cy = ((y1 + y2) / 2) / h_img
        bw = (x2 - x1) / w_img
        bh = (y2 - y1) / h_img

        corrected.append(
            (
                cls_id,
                cx,
                cy,
                bw,
                bh
            )
        )

        color = {
            0: (0, 0, 0),
            1: (0, 0, 255),
            2: (0, 255, 255)
        }[cls_id]

        cv2.rectangle(
            debug,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            debug,
            class_name,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    label_path = (
        Path(OUTPUT_LABELS)
        / f"{image_path.stem}.txt"
    )

    save_yolo_label(
        label_path,
        corrected
    )

    cv2.imwrite(
        str(
            Path(OUTPUT_DEBUG)
            / image_path.name
        ),
        debug
    )

    print(
        f"Processed: {image_path.name}"
    )

print("\nDone!")
print(f"Labels: {OUTPUT_LABELS}")
print(f"Debug:  {OUTPUT_DEBUG}")
