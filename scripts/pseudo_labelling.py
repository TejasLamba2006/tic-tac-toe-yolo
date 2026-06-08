from pathlib import Path
from ultralytics import YOLO

# =========================
# CONFIG
# =========================

MODEL_PATH = "runs/detect/train-2/weights/best.pt"

IMAGE_DIR = "dataset4/images"

OUTPUT_DIR = "pseudo_labels1"

CONFIDENCE = 0.50

# =========================

model = YOLO(MODEL_PATH)

results = model.predict(
    source=IMAGE_DIR,
    conf=CONFIDENCE,
    save=True,
    save_txt=True,
    save_conf=False,
    project=OUTPUT_DIR,
    name="labels",
    exist_ok=True
)

print(f"Processed {len(results)} images")
print(f"Labels saved in:")
print(f"{OUTPUT_DIR}/labels/labels/")
