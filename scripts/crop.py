import cv2
import numpy as np
from pathlib import Path

INPUT_DIR = "dataset/images"
OUTPUT_DIR = "dataset_cropped"

Path(OUTPUT_DIR).mkdir(exist_ok=True)

for image_path in Path(INPUT_DIR).glob("*.*"):

    img = cv2.imread(str(image_path))

    if img is None:
        continue

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Blue border range
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print(f"Skipped: {image_path.name}")
        continue

    largest = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest)

    pad = 10

    x = max(0, x - pad)
    y = max(0, y - pad)

    w = min(img.shape[1] - x, w + pad * 2)
    h = min(img.shape[0] - y, h + pad * 2)

    cropped = img[y:y+h, x:x+w]

    cv2.imwrite(
        str(Path(OUTPUT_DIR) / image_path.name),
        cropped
    )

    print(f"Cropped: {image_path.name}")

print("Done")
