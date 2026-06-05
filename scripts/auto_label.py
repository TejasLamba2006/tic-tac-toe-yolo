import cv2
import numpy as np
from pathlib import Path

# ==========================================================
# CONFIG
# ==========================================================

IMAGE_DIR = "dataset/images"
LABEL_DIR = "dataset/labels"
DEBUG_DIR = "dataset/debug"

Path(LABEL_DIR).mkdir(parents=True, exist_ok=True)
Path(DEBUG_DIR).mkdir(parents=True, exist_ok=True)

# YOLO class IDs
EMPTY = 0
RED = 1
YELLOW = 2

# ==========================================================
# COLOR DETECTION
# ==========================================================


def classify_cell(cell):
    hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)

    # RED MASK
    lower_red1 = np.array([0, 80, 50])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([160, 80, 50])
    upper_red2 = np.array([180, 255, 255])

    red_mask = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask |= cv2.inRange(hsv, lower_red2, upper_red2)

    # YELLOW MASK
    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([40, 255, 255])

    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    red_pixels = cv2.countNonZero(red_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)

    if red_pixels > 500:
        return RED

    if yellow_pixels > 500:
        return YELLOW

    return EMPTY


# ==========================================================
# BOARD DETECTION
# ==========================================================

def find_board(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    largest = None
    largest_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > largest_area:
            largest_area = area
            largest = cnt

    if largest is None:
        return None

    x, y, w, h = cv2.boundingRect(largest)

    return x, y, w, h


# ==========================================================
# YOLO LABEL WRITER
# ==========================================================

def yolo_box(class_id, cx, cy, w, h):
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


# ==========================================================
# PROCESS IMAGE
# ==========================================================

def process_image(image_path):
    img = cv2.imread(str(image_path))

    board = find_board(img)

    if board is None:
        print(f"Board not found: {image_path.name}")
        return

    x, y, w, h = board

    debug = img.copy()

    cell_w = w / 3
    cell_h = h / 3

    labels = []

    for row in range(3):
        for col in range(3):

            x1 = int(x + col * cell_w)
            y1 = int(y + row * cell_h)

            x2 = int(x + (col + 1) * cell_w)
            y2 = int(y + (row + 1) * cell_h)

            cell = img[y1:y2, x1:x2]

            cls = classify_cell(cell)

            # YOLO normalized coords
            cx = ((x1 + x2) / 2) / img.shape[1]
            cy = ((y1 + y2) / 2) / img.shape[0]

            bw = (x2 - x1) / img.shape[1]
            bh = (y2 - y1) / img.shape[0]

            labels.append(
                yolo_box(cls, cx, cy, bw, bh)
            )

            color = {
                EMPTY: (0, 0, 0),
                RED: (0, 0, 255),
                YELLOW: (0, 255, 255)
            }[cls]

            cv2.rectangle(
                debug,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

    label_path = Path(LABEL_DIR) / f"{image_path.stem}.txt"

    with open(label_path, "w") as f:
        f.write("\n".join(labels))

    cv2.imwrite(
        str(Path(DEBUG_DIR) / image_path.name),
        debug
    )

    print(f"Done: {image_path.name}")


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    }

    images = [
        p for p in Path(IMAGE_DIR).iterdir()
        if p.suffix.lower() in image_extensions
    ]

    print(f"Found {len(images)} images")

    for img in images:
        process_image(img)

    print("Finished")
