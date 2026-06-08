from ultralytics import YOLO
import numpy as np

# Class IDs from data.yaml
CLASS_MAP = {
    0: "E",  # empty
    1: "R",  # red_ball
    2: "Y",  # yellow_ball
}


def detections_to_board(results):
    """
    Convert YOLO detections into a 3x3 tic tac toe board.
    """

    boxes = results[0].boxes

    detections = []

    for box in boxes:
        cls_id = int(box.cls.item())

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        detections.append({
            "class": CLASS_MAP[cls_id],
            "cx": cx,
            "cy": cy
        })

    # sort by rows first (top -> bottom)
    detections.sort(key=lambda d: d["cy"])

    rows = [
        detections[0:3],
        detections[3:6],
        detections[6:9]
    ]

    board = []

    for row in rows:
        row.sort(key=lambda d: d["cx"])
        board.append([cell["class"] for cell in row])

    return board


if __name__ == "__main__":
    model = YOLO("runs/detect/train/weights/best.pt")

    results = model.predict(
        source="test_images/c6bdd9d8-ball_20260529_141453_109869.jpg",
        conf=0.25,
        save=False
    )

    board = detections_to_board(results)

    print("\nBoard State:")
    for row in board:
        print(row)
