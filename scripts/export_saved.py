from ultralytics import YOLO

model = YOLO("runs/detect/train-5/weights/best.pt")

model.export(
    format="saved_model",
    imgsz=320
)
