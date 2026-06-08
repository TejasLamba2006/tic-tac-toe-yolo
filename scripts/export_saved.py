from ultralytics import YOLO

model = YOLO("runs/detect/train-5/weights/best.pt")
model.export(
    format="tflite",
    imgsz=320,
    int8=True,        
    data="data.yaml",   
)
