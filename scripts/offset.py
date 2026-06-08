import onnx

m = onnx.load("runs/detect/train-5/weights/best.onnx")

print(m.opset_import)
