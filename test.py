import tflite_runtime.interpreter as tflite
import numpy as np
import cv2

interp = tflite.Interpreter('runs/detect/train-5/weights/best_saved_model/best_int8.tflite')
interp.allocate_tensors()
print('Input:', interp.get_input_details()[0])
print('Output:', interp.get_output_details()[0])
"
