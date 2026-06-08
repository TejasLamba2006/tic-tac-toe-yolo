import tensorflow as tf
import numpy as np

model_path = r"c:\Users\Tejas\Desktop\Source-Codes\Internships\STMicroelectronics\STM32M257f-dk-x-linux-ai\quantized_models\tictactoe_yolov8_quant_pt_uf_od_tictactoe.tflite"

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Print details of the last 10 tensors
details = interpreter.get_tensor_details()
print("Last 10 tensors:")
for d in details[-15:]:
    print(f"Tensor {d['index']}: name={d['name']}, shape={d['shape']}, dtype={d['dtype']}, quantization={d['quantization']}")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

dummy_input = np.zeros(input_details[0]['shape'], dtype=input_details[0]['dtype'])
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()

print("\nRunning dummy inference...")
out_float = interpreter.get_tensor(output_details[0]['index'])
print("Float Output 0 min/max:", out_float.min(), out_float.max())

# By searching for "PartitionedCall:01" or tensor index that is INT8 and has shape [1, 7, 2100]
for d in details:
    if d['name'] == 'PartitionedCall:01' or (list(d['shape']) == [1, 7, 2100] and d['dtype'] in (np.int8, np.uint8)):
        raw_tensor = interpreter.get_tensor(d['index'])
        print(f"\nFound raw quantized tensor index {d['index']}: name={d['name']}")
        print("Raw INT8 min/max:", raw_tensor.min(), raw_tensor.max())
        print("First 20 raw values of class 0 (row 4):", raw_tensor[0, 4, :20])
        print("First 20 raw values of class 1 (row 5):", raw_tensor[0, 5, :20])
        print("First 20 raw values of class 2 (row 6):", raw_tensor[0, 6, :20])
        break
