# STM32 Deployment Notes

This project keeps the export path simple on purpose:

```text
best.pt -> ONNX -> TFLite -> STM32MP257
```

The shipped ST examples in `source-files/` show the runtime side of that path:

- [ort-vsinpu-ep-example.py](../../source-files/bin/ort-vsinpu-ep-example/ort-vsinpu-ep-example.py#L1)
- [tflite-vx-delegate-example.py](../../source-files/bin/tflite-vx-delegate-example/tflite-vx-delegate-example.py#L1)
- [config_board_npu.sh](../../source-files/x-linux-ai/resources/config_board_npu.sh#L1)

The practical board workflow is:

1. Export the trained model on Windows.
2. Validate the exported model locally with the live camera app.
3. Copy the exported artifact to the STM32MP257 board.
4. Use the ST runtime example that matches the model format.

The board-side camera plumbing in the ST package already supports `libcamerasrc` and `v4l2src`, so the camera source is not the hard part. The model format is.

For MP25-class boards, `config_board_npu.sh` shows that the X-LINUX-AI stack prefers `.nb` when the NPU path is used. If you want the absolute fastest runtime later, the next step after TFLite/ONNX is ST Edge AI packaging into `.nb`.
