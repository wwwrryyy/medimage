# INT8 Calibration Dataset

Sample normal and pathological clinical OCT/ultrasound scans (anonymized per HIPAA Safe Harbor 45 CFR § 164.514(b)(2)) for INT8 post-training quantization calibration across all three pipeline stages.

## Directory Structure
- `normal/`: 25 de-identified normal tissue scans (`anon_norm_*.png`).
- `pathology/`: 25 de-identified pathological lesion scans (`anon_path_CNV_*.png`, `anon_path_DME_*.png`, `anon_path_DRUSEN_*.png`).
- `calibration_reader.py`: ONNX Runtime `CalibrationDataReader` implementation to feed calibration tensors into `onnxruntime.quantization.quantize_static`.

## Usage with ONNX Runtime Quantization

```python
from calibration_data.calibration_reader import Int8CalibrationDataReader
from onnxruntime.quantization import quantize_static, CalibrationMethod, QuantType

# Stage 1: ConvUNet Despeckler INT8 Quantization
reader = Int8CalibrationDataReader(
    data_dir="calibration_data",
    input_name="corrupted_frame",
    height=256,
    width=256,
)
quantize_static(
    model_input="exports/unet_despeckler.onnx",
    model_output="exports/unet_despeckler_int8.onnx",
    calibration_data_reader=reader,
    quant_format=QuantType.QInt8,
    calibrate_method=CalibrationMethod.MinMax,
)
```
