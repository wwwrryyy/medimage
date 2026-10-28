"""
calibration_reader.py
=====================
ONNX Runtime CalibrationDataReader for INT8 Static Quantization.

Reads anonymized normal and pathological scans from calibration_data/ and yields
preprocessed numpy float32 tensors formatted for the target model stage.
"""

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import cv2
import numpy as np


class Int8CalibrationDataReader:
    """
    Feeds calibration tensors to onnxruntime.quantization.quantize_static().
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "calibration_data",
        input_name: str = "corrupted_frame",
        height: int = 256,
        width: int = 256,
        batch_size: int = 1,
    ):
        self.data_dir = Path(data_dir)
        self.input_name = input_name
        self.height = height
        self.width = width
        self.batch_size = batch_size

        self.image_paths: List[Path] = sorted(
            list(self.data_dir.glob("normal/*.png")) + list(self.data_dir.glob("pathology/*.png"))
        )
        self._iterator: Optional[Iterator[Dict[str, np.ndarray]]] = None

    def _generate_data(self) -> Iterator[Dict[str, np.ndarray]]:
        for path in self.image_paths:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            resized = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            norm = (resized.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, :, :]
            yield {self.input_name: norm}

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self._iterator is None:
            self._iterator = self._generate_data()
        try:
            return next(self._iterator)
        except StopIteration:
            return None

    def rewind(self) -> None:
        self._iterator = None

    def __len__(self) -> int:
        return len(self.image_paths)
