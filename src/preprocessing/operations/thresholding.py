"""Independent global and local thresholding operations."""

from __future__ import annotations

import cv2
import numpy as np

from src.preprocessing.operation import ImageArray, PreprocessingOperation
from src.preprocessing.operations._image_utils import grayscale


class OtsuThresholdOperation(PreprocessingOperation):
    """Apply binary Otsu thresholding to a grayscale representation."""

    name = "otsu_threshold"
    defaults = {"threshold_type": "binary"}
    allowed_parameters = {"threshold_type"}

    def validate_parameters(self, parameters: dict) -> None:
        if parameters["threshold_type"] not in {"binary", "binary_inverse"}:
            raise ValueError("threshold_type must be 'binary' or 'binary_inverse'")

    def _apply(
        self, image: ImageArray, parameters: dict
    ) -> tuple[ImageArray, list[str], dict]:
        mode = (
            cv2.THRESH_BINARY
            if parameters["threshold_type"] == "binary"
            else cv2.THRESH_BINARY_INV
        )
        threshold, output = cv2.threshold(
            grayscale(image), 0, 255, mode | cv2.THRESH_OTSU
        )
        return output, [], {"otsu_threshold": float(threshold)}


class _AdaptiveThresholdOperation(PreprocessingOperation):
    method: str
    name: str
    defaults = {"max_value": 255, "block_size": 11, "constant": 2.0}
    allowed_parameters = {"max_value", "block_size", "constant"}

    def validate_parameters(self, parameters: dict) -> None:
        maximum = parameters["max_value"]
        if not isinstance(maximum, int) or not 1 <= maximum <= 255:
            raise ValueError("max_value must be an integer from 1 to 255")
        block_size = parameters["block_size"]
        if not isinstance(block_size, int) or not 3 <= block_size <= 255 or block_size % 2 == 0:
            raise ValueError("block_size must be an odd integer from 3 to 255")
        constant = float(parameters["constant"])
        if not -255 <= constant <= 255:
            raise ValueError("constant must be in the range [-255, 255]")
        parameters["constant"] = constant

    def _apply(
        self, image: ImageArray, parameters: dict
    ) -> tuple[ImageArray, list[str], dict]:
        gray = grayscale(image)
        height, width = gray.shape
        effective = min(int(parameters["block_size"]), min(height, width))
        if effective % 2 == 0:
            effective -= 1
        if effective < 3:
            raise ValueError(
                "adaptive thresholding requires both image dimensions to be at least 3"
            )
        warnings = []
        if effective != parameters["block_size"]:
            warnings.append(
                f"block_size reduced from {parameters['block_size']} to {effective} "
                "to fit the image dimensions"
            )
        requested_block_size = int(parameters["block_size"])
        parameters["block_size"] = effective
        method = (
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            if self.method == "gaussian"
            else cv2.ADAPTIVE_THRESH_MEAN_C
        )
        output = cv2.adaptiveThreshold(
            gray,
            int(parameters["max_value"]),
            method,
            cv2.THRESH_BINARY,
            effective,
            float(parameters["constant"]),
        )
        return output, warnings, {
            "requested_block_size": requested_block_size,
            "effective_block_size": effective,
        }


class AdaptiveGaussianThresholdOperation(_AdaptiveThresholdOperation):
    """Apply adaptive Gaussian thresholding."""

    name = "adaptive_gaussian_threshold"
    method = "gaussian"


class AdaptiveMeanThresholdOperation(_AdaptiveThresholdOperation):
    """Apply adaptive mean thresholding."""

    name = "adaptive_mean_threshold"
    method = "mean"
