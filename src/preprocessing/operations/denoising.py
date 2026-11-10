"""Independent, configurable denoising operations."""

from __future__ import annotations

import cv2
import numpy as np

from src.preprocessing.operation import ImageArray, PreprocessingOperation
from src.preprocessing.operations._image_utils import apply_to_color_planes, split_alpha


def _odd_kernel(value: object, name: str, maximum: int) -> int:
    if not isinstance(value, int) or value < 1 or value > maximum or value % 2 == 0:
        raise ValueError(f"{name} must be an odd integer from 1 to {maximum}")
    return value


class GaussianFilterOperation(PreprocessingOperation):
    """Apply Gaussian smoothing with independently configurable axis sigmas."""

    name = "gaussian_filter"
    defaults = {"kernel_size": 5, "sigma_x": 0.0, "sigma_y": 0.0}
    allowed_parameters = {"kernel_size", "sigma_x", "sigma_y"}

    def validate_parameters(self, parameters: dict) -> None:
        _odd_kernel(parameters["kernel_size"], "kernel_size", 101)
        for key in ("sigma_x", "sigma_y"):
            value = float(parameters[key])
            if not 0 <= value <= 100:
                raise ValueError(f"{key} must be in the range [0, 100]")
            parameters[key] = value

    def _apply(
        self, image: ImageArray, parameters: dict
    ) -> tuple[ImageArray, list[str], dict]:
        kernel = int(parameters["kernel_size"])
        output = apply_to_color_planes(
            image,
            lambda plane: cv2.GaussianBlur(
                plane,
                (kernel, kernel),
                sigmaX=float(parameters["sigma_x"]),
                sigmaY=float(parameters["sigma_y"]),
            ),
        )
        return output, [], {}


class MedianFilterOperation(PreprocessingOperation):
    """Apply median filtering, reducing the kernel for very small inputs."""

    name = "median_filter"
    defaults = {"kernel_size": 3}
    allowed_parameters = {"kernel_size"}

    def validate_parameters(self, parameters: dict) -> None:
        _odd_kernel(parameters["kernel_size"], "kernel_size", 31)

    def _apply(
        self, image: ImageArray, parameters: dict
    ) -> tuple[ImageArray, list[str], dict]:
        requested = int(parameters["kernel_size"])
        height, width = image.shape[:2]
        maximum_odd = min(height, width)
        if maximum_odd % 2 == 0:
            maximum_odd -= 1
        effective = min(requested, max(1, maximum_odd))
        warnings = []
        if effective != requested:
            warnings.append(
                f"kernel_size reduced from {requested} to {effective} for this image size"
            )
        parameters["kernel_size"] = effective
        output = apply_to_color_planes(
            image, lambda plane: cv2.medianBlur(plane, effective)
        )
        return output, warnings, {
            "requested_kernel_size": requested,
            "effective_kernel_size": effective,
        }


class BilateralFilterOperation(PreprocessingOperation):
    """Apply edge-preserving bilateral smoothing to grayscale or RGB pixels."""

    name = "bilateral_filter"
    defaults = {"diameter": 5, "sigma_color": 50.0, "sigma_space": 50.0}
    allowed_parameters = {"diameter", "sigma_color", "sigma_space"}

    def validate_parameters(self, parameters: dict) -> None:
        diameter = parameters["diameter"]
        if not isinstance(diameter, int) or not 1 <= diameter <= 31:
            raise ValueError("diameter must be an integer from 1 to 31")
        for key in ("sigma_color", "sigma_space"):
            value = float(parameters[key])
            if not 0 < value <= 500:
                raise ValueError(f"{key} must be in the range (0, 500]")
            parameters[key] = value

    def _apply(
        self, image: ImageArray, parameters: dict
    ) -> tuple[ImageArray, list[str], dict]:
        diameter = int(parameters["diameter"])
        sigma_color = float(parameters["sigma_color"])
        sigma_space = float(parameters["sigma_space"])
        output = apply_to_color_planes(
            image,
            lambda plane: cv2.bilateralFilter(
                plane, diameter, sigma_color, sigma_space
            ),
        )
        return output, [], {}
