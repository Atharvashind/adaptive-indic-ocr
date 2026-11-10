"""Independent contrast-enhancement operations."""

from __future__ import annotations

import cv2
import numpy as np

from src.preprocessing.operation import ImageArray, PreprocessingOperation
from src.preprocessing.operations._image_utils import split_alpha


class ClaheOperation(PreprocessingOperation):
    """Apply CLAHE to grayscale data or the lightness plane of RGB data."""

    name = "clahe"
    defaults = {"clip_limit": 2.0, "tile_grid_size": (8, 8)}
    allowed_parameters = {"clip_limit", "tile_grid_size"}

    def validate_parameters(self, parameters: dict) -> None:
        clip_limit = float(parameters["clip_limit"])
        grid = parameters["tile_grid_size"]
        if not 0 < clip_limit <= 40:
            raise ValueError("clip_limit must be in the range (0, 40]")
        if (
            not isinstance(grid, (list, tuple))
            or len(grid) != 2
            or any(not isinstance(value, int) or not 1 <= value <= 64 for value in grid)
        ):
            raise ValueError("tile_grid_size must contain two integers from 1 to 64")
        parameters["clip_limit"] = clip_limit
        parameters["tile_grid_size"] = tuple(grid)

    def _apply(
        self, image: ImageArray, parameters: dict
    ) -> tuple[ImageArray, list[str], dict]:
        color, alpha = split_alpha(image)
        clip_limit = float(parameters["clip_limit"])
        grid = tuple(parameters["tile_grid_size"])
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
        if color.ndim == 2 or color.shape[2] == 1:
            plane = color.reshape(color.shape[:2])
            output = clahe.apply(plane)
            if color.ndim == 3:
                output = output[:, :, None]
        else:
            lab = cv2.cvtColor(color, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            output = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        if alpha is not None:
            output = np.dstack((output, alpha))
        return output, [], {}


class HistogramEqualizationOperation(PreprocessingOperation):
    """Apply global histogram equalization without changing colour chroma."""

    name = "histogram_equalization"
    defaults = {}
    allowed_parameters: set[str] = set()

    def _apply(
        self, image: ImageArray, parameters: dict
    ) -> tuple[ImageArray, list[str], dict]:
        color, alpha = split_alpha(image)
        if color.ndim == 2 or color.shape[2] == 1:
            plane = color.reshape(color.shape[:2])
            output = cv2.equalizeHist(plane)
            if color.ndim == 3:
                output = output[:, :, None]
        else:
            lab = cv2.cvtColor(color, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
            output = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        if alpha is not None:
            output = np.dstack((output, alpha))
        return output, [], {}
