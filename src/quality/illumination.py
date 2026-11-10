"""Spatial illumination variation indicator."""

from __future__ import annotations

import cv2
import numpy as np


def measure_illumination(
    grayscale: np.ndarray,
    uneven_above: float = 18.0,
    grid_size: int = 4,
) -> dict[str, object]:
    """Measure standard deviation of coarse spatial mean brightness."""
    if grayscale.ndim != 2 or grayscale.size == 0:
        raise ValueError("illumination measurement expects a non-empty grayscale image")
    if grid_size <= 0:
        raise ValueError("illumination grid_size must be positive")
    height, width = grayscale.shape
    small_width, small_height = max(1, width // grid_size), max(1, height // grid_size)
    coarse = cv2.resize(
        grayscale.astype(np.float32),
        (small_width, small_height),
        interpolation=cv2.INTER_AREA,
    )
    value = float(np.std(coarse))
    return {
        "value": value,
        "interpretation": (
            "possible uneven illumination"
            if value >= uneven_above
            else "no strong coarse illumination variation detected"
        ),
        "confidence": 0.5,
        "warnings": [
            "Spatial brightness variation also reflects page content, shadows, and background."
        ],
    }
