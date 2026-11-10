"""Variance-of-Laplacian blur indicator."""

from __future__ import annotations

import cv2
import numpy as np


def measure_blur(
    grayscale: np.ndarray,
    blurred_below: float = 100.0,
    sharp_above: float = 300.0,
) -> dict[str, object]:
    """Measure Laplacian variance; thresholds are indicative, not universal."""
    if grayscale.ndim != 2 or grayscale.size == 0:
        raise ValueError("blur measurement expects a non-empty grayscale image")
    if min(grayscale.shape) < 3:
        return {
            "value": None,
            "interpretation": "unavailable",
            "confidence": 0.0,
            "warnings": ["Image is too small for a stable Laplacian measurement."],
        }
    value = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
    if value < blurred_below:
        interpretation = "low edge sharpness; possible blur"
    elif value >= sharp_above:
        interpretation = "high edge sharpness indicator"
    else:
        interpretation = "intermediate edge sharpness"
    return {
        "value": value,
        "interpretation": interpretation,
        "confidence": 0.55,
        "warnings": [
            "Laplacian variance depends on image scale, content, and writing texture."
        ],
    }
