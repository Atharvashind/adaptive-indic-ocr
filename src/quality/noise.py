"""Robust high-frequency residual indicator with documented limitations."""

from __future__ import annotations

import cv2
import numpy as np


def measure_noise(
    grayscale: np.ndarray,
    elevated_above: float = 10.0,
) -> dict[str, object]:
    """Estimate residual scale using MAD after a 3x3 median filter.

    The value is not a calibrated sensor-noise estimate: strokes, texture, and
    compression artifacts remain in the high-frequency residual.
    """
    if grayscale.ndim != 2 or grayscale.size == 0:
        raise ValueError("noise measurement expects a non-empty grayscale image")
    if min(grayscale.shape) < 3:
        return {
            "value": None,
            "interpretation": "unavailable",
            "confidence": 0.0,
            "warnings": ["Image is too small for the 3x3 median residual estimator."],
        }
    residual = (
        grayscale.astype(np.float32)
        - cv2.medianBlur(grayscale, 3).astype(np.float32)
    )
    median = float(np.median(residual))
    value = float(np.median(np.abs(residual - median)) / 0.6745)
    return {
        "value": value,
        "interpretation": (
            "elevated high-frequency residual"
            if value >= elevated_above
            else "low-to-moderate high-frequency residual"
        ),
        "confidence": 0.25,
        "warnings": [
            "Estimator: median absolute deviation of grayscale minus a 3x3 median filter, "
            "scaled by 0.6745.",
            "Text strokes, handwriting, image texture, and compression artifacts can "
            "increase this value; it must not be interpreted as noise alone.",
        ],
    }
