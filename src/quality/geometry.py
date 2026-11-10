"""Measure image dimensions and cautiously estimate dominant line skew."""

from __future__ import annotations

import cv2
import numpy as np


def measure_geometry(
    grayscale: np.ndarray,
    minimum_line_length_ratio: float = 0.1,
    minimum_lines: int = 5,
    maximum_angle_spread: float = 2.0,
) -> dict[str, object]:
    """Estimate skew only when several long, near-horizontal edges agree."""
    if grayscale.ndim != 2 or grayscale.size == 0:
        raise ValueError("geometry measurement expects a non-empty grayscale image")
    height, width = grayscale.shape
    aspect_ratio = float(width / height) if height else None
    warnings = [
        "Skew is an image-only estimate; non-text edges and scene layouts can dominate."
    ]
    angle: float | None = None
    confidence = 0.0
    if min(height, width) >= 16:
        edges = cv2.Canny(grayscale, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(10, int(min(height, width) * 0.05)),
            minLineLength=max(10, int(width * minimum_line_length_ratio)),
            maxLineGap=max(2, int(width * 0.01)),
        )
        angles: list[float] = []
        if lines is not None:
            line_groups = lines[:, 0, :] if lines.ndim == 3 else lines
            for line in line_groups:
                if len(line) < 4:
                    continue
                x1, y1, x2, y2 = (int(value) for value in line[:4])
                line_angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if abs(line_angle) <= 15:
                    angles.append(line_angle)
        if len(angles) >= minimum_lines:
            median_angle = float(np.median(angles))
            spread = float(np.median(np.abs(np.asarray(angles) - median_angle)))
            if spread <= maximum_angle_spread:
                angle = median_angle
                confidence = min(0.9, 0.4 + len(angles) / 30)
                interpretation = "dominant near-horizontal edge orientation estimate"
            else:
                interpretation = "unavailable; candidate line orientations disagree"
                warnings.append("Candidate line angles have high dispersion.")
        else:
            interpretation = "unavailable; insufficient consistent line evidence"
            warnings.append(
                "Text orientation could not be measured reliably from enough line segments."
            )
    else:
        interpretation = "unavailable; image is too small"
        warnings.append("Image is too small for a stable line-orientation estimate.")
    return {
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "skew_angle_degrees": {
            "value": angle,
            "interpretation": interpretation,
            "confidence": confidence,
            "warnings": warnings,
        },
    }
