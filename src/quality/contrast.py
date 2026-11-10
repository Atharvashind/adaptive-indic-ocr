"""Grayscale and tile-based local contrast indicators."""

from __future__ import annotations

import numpy as np


def measure_contrast(
    grayscale: np.ndarray,
    low_standard_deviation: float = 30.0,
    tile_rows: int = 4,
    tile_columns: int = 4,
) -> dict[str, dict[str, object]]:
    """Return global grayscale standard deviation and mean local tile deviation."""
    if grayscale.ndim != 2 or grayscale.size == 0:
        raise ValueError("contrast measurement expects a non-empty grayscale image")
    if tile_rows <= 0 or tile_columns <= 0:
        raise ValueError("contrast tile rows and columns must be positive")
    global_std = float(np.std(grayscale.astype(np.float32)))
    height, width = grayscale.shape
    local_values: list[float] = []
    for row in range(tile_rows):
        y0, y1 = row * height // tile_rows, (row + 1) * height // tile_rows
        for column in range(tile_columns):
            x0, x1 = column * width // tile_columns, (column + 1) * width // tile_columns
            tile = grayscale[y0:y1, x0:x1]
            if tile.size:
                local_values.append(float(np.std(tile.astype(np.float32))))
    local_contrast = float(np.mean(local_values)) if local_values else 0.0
    interpretation = (
        "low global contrast indicator"
        if global_std < low_standard_deviation
        else "moderate-to-high global contrast indicator"
    )
    warning = "Contrast depends on page content and should not be read as OCR legibility."
    return {
        "grayscale_standard_deviation": {
            "value": global_std,
            "interpretation": interpretation,
            "confidence": 0.7,
            "warnings": [warning],
        },
        "local_contrast": {
            "value": local_contrast,
            "interpretation": "mean within-tile grayscale standard deviation",
            "confidence": 0.6,
            "warnings": [warning, "Tile boundaries and image layout affect this estimate."],
        },
    }
