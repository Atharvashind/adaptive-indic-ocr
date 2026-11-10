"""Read-only image quality analysis and explainable metric results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from src.quality.blur import measure_blur
from src.quality.contrast import measure_contrast
from src.quality.geometry import measure_geometry
from src.quality.illumination import measure_illumination
from src.quality.noise import measure_noise

logger = logging.getLogger(__name__)


class ImageReadError(OSError):
    """Raised when an image cannot be decoded for measurement."""


def load_grayscale(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode an image via Pillow and return an 8-bit grayscale array."""
    try:
        with Image.open(path) as image:
            image_format = image.format or path.suffix.lstrip(".").upper()
            source_mode = image.mode
            width, height = image.size
            rgb = np.asarray(image.convert("RGB"))
            file_size_bytes = path.stat().st_size
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise ImageReadError(f"Unable to read image {path}: {error}") from error
    if rgb.size == 0 or width <= 0 or height <= 0:
        raise ImageReadError(f"Image has invalid dimensions: {path}")
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return grayscale, {
        "format": image_format.upper(),
        "source_mode": source_mode,
        "width": int(width),
        "height": int(height),
        "color_category": (
            "grayscale"
            if source_mode in {"1", "L", "I", "F"}
            or (
                np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
                and np.array_equal(rgb[:, :, 1], rgb[:, :, 2])
            )
            else "colour"
        ),
        "file_size_bytes": file_size_bytes,
    }


class QualityAnalyzer:
    """Compute configured, non-destructive image quality indicators."""

    def __init__(self, thresholds: dict[str, Any] | None = None) -> None:
        self.thresholds = thresholds or {}

    def analyze(self, path: Path) -> dict[str, Any]:
        """Return dimensions, metadata, measurements, and relevant warnings."""
        grayscale, metadata = load_grayscale(path)
        return self._analyze_grayscale(grayscale, metadata)

    def analyze_array(
        self,
        image: np.ndarray,
        *,
        image_identifier: str,
        color_order: str = "RGB",
    ) -> dict[str, Any]:
        """Analyze a uint8 grayscale, RGB/BGR, or RGBA/BGRA array in memory."""
        if not isinstance(image, np.ndarray):
            raise ValueError("image must be a NumPy array")
        if image.dtype != np.uint8 or image.size == 0 or image.ndim not in {2, 3}:
            raise ValueError("image must be non-empty uint8 grayscale or colour data")
        if image.ndim == 3 and image.shape[2] not in {1, 3, 4}:
            raise ValueError("colour image must have 1, 3, or 4 channels")
        order = color_order.upper()
        if order not in {"RGB", "BGR"}:
            raise ValueError("color_order must be 'RGB' or 'BGR'")
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("image dimensions must be positive")

        if image.ndim == 2:
            grayscale = image
            color_category = "grayscale"
            source_mode = "L"
        elif image.shape[2] == 1:
            grayscale = image[:, :, 0]
            color_category = "grayscale"
            source_mode = "L"
        else:
            color = image[:, :, :3]
            conversion = cv2.COLOR_RGB2GRAY if order == "RGB" else cv2.COLOR_BGR2GRAY
            grayscale = cv2.cvtColor(color, conversion)
            color_category = (
                "grayscale"
                if np.array_equal(color[:, :, 0], color[:, :, 1])
                and np.array_equal(color[:, :, 1], color[:, :, 2])
                else "colour"
            )
            source_mode = order + ("A" if image.shape[2] == 4 else "")
        return self._analyze_grayscale(
            grayscale,
            {
                "format": None,
                "source_mode": source_mode,
                "width": int(width),
                "height": int(height),
                "color_category": color_category,
                "file_size_bytes": None,
                "image_identifier": image_identifier,
            },
        )

    def _analyze_grayscale(
        self, grayscale: np.ndarray, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply shared Phase 1 metrics to an already-decoded grayscale image."""
        blur_options = self.thresholds.get("blur", {})
        contrast_options = self.thresholds.get("contrast", {})
        illumination_options = self.thresholds.get("illumination", {})
        noise_options = self.thresholds.get("noise", {})
        geometry_options = self.thresholds.get("geometry", {})

        blur = measure_blur(
            grayscale,
            blurred_below=float(blur_options.get("blurred_below", 100.0)),
            sharp_above=float(blur_options.get("sharp_above", 300.0)),
        )
        contrast = measure_contrast(
            grayscale,
            low_standard_deviation=float(
                contrast_options.get("low_standard_deviation", 30.0)
            ),
            tile_rows=int(contrast_options.get("tile_rows", 4)),
            tile_columns=int(contrast_options.get("tile_columns", 4)),
        )
        brightness_mean = float(np.mean(grayscale, dtype=np.float64))
        brightness_std = float(np.std(grayscale.astype(np.float32)))
        histogram, _ = np.histogram(grayscale, bins=16, range=(0, 256))
        histogram_total = max(1, int(histogram.sum()))
        brightness = {
            "value": brightness_mean,
            "interpretation": "mean grayscale intensity on the 0-255 scale",
            "confidence": 1.0,
            "warnings": ["Brightness depends on source content and image encoding."],
            "standard_deviation": brightness_std,
            "distribution_16_bins": (histogram / histogram_total).tolist(),
        }
        illumination = measure_illumination(
            grayscale,
            uneven_above=float(
                illumination_options.get("uneven_above", 18.0)
            ),
            grid_size=int(illumination_options.get("grid_size", 4)),
        )
        noise = measure_noise(
            grayscale,
            elevated_above=float(noise_options.get("elevated_above", 10.0)),
        )
        geometry = measure_geometry(
            grayscale,
            minimum_line_length_ratio=float(
                geometry_options.get("minimum_line_length_ratio", 0.1)
            ),
            minimum_lines=int(geometry_options.get("minimum_lines", 5)),
            maximum_angle_spread=float(
                geometry_options.get("maximum_angle_spread", 2.0)
            ),
        )
        aspect_ratio = float(metadata["width"] / metadata["height"])
        measurements: dict[str, Any] = {
            "blur": blur,
            **contrast,
            "brightness": brightness,
            "illumination": illumination,
            "noise": noise,
            "skew_angle_degrees": geometry["skew_angle_degrees"],
        }
        warnings = [
            warning
            for measurement in measurements.values()
            for warning in measurement.get("warnings", [])
        ]
        result: dict[str, Any] = {
            **metadata,
            "aspect_ratio": aspect_ratio,
            "measurements": measurements,
            "warnings": warnings,
            "valid": True,
            "error": None,
        }
        return result

    def analyze_safely(self, path: Path) -> dict[str, Any]:
        """Return a structured unreadable result rather than hiding decode errors."""
        try:
            result = self.analyze(path)
            result["path"] = str(path)
            return result
        except ImageReadError as error:
            logger.warning("%s", error)
            file_size_bytes: int | None = None
            warnings = [str(error)]
            try:
                file_size_bytes = path.stat().st_size
            except OSError as stat_error:
                warnings.append(f"Could not read image file size: {stat_error}")
            return {
                "path": str(path),
                "valid": False,
                "error": str(error),
                "file_size_bytes": file_size_bytes,
                "warnings": warnings,
            }
