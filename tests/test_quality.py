"""Tests for dimension, blur, and contrast indicators."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.quality.blur import measure_blur
from src.quality.contrast import measure_contrast
from src.quality.geometry import measure_geometry
from src.quality.quality_analyzer import QualityAnalyzer


def test_quality_analyzer_calculates_dimensions_and_aspect_ratio(tmp_path: Path) -> None:
    path = tmp_path / "gray.png"
    Image.new("L", (20, 10), color=128).save(path)

    result = QualityAnalyzer().analyze(path)

    assert result["valid"] is True
    assert result["error"] is None
    assert result["width"] == 20
    assert result["height"] == 10
    assert result["aspect_ratio"] == 2.0
    assert result["color_category"] == "grayscale"


def test_quality_analyzer_marks_decoded_rgb_image_valid(tmp_path: Path) -> None:
    path = tmp_path / "rgb.png"
    Image.new("RGB", (24, 12), color=(200, 80, 40)).save(path)

    result = QualityAnalyzer().analyze_safely(path)

    assert result["valid"] is True
    assert result["error"] is None
    assert result["width"] == 24
    assert result["height"] == 12


def test_unreadable_image_is_returned_as_invalid(tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")

    result = QualityAnalyzer().analyze_safely(path)

    assert result["valid"] is False
    assert "Unable to read image" in result["error"]


def test_blur_and_contrast_metrics_are_numerical_and_explainable() -> None:
    flat = np.full((32, 32), 128, dtype=np.uint8)
    blur = measure_blur(flat)
    contrast = measure_contrast(flat)

    assert blur["value"] == 0.0
    assert "possible blur" in blur["interpretation"]
    assert contrast["grayscale_standard_deviation"]["value"] == 0.0
    assert contrast["local_contrast"]["value"] == 0.0


def test_geometry_handles_small_blank_images_without_error() -> None:
    blank = np.full((8, 8), 255, dtype=np.uint8)
    result = measure_geometry(blank)

    assert result["skew_angle_degrees"]["value"] is None
    assert "too small" in result["skew_angle_degrees"]["interpretation"]
