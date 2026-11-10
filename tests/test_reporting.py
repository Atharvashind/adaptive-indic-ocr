"""Tests for non-overwriting report generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.ingestion.base_dataset import (
    AnnotationRecord,
    DatasetImage,
    DatasetManifest,
)
from src.profiling.dataset_profiler import profile_dataset
from src.quality.quality_analyzer import QualityAnalyzer
from src.reporting.dataset_report import generate_reports


def test_report_generation_writes_required_artifacts_and_preserves_runs(
    tmp_path: Path,
) -> None:
    manifest = DatasetManifest(
        name="empty-sample",
        image_root=tmp_path,
        images=[],
        annotations=[],
        annotation_status="missing",
        annotation_issues=["No annotation file configured"],
    )
    profile = profile_dataset(manifest, {})

    first = generate_reports(profile, manifest, {}, tmp_path / "reports")
    second = generate_reports(profile, manifest, {}, tmp_path / "reports")

    for directory in (first, second):
        assert {
            "dataset_profile.json",
            "image_quality.csv",
            "annotation_profile.csv",
            "dataset_summary.md",
            "dataset_comparison.csv",
            "image_dimensions.png",
            "aspect_ratio_distribution.png",
            "brightness_distribution.png",
            "contrast_distribution.png",
            "blur_distribution.png",
            "representative_samples.png",
        }.issubset({path.name for path in directory.iterdir()})
    assert first != second
    saved_profile = json.loads(
        (first / "dataset_profile.json").read_text(encoding="utf-8")
    )
    assert saved_profile["run_id"] == first.name


def test_quality_report_statuses_match_profile_and_keep_identifiers(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    rgb_path = image_root / "rgb.png"
    gray_path = image_root / "gray.png"
    broken_path = image_root / "broken.png"
    Image.new("RGB", (20, 10), color=(200, 80, 40)).save(rgb_path)
    Image.new("L", (16, 8), color=128).save(gray_path)
    broken_path.write_bytes(b"not an image")

    images = []
    for path, source_image_id in (
        (rgb_path, "board_rgb"),
        (gray_path, "board_gray"),
        (broken_path, "board_broken"),
    ):
        annotation = AnnotationRecord(
            source_image_id=source_image_id,
            image_path=str(path),
            granularity="line_crop",
            transcription="भारत निर्माण",
            language="hi",
            bbox=(1.0, 2.0, 3.0, 4.0),
            metadata={
                "filename": path.name,
                "source_image_id": source_image_id,
                "board_type": "GENERAL_INFORMATORY",
                "split": "train",
            },
        )
        images.append(
            DatasetImage(
                path=path,
                relative_path=path.name,
                source_image_id=source_image_id,
                annotations=[annotation],
            )
        )

    analyzer = QualityAnalyzer()
    quality_results = {
        image.relative_path: analyzer.analyze_safely(image.path) for image in images
    }
    failed_result = quality_results["broken.png"]
    failed_result["measurements"] = {
        "brightness": {"value": 255.0},
        "grayscale_standard_deviation": {"value": 255.0},
        "blur": {"value": 999999.0},
    }
    manifest = DatasetManifest(
        name="quality-contract",
        image_root=image_root,
        images=images,
        annotations=[annotation for image in images for annotation in image.annotations],
        annotation_status="loaded",
        discovered_image_count=3,
        dataset_metadata={
            "source_domain": "scene_text",
            "source_content": "photographed public information boards",
            "image_granularity": "line_crop",
        },
    )

    profile = profile_dataset(manifest, quality_results)
    output = generate_reports(profile, manifest, quality_results, tmp_path / "reports")
    saved_profile = json.loads(
        (output / "dataset_profile.json").read_text(encoding="utf-8")
    )
    with (output / "image_quality.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        quality_rows = list(csv.DictReader(handle))
    with (output / "annotation_profile.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        annotation_rows = list(csv.DictReader(handle))

    valid_rows = [row for row in quality_rows if row["valid"].lower() == "true"]
    invalid_rows = [row for row in quality_rows if row["valid"].lower() == "false"]
    assert len(valid_rows) == saved_profile["valid_images"] == 2
    assert len(invalid_rows) == saved_profile["unreadable_images"] == 1
    assert all(row["error"] for row in invalid_rows)
    assert saved_profile["image_dimensions"]["width_pixels"]["mean"] == 18.0
    assert saved_profile["brightness_mean_grayscale"]["mean"] < 255.0
    assert saved_profile["contrast_grayscale_standard_deviation"]["mean"] < 255.0
    assert saved_profile["blur_laplacian_variance"]["maximum"] < 999999.0
    assert all(row["granularity"] == "line_crop" for row in annotation_rows)
    assert all(row["source_image_id"] for row in annotation_rows)
    assert all(row["filename"] for row in annotation_rows)
    assert all(row["filename"] != row["source_image_id"] for row in annotation_rows)
    assert quality_rows[0]["source_image_id"] == "board_rgb"
    assert quality_rows[0]["filename"] == "rgb.png"
