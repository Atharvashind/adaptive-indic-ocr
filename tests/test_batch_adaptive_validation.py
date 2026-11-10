"""Tests for reproducible batch adaptive-selector validation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from src.ingestion.base_dataset import AnnotationRecord, DatasetImage, DatasetManifest
from src.preprocessing.adaptive_selector import AdaptivePreprocessorSelector
from src.preprocessing.batch_validation import (
    BatchAdaptiveValidation,
    sample_manifest_by_source,
)
from src.quality.quality_analyzer import QualityAnalyzer


def _manifest(root: Path, include_missing: bool = False) -> DatasetManifest:
    image_root = root / "images"
    image_root.mkdir()
    images = []
    annotations = []
    records = [
        ("a-1.png", "source-a", "train", "hi", 20),
        ("a-2.png", "source-a", "train", "hi", 21),
        ("b-1.png", "source-b", "train", "bn", 100),
        ("c-1.png", "source-c", "val", "mr", 160),
        ("d-1.png", "source-d", "test", "ta", 230),
        ("e-1.png", "source-e", "test", "en", 70),
    ]
    for filename, source_id, split, language, contrast in records:
        path = image_root / filename
        pixels = np.tile(np.linspace(contrast, min(contrast + 20, 255), 64, dtype=np.uint8), (24, 1))
        Image.fromarray(pixels).save(path)
        annotation = AnnotationRecord(
            source_image_id=source_id,
            image_path=str(path),
            language=language,
            metadata={"split": split, "board_type": "test-board"},
        )
        annotations.append(annotation)
        images.append(
            DatasetImage(
                path=path,
                relative_path=filename,
                source_image_id=source_id,
                annotations=[annotation],
            )
        )
    if include_missing:
        missing_path = image_root / "missing.png"
        annotation = AnnotationRecord(
            source_image_id="source-missing",
            image_path=str(missing_path),
            language="te",
            metadata={"split": "test"},
        )
        annotations.append(annotation)
        images.append(
            DatasetImage(
                path=missing_path,
                relative_path=missing_path.name,
                source_image_id="source-missing",
                annotations=[annotation],
            )
        )
    return DatasetManifest(
        name="fixture",
        image_root=image_root,
        images=images,
        annotations=annotations,
        annotation_status="loaded",
        discovered_image_count=len(images),
        dataset_metadata={
            "source_domain": "scene_text",
            "domain_labels": ["scene_text"],
        },
    )


def _runner() -> BatchAdaptiveValidation:
    config = yaml.safe_load(
        Path("configs/preprocessing.yaml").read_text(encoding="utf-8")
    )
    return BatchAdaptiveValidation(
        AdaptivePreprocessorSelector(
            config,
            quality_analyzer=QualityAnalyzer(),
            operation_configuration=config,
        ),
        QualityAnalyzer(),
        user_domain="scene_text",
    )


def test_source_group_sampling_is_reproducible_and_preserves_strata(tmp_path) -> None:
    manifest = _manifest(tmp_path)

    first = sample_manifest_by_source(manifest, sample_limit=5, seed=7)
    second = sample_manifest_by_source(manifest, sample_limit=5, seed=7)

    assert [image.relative_path for image in first.images] == [
        image.relative_path for image in second.images
    ]
    source_ids = [image.source_image_id for image in first.images]
    assert len(source_ids) == len(set(source_ids))
    assert {str(image.annotations[0].metadata["split"]) for image in first.images} >= {
        "train",
        "val",
        "test",
    }


def test_batch_validation_writes_reports_and_preserves_sources(tmp_path) -> None:
    manifest = _manifest(tmp_path, include_missing=True)
    sampled = sample_manifest_by_source(manifest, sample_limit=6, seed=7)
    before_hashes = {
        image.relative_path: (
            hashlib.sha256(image.path.read_bytes()).hexdigest()
            if image.path.is_file()
            else None
        )
        for image in sampled.images
    }
    output = tmp_path / "adaptive-run"

    summary = _runner().run(sampled, output)

    required = {
        "sample_manifest.csv",
        "automatic_decisions.jsonl",
        "user_guided_decisions.jsonl",
        "decision_summary.json",
        "decision_summary.csv",
        "quality_measurements.csv",
        "validation_report.md",
        "operation_recommendation_distribution.png",
        "confidence_distribution.png",
        "quality_metrics_by_selected_operation.png",
        "language_recommendation_distribution.png",
        "mode_agreement.png",
    }
    assert required.issubset({path.name for path in output.iterdir()})
    assert summary["sample_size"] == 6
    assert summary["quality_failures"] == 1
    assert summary["source_file_integrity"]["unchanged"] == 5
    assert summary["source_file_integrity"]["missing_or_unverified"] == 1
    assert summary["sampling"]["source_ids_are_not_repartitioned"] is True

    auto_rows = [
        json.loads(line)
        for line in (output / "automatic_decisions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    user_rows = [
        json.loads(line)
        for line in (output / "user_guided_decisions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(auto_rows) == len(user_rows) == 6
    assert next(row for row in auto_rows if row["status"] == "failure")[
        "selected_operation"
    ] == "keep_original"
    assert all(row["output_path"] is None for row in auto_rows + user_rows)
    assert all(
        hashlib.sha256(image.path.read_bytes()).hexdigest() == before_hashes[image.relative_path]
        for image in sampled.images
        if image.path.is_file()
    )

    with (output / "sample_manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as source:
        manifest_rows = list(csv.DictReader(source))
    assert len(manifest_rows) == 6
    assert {row["language"] for row in manifest_rows} >= {"hi", "bn", "mr", "ta"}
    assert "failure" in {row["status"] for row in manifest_rows}

    with (output / "quality_measurements.csv").open(
        encoding="utf-8-sig", newline=""
    ) as source:
        quality_rows = list(csv.DictReader(source))
    failed_quality = next(row for row in quality_rows if row["valid"] == "False")
    assert failed_quality["error"]
    assert failed_quality["blur_variance_laplacian"] == ""


def test_batch_records_an_existing_but_unreadable_image(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    path = manifest.images[0].path
    path.write_bytes(b"not an image")
    output = tmp_path / "unreadable-run"

    summary = _runner().run(manifest, output)

    assert summary["quality_failures"] == 1
    rows = [
        json.loads(line)
        for line in (output / "automatic_decisions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    failed = next(row for row in rows if row["status"] == "failure")
    assert failed["error"]
    assert failed["selected_operation"] == "keep_original"
