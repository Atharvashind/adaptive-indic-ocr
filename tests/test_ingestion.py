"""Tests for image discovery and configured annotation adapters."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from src.ingestion.annotation_parser import (
    UnsupportedAnnotationFormatError,
    parse_annotations,
    parse_text_reference_annotations,
)
from src.ingestion.base_dataset import AnnotationRecord, DatasetImage, DatasetManifest
from src.ingestion.dataset_loader import load_dataset
from src.profiling.dataset_profiler import profile_dataset
from src.run_profiling import _apply_sample_limit


def _write_image(path: Path, size: tuple[int, int] = (16, 8)) -> None:
    Image.new("RGB", size, color=(200, 200, 200)).save(path)


def _image_split_for_test(image: DatasetImage) -> str:
    return str(image.annotations[0].metadata["split"])


def _image_language_for_test(image: DatasetImage) -> str:
    return str(image.annotations[0].language)


def test_loads_supported_images_and_keeps_unreadable_file(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_image(image_root / "good.PNG")
    (image_root / "broken.jpg").write_bytes(b"not an image")
    (image_root / "ignored.txt").write_text("not an image", encoding="utf-8")

    manifest = load_dataset(
        "sample",
        {"image_root": "images", "recursive": True, "annotation": None},
        tmp_path,
    )

    assert [image.path.name for image in manifest.images] == ["broken.jpg", "good.PNG"]
    assert manifest.annotation_status == "not_configured"


def test_csv_adapter_maps_fields_and_preserves_source_image_id(tmp_path: Path) -> None:
    root = tmp_path / "images"
    root.mkdir()
    _write_image(root / "page.png")
    annotation_file = tmp_path / "labels.csv"
    with annotation_file.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "file",
                "page_key",
                "word",
                "level",
                "lang",
                "x",
                "y",
                "w",
                "h",
                "poly",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "file": "page.png",
                "page_key": "original-42",
                "word": "ಕನ್ನಡ",
                "level": "word",
                "lang": "kn",
                "x": "1",
                "y": "2",
                "w": "6",
                "h": "3",
                "poly": "[[1, 2], [7, 2], [7, 5]]",
            }
        )

    manifest = load_dataset(
        "sample",
        {
            "image_root": "images",
            "annotation": {
                "format": "csv",
                "path": "labels.csv",
                "columns": {
                    "image_path": "file",
                    "source_image_id": "page_key",
                    "transcription": "word",
                    "granularity": "level",
                    "language": "lang",
                },
                "bbox_columns": ["x", "y", "w", "h"],
                "polygon_field": "poly",
            },
        },
        tmp_path,
    )

    assert manifest.images[0].source_image_id == "original-42"
    annotation = manifest.annotations[0]
    assert annotation.transcription == "ಕನ್ನಡ"
    assert annotation.granularity == "word"
    assert annotation.bbox == (1.0, 2.0, 6.0, 3.0)
    assert annotation.polygon == [(1.0, 2.0), (7.0, 2.0), (7.0, 5.0)]


def test_missing_annotation_file_is_reported_without_failing(tmp_path: Path) -> None:
    root = tmp_path / "images"
    root.mkdir()
    _write_image(root / "page.png")

    manifest = load_dataset(
        "sample",
        {
            "image_root": "images",
            "annotation": {"format": "csv", "path": "not-found.csv"},
        },
        tmp_path,
    )

    assert len(manifest.images) == 1
    assert manifest.annotations == []
    assert manifest.annotation_status == "missing"
    assert "not found" in manifest.annotation_issues[0].lower()


def test_unsupported_format_is_explicit(tmp_path: Path) -> None:
    annotation_file = tmp_path / "labels.xml"
    annotation_file.write_text("<annotations/>", encoding="utf-8")
    with pytest.raises(UnsupportedAnnotationFormatError, match="Unsupported"):
        parse_annotations(
            annotation_file,
            tmp_path,
            {"format": "xml"},
        )


def test_coco_instances_adapter_preserves_ids_and_polygon_parts(
    tmp_path: Path,
) -> None:
    annotation_file = tmp_path / "coco.json"
    annotation_file.write_text(
        json.dumps(
            {
                "images": [{"id": 17, "file_name": "page.png"}],
                "annotations": [
                    {
                        "id": 89,
                        "image_id": 17,
                        "bbox": [1, 2, 3, 4],
                        "segmentation": [[1, 2, 4, 2, 4, 6], [8, 8, 9, 8, 9, 9]],
                        "text": "sample",
                        "lang": "ta",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    parsed = parse_annotations(
        annotation_file,
        tmp_path,
        {
            "format": "coco_instances",
            "transcription_field": "text",
            "language_field": "lang",
        },
    )

    annotation = parsed.annotations[0]
    assert annotation.source_image_id == "17"
    assert annotation.annotation_id == "89"
    assert annotation.transcription == "sample"
    assert annotation.language == "ta"
    assert len(annotation.polygons or []) == 2


def test_metadata_csv_parses_bbox_language_and_unicode_transcription(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_image(image_root / "line_000001.png")
    metadata_path = tmp_path / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename",
                "source_image_id",
                "bbox",
                "language",
                "board_type",
                "transcription",
                "split",
                "crop_width",
                "crop_height",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "filename": "line_000001.png",
                "source_image_id": "board_0001",
                "bbox": "[10, 20, 30, 40]",
                "language": "mr",
                "board_type": "TECHNICAL_CROSS_SECTION_BOARD_A",
                "transcription": "भारत निर्माण",
                "split": "train",
                "crop_width": "21",
                "crop_height": "21",
            }
        )

    manifest = load_dataset(
        "synth6k",
        {
            "image_root": "images",
            "source_domain": "scene_text",
            "source_content": "photographed public information boards",
            "image_granularity": "line_crop",
            "annotation": {
                "format": "csv",
                "path": "metadata.csv",
                "columns": {
                    "image_path": "filename",
                    "source_image_id": "source_image_id",
                    "transcription": "transcription",
                    "language": "language",
                    "board_type": "board_type",
                    "split": "split",
                },
                "granularity": "line_crop",
                "bbox_field": "bbox",
            },
        },
        tmp_path,
    )

    assert len(manifest.annotations) == 1
    assert manifest.images[0].source_image_id == "board_0001"
    assert manifest.annotations[0].bbox == (10.0, 20.0, 30.0, 40.0)
    assert manifest.annotations[0].language == "mr"
    assert manifest.annotations[0].transcription == "भारत निर्माण"
    assert manifest.annotations[0].metadata["split"] == "train"
    assert manifest.annotations[0].metadata["filename"] == "line_000001.png"
    assert manifest.annotations[0].metadata["bbox"] == "[10, 20, 30, 40]"
    assert manifest.annotations[0].metadata["crop_width"] == "21"
    assert manifest.annotations[0].granularity == "line_crop"


def test_text_reference_annotations_preserve_transcription_whitespace(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_image(image_root / "line_000001.png")
    annotation_file = tmp_path / "annotations" / "train.txt"
    annotation_file.parent.mkdir()
    annotation_file.write_text(
        "line_000001.png भारत निर्माण और सड़क योजना\n",
        encoding="utf-8",
    )

    parsed = parse_text_reference_annotations(
        annotation_file,
        image_root,
        {"granularity": "line", "split_name": "train"},
    )

    assert len(parsed.annotations) == 1
    assert parsed.annotations[0].transcription == "भारत निर्माण और सड़क योजना"
    assert parsed.annotations[0].metadata["split"] == "train"


def test_metadata_and_text_reference_consistency_are_retained(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_image(image_root / "line_000001.png")
    metadata_path = tmp_path / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "source_image_id", "bbox", "language", "board_type", "transcription", "split", "crop_width", "crop_height"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "filename": "line_000001.png",
                "source_image_id": "board_0001",
                "bbox": "[1, 2, 3, 4]",
                "language": "hi",
                "board_type": "GENERAL_INFORMATORY",
                "transcription": "भारत निर्माण",
                "split": "train",
                "crop_width": "10",
                "crop_height": "5",
            }
        )
    annotation_file = tmp_path / "annotations" / "train.txt"
    annotation_file.parent.mkdir()
    annotation_file.write_text("line_000001.png भारत निर्माण\n", encoding="utf-8")

    manifest = load_dataset(
        "synth6k",
        {
            "image_root": "images",
            "source_domain": "scene_text",
            "source_content": "photographed public information boards",
            "image_granularity": "line_crop",
            "annotation": {
                "format": "csv",
                "path": "metadata.csv",
                "columns": {
                    "image_path": "filename",
                    "source_image_id": "source_image_id",
                    "transcription": "transcription",
                    "language": "language",
                    "board_type": "board_type",
                    "split": "split",
                },
                "bbox_field": "bbox",
                "reference_files": ["annotations/train.txt"],
            },
        },
        tmp_path,
    )

    assert manifest.annotations[0].transcription == "भारत निर्माण"
    assert manifest.reference_annotations[0].transcription == "भारत निर्माण"
    assert manifest.reference_annotations[0].metadata["split"] == "train"
    assert manifest.dataset_metadata["text_reference_validation"]["status"] == "consistent"
    assert manifest.dataset_metadata["text_reference_validation"]["transcription_mismatch_count"] == 0


def test_split_preservation_and_source_image_overlap_report(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    for name in ["line_000001.png", "line_000002.png", "line_000003.png"]:
        _write_image(image_root / name)

    metadata_path = tmp_path / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["filename", "source_image_id", "bbox", "language", "board_type", "transcription", "split", "crop_width", "crop_height"],
        )
        writer.writeheader()
        writer.writerow({"filename": "line_000001.png", "source_image_id": "board_1", "bbox": "[1, 2, 3, 4]", "language": "mr", "board_type": "BOARD_A", "transcription": "भारत", "split": "train", "crop_width": "10", "crop_height": "5"})
        writer.writerow({"filename": "line_000002.png", "source_image_id": "board_2", "bbox": "[1, 2, 3, 4]", "language": "en", "board_type": "BOARD_A", "transcription": "road", "split": "train", "crop_width": "10", "crop_height": "5"})
        writer.writerow({"filename": "line_000003.png", "source_image_id": "board_1", "bbox": "[1, 2, 3, 4]", "language": "mr", "board_type": "BOARD_A", "transcription": "भारत", "split": "val", "crop_width": "10", "crop_height": "5"})

    manifest = load_dataset(
        "synth6k",
        {
            "image_root": "images",
            "annotation": {
                "format": "csv",
                "path": "metadata.csv",
                "columns": {
                    "image_path": "filename",
                    "source_image_id": "source_image_id",
                    "transcription": "transcription",
                    "language": "language",
                    "board_type": "board_type",
                    "split": "split",
                },
                "bbox_field": "bbox",
            },
        },
        tmp_path,
    )

    profile = profile_dataset(manifest, {})
    assert profile["split_distribution"]["train"] == 2
    assert profile["split_distribution"]["val"] == 1
    assert profile["source_image_overlap"]["overlap_pairs"]["train_vs_val"]["overlap_count"] == 1


def test_sampling_is_reproducible_and_covers_each_split_and_language(
    tmp_path: Path,
) -> None:
    images = []
    for split in ("train", "val", "test"):
        for language in ("hi", "mr", "en"):
            for index in range(8):
                image_path = tmp_path / f"{split}_{language}_{index}.png"
                _write_image(image_path)
                record = AnnotationRecord(
                    language=language,
                    metadata={"split": split},
                )
                images.append(
                    DatasetImage(
                        path=image_path,
                        relative_path=image_path.name,
                        annotations=[record],
                    )
                )

    manifests = [
        DatasetManifest(
            name="sample",
            image_root=tmp_path,
            images=list(images),
            annotations=[
                annotation for image in images for annotation in image.annotations
            ],
            annotation_status="loaded",
            discovered_image_count=len(images),
        )
        for _ in range(2)
    ]
    for manifest in manifests:
        _apply_sample_limit(manifest, 18, 19)

    names = [
        [image.relative_path for image in manifest.images]
        for manifest in manifests
    ]
    assert names[0] == names[1]
    assert len(names[0]) == 18
    assert {_image_split_for_test(image) for image in manifests[0].images} == {
        "train",
        "val",
        "test",
    }
    assert {_image_language_for_test(image) for image in manifests[0].images} == {
        "hi",
        "mr",
        "en",
    }


def test_missing_image_is_reported_in_dataset_issues(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    metadata_path = tmp_path / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["filename", "source_image_id", "bbox", "language", "board_type", "transcription", "split", "crop_width", "crop_height"],
        )
        writer.writeheader()
        writer.writerow({"filename": "missing.png", "source_image_id": "board_99", "bbox": "[1, 2, 3, 4]", "language": "bn", "board_type": "BOARD_A", "transcription": "missing", "split": "train", "crop_width": "10", "crop_height": "5"})

    manifest = load_dataset(
        "synth6k",
        {
            "image_root": "images",
            "annotation": {
                "format": "csv",
                "path": "metadata.csv",
                "columns": {
                    "image_path": "filename",
                    "source_image_id": "source_image_id",
                    "transcription": "transcription",
                    "language": "language",
                    "board_type": "board_type",
                    "split": "split",
                },
                "bbox_field": "bbox",
            },
        },
        tmp_path,
    )

    assert len(manifest.annotations) == 1
    assert any("missing" in issue.lower() or "linked" in issue.lower() for issue in manifest.annotation_issues)
