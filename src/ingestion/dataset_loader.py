"""Discover dataset images and attach annotations without modifying source files."""

from __future__ import annotations

import logging
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

from src.ingestion.annotation_parser import (
    parse_annotations,
    parse_text_reference_annotations,
)
from src.ingestion.base_dataset import DatasetImage, DatasetManifest

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}


class DatasetConfigurationError(ValueError):
    """Raised when a dataset cannot be loaded from its configuration."""


def _normal_key(path: Path) -> str:
    return str(path.resolve()).casefold()


def _source_split_report(annotations: list[Any]) -> dict[str, Any]:
    ids_by_split: dict[str, set[str]] = {}
    for annotation in annotations:
        split = str((annotation.metadata or {}).get("split") or "").strip().lower()
        if split and annotation.source_image_id:
            ids_by_split.setdefault(split, set()).add(annotation.source_image_id)
    overlaps = {}
    for left, right in combinations(sorted(ids_by_split), 2):
        source_ids = sorted(ids_by_split[left] & ids_by_split[right])
        overlaps[f"{left}_vs_{right}"] = {
            "overlap_count": len(source_ids),
            "source_image_ids": source_ids,
        }
    split_counts = {
        split: len(source_ids) for split, source_ids in sorted(ids_by_split.items())
    }
    return {
        "split_distribution": split_counts,
        "unique_source_image_ids_by_split": split_counts,
        "overlap_pairs": overlaps,
    }


def discover_images(
    image_root: Path,
    extensions: list[str] | None = None,
    recursive: bool = True,
) -> list[Path]:
    """Return supported image paths in deterministic order."""
    allowed = {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in (extensions or sorted(SUPPORTED_IMAGE_EXTENSIONS))
    }
    unsupported = allowed - SUPPORTED_IMAGE_EXTENSIONS
    if unsupported:
        logger.warning(
            "Ignoring unsupported configured image extensions: %s", sorted(unsupported)
        )
    allowed &= SUPPORTED_IMAGE_EXTENSIONS
    iterator = image_root.rglob("*") if recursive else image_root.glob("*")
    return sorted(
        (path for path in iterator if path.is_file() and path.suffix.lower() in allowed),
        key=lambda path: str(path).casefold(),
    )


def load_dataset(
    name: str,
    dataset_config: dict[str, Any],
    project_root: Path,
) -> DatasetManifest:
    """Load image paths and configured annotations from a dataset entry."""
    raw_image_root = dataset_config.get("image_root")
    if not raw_image_root:
        raise DatasetConfigurationError(f"Dataset {name!r} has no image_root configured.")
    image_root = Path(str(raw_image_root)).expanduser()
    if not image_root.is_absolute():
        image_root = project_root / image_root
    image_root = image_root.resolve()
    if not image_root.is_dir():
        raise DatasetConfigurationError(
            f"Image directory for dataset {name!r} does not exist: {image_root}. "
            "Update configs/datasets.yaml with the verified local path."
        )

    image_paths = discover_images(
        image_root,
        extensions=dataset_config.get("image_extensions"),
        recursive=bool(dataset_config.get("recursive", True)),
    )
    images = [
        DatasetImage(path=path, relative_path=path.relative_to(image_root).as_posix())
        for path in image_paths
    ]

    annotation_config = dataset_config.get("annotation")
    annotation_path: Path | None = None
    if isinstance(annotation_config, dict) and annotation_config.get("path"):
        annotation_path = Path(str(annotation_config["path"])).expanduser()
        if not annotation_path.is_absolute():
            annotation_path = project_root / annotation_path
        annotation_path = annotation_path.resolve()

    if annotation_config is not None and not isinstance(annotation_config, dict):
        raise DatasetConfigurationError(
            f"Annotation configuration for dataset {name!r} must be a mapping."
        )
    parsed = parse_annotations(annotation_path, image_root, annotation_config)
    reference_annotations: list[Any] = []
    reference_issues: list[str] = []
    reference_files = (
        annotation_config.get("reference_files", [])
        if isinstance(annotation_config, dict)
        else []
    )
    if not isinstance(reference_files, list):
        raise DatasetConfigurationError(
            f"reference_files for dataset {name!r} must be a list."
        )
    for reference_file in reference_files:
        ref_path = Path(str(reference_file)).expanduser()
        if not ref_path.is_absolute():
            ref_path = project_root / ref_path
        ref_path = ref_path.resolve()
        ref_result = parse_text_reference_annotations(
            ref_path,
            image_root,
            {"granularity": "line", "split_name": ref_path.stem},
        )
        reference_annotations.extend(ref_result.annotations)
        reference_issues.extend(ref_result.issues)

    metadata_by_path = {
        _normal_key(Path(annotation.image_path)): annotation
        for annotation in parsed.annotations
        if annotation.image_path
    }
    image_ids_by_path = {
        _normal_key(Path(annotation.image_path)): annotation.source_image_id
        for annotation in parsed.annotations
        if annotation.image_path and annotation.source_image_id
    }
    reference_paths: list[str] = []
    matched_reference_count = 0
    transcription_mismatch_count = 0
    split_mismatch_count = 0
    for reference in reference_annotations:
        if not reference.image_path:
            continue
        reference_key = _normal_key(Path(reference.image_path))
        reference_paths.append(reference_key)
        source_image_id = image_ids_by_path.get(reference_key)
        if source_image_id:
            reference.source_image_id = source_image_id
        metadata_annotation = metadata_by_path.get(reference_key)
        if metadata_annotation is None:
            continue
        matched_reference_count += 1
        if reference.transcription != metadata_annotation.transcription:
            transcription_mismatch_count += 1
        metadata_split = str(
            (metadata_annotation.metadata or {}).get("split") or ""
        ).strip().lower()
        reference_split = str(
            (reference.metadata or {}).get("split") or ""
        ).strip().lower()
        if metadata_split and reference_split and metadata_split != reference_split:
            split_mismatch_count += 1
    reference_counts = Counter(reference_paths)
    unique_metadata_paths = {
        _normal_key(Path(annotation.image_path))
        for annotation in parsed.annotations
        if annotation.image_path
    }
    unique_reference_paths = set(reference_paths)
    reference_validation = {
        "reference_record_count": len(reference_annotations),
        "matched_record_count": matched_reference_count,
        "metadata_records_without_reference": len(
            unique_metadata_paths - unique_reference_paths
        ),
        "reference_records_without_metadata": len(
            unique_reference_paths - unique_metadata_paths
        ),
        "duplicate_reference_record_count": sum(
            count - 1 for count in reference_counts.values() if count > 1
        ),
        "transcription_mismatch_count": transcription_mismatch_count,
        "split_mismatch_count": split_mismatch_count,
        "status": (
            "unavailable"
            if not reference_files
            else "consistent"
            if (
                matched_reference_count == len(parsed.annotations)
                and len(unique_metadata_paths) == len(unique_reference_paths)
                and not any(reference_counts[path] > 1 for path in reference_counts)
                and transcription_mismatch_count == 0
                and split_mismatch_count == 0
            )
            else "inconsistent"
        ),
    }
    by_path = {_normal_key(image.path): image for image in images}
    for path_key, source_image_id in parsed.image_ids_by_path.items():
        image = by_path.get(path_key)
        if image is None:
            parsed.issues.append(
                f"Annotation references an image path not discovered under "
                f"{image_root}: {path_key}"
            )
            continue
        if image.source_image_id and image.source_image_id != source_image_id:
            parsed.issues.append(
                f"Conflicting source image IDs for {image.relative_path!r}: "
                f"{image.source_image_id!r} and {source_image_id!r}"
            )
        else:
            image.source_image_id = source_image_id

    pattern = dataset_config.get("image_id_filename_pattern")
    for annotation in parsed.annotations:
        image: DatasetImage | None = None
        if annotation.image_path:
            image = by_path.get(_normal_key(Path(annotation.image_path)))
        elif annotation.source_image_id and pattern:
            try:
                expected = str(pattern).format(image_id=annotation.source_image_id)
            except (KeyError, ValueError) as error:
                parsed.issues.append(
                    f"Invalid image_id_filename_pattern {pattern!r}: {error}"
                )
                continue
            expected_path = Path(expected)
            if not expected_path.is_absolute():
                expected_path = image_root / expected_path
            image = by_path.get(_normal_key(expected_path))
        if image is None:
            parsed.issues.append(
                f"Annotation {annotation.annotation_id or '<without ID>'!r} "
                "could not be linked to a discovered image"
            )
            continue
        if annotation.source_image_id:
            if image.source_image_id and image.source_image_id != annotation.source_image_id:
                parsed.issues.append(
                    f"Conflicting source image IDs for {image.relative_path!r}"
                )
            image.source_image_id = annotation.source_image_id
        image.annotations.append(annotation)

    for issue in parsed.issues:
        logger.warning("%s", issue)
    if not annotation_config or not annotation_path:
        annotation_status = "not_configured"
    elif not annotation_path.is_file():
        annotation_status = "missing"
    elif parsed.annotations:
        annotation_status = "loaded"
    elif parsed.issues:
        annotation_status = "inconsistent"
    else:
        annotation_status = "empty"
    return DatasetManifest(
        name=name,
        image_root=image_root,
        images=images,
        annotations=parsed.annotations,
        annotation_status=annotation_status,
        annotation_issues=parsed.issues + reference_issues,
        discovered_image_count=len(images),
        reference_annotations=reference_annotations,
        dataset_metadata={
            "source_domain": dataset_config.get("source_domain", "scene_text"),
            "source_content": dataset_config.get(
                "source_content",
                "photographed public information boards",
            ),
            "image_granularity": dataset_config.get("image_granularity", "line_crop"),
            "domain_labels": dataset_config.get("domain_labels", []),
            "source_image_overlap": _source_split_report(parsed.annotations),
            "text_reference_validation": reference_validation,
        },
    )
