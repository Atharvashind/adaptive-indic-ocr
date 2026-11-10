"""Aggregate image, quality, and annotation evidence for a dataset."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from statistics import mean, median
from typing import Any

from src.ingestion.base_dataset import DatasetManifest
from src.profiling.annotation_profiler import profile_annotations
from src.profiling.domain_profiler import profile_domains


def _numeric_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"minimum": None, "median": None, "mean": None, "maximum": None}
    return {
        "minimum": min(values),
        "median": float(median(values)),
        "mean": float(mean(values)),
        "maximum": max(values),
    }


def _split_size_overlap_report(manifest: DatasetManifest) -> dict[str, Any]:
    configured_report = (manifest.dataset_metadata or {}).get(
        "source_image_overlap"
    )
    if isinstance(configured_report, dict):
        return configured_report
    split_to_ids: dict[str, set[str]] = {}
    records = list(manifest.annotations) + list(manifest.reference_annotations)
    for annotation in records:
        split_name = str((annotation.metadata or {}).get("split") or "").strip().lower()
        if not split_name or not annotation.source_image_id:
            continue
        split_to_ids.setdefault(split_name, set()).add(annotation.source_image_id)
    pair_report: dict[str, dict[str, Any]] = {}
    for left, right in combinations(sorted(split_to_ids), 2):
        overlap = sorted(split_to_ids[left] & split_to_ids[right])
        pair_report[f"{left}_vs_{right}"] = {
            "overlap_count": len(overlap),
            "source_image_ids": overlap,
        }
    return {
        "split_distribution": {split: len(source_ids) for split, source_ids in sorted(split_to_ids.items())},
        "overlap_pairs": pair_report,
    }


def _quality_results_by_image(
    manifest: DatasetManifest,
    quality_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Align quality results to manifest paths and mark absent results invalid."""
    aligned: list[dict[str, Any]] = []
    for image in manifest.images:
        result = quality_results.get(image.relative_path)
        if result is None:
            result = {
                "valid": False,
                "error": (
                    f"No quality-analysis result was recorded for "
                    f"{image.relative_path}"
                ),
            }
        aligned.append(result)
    return aligned


def profile_dataset(
    manifest: DatasetManifest,
    quality_results: dict[str, dict[str, Any]],
    configured_labels: list[str] | None = None,
    user_domain: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-ready dataset profile from read-only image assessments."""
    image_results = _quality_results_by_image(manifest, quality_results)
    valid_results = [
        result for result in image_results if result.get("valid") is True
    ]
    unreadable_count = len(manifest.images) - len(valid_results)
    widths = [float(result["width"]) for result in valid_results]
    heights = [float(result["height"]) for result in valid_results]
    aspect_ratios = [float(result["aspect_ratio"]) for result in valid_results]
    file_sizes = [
        float(result["file_size_bytes"])
        for result in image_results
        if result.get("file_size_bytes") is not None
    ]

    format_counts = Counter(str(result["format"]) for result in valid_results)
    color_counts = Counter(str(result["color_category"]) for result in valid_results)
    language_counts = Counter(
        annotation.language
        for annotation in manifest.annotations
        if annotation.language
    )
    board_type_counts = Counter(
        str((annotation.metadata or {}).get("board_type") or "")
        for annotation in manifest.annotations
        if (annotation.metadata or {}).get("board_type")
    )
    split_counts = Counter(
        str((annotation.metadata or {}).get("split") or "").strip().lower()
        for annotation in manifest.annotations
        if (annotation.metadata or {}).get("split")
    )
    script_counts = Counter(
        annotation.script for annotation in manifest.annotations if annotation.script
    )
    brightness = [
        float(result["measurements"]["brightness"]["value"])
        for result in valid_results
    ]
    contrast = [
        float(result["measurements"]["grayscale_standard_deviation"]["value"])
        for result in valid_results
    ]
    blur = [
        float(result["measurements"]["blur"]["value"])
        for result in valid_results
        if result["measurements"]["blur"]["value"] is not None
    ]
    dimensions = [
        {"width": int(result["width"]), "height": int(result["height"])}
        for result in valid_results
    ]
    annotation_profile = profile_annotations(manifest)
    domain_profile = profile_domains(manifest, configured_labels, user_domain)
    source_image_overlap = _split_size_overlap_report(manifest)

    return {
        "dataset": manifest.name,
        "image_root": str(manifest.image_root),
        "total_images": (
            manifest.discovered_image_count
            if manifest.discovered_image_count is not None
            else len(manifest.images)
        ),
        "profiled_images": len(manifest.images),
        "sampling": {
            "sample_limit": manifest.sample_limit,
            "seed": manifest.sample_seed,
            "method": "stratified_by_split_and_language",
        },
        "valid_images": len(valid_results),
        "unreadable_images": unreadable_count,
        "image_dimensions": {
            "width_pixels": _numeric_summary(widths),
            "height_pixels": _numeric_summary(heights),
            "aspect_ratio": _numeric_summary(aspect_ratios),
            "per_image": dimensions,
        },
        "image_format_distribution": dict(sorted(format_counts.items())),
        "grayscale_vs_colour_distribution": dict(sorted(color_counts.items())),
        "file_size_bytes": _numeric_summary(file_sizes),
        "language_distribution": dict(sorted(language_counts.items())),
        "board_type_distribution": dict(sorted(board_type_counts.items())),
        "split_distribution": dict(sorted(split_counts.items())),
        "script_distribution": dict(sorted(script_counts.items())),
        "brightness_mean_grayscale": _numeric_summary(brightness),
        "contrast_grayscale_standard_deviation": _numeric_summary(contrast),
        "blur_laplacian_variance": _numeric_summary(blur),
        "source_image_overlap": source_image_overlap,
        "dataset_metadata": dict(manifest.dataset_metadata or {}),
        "annotations": annotation_profile,
        "domain": domain_profile,
        "mode": "user_specified" if user_domain else "automatic",
    }
