"""Generate preserved, per-run dataset profiling reports and visualizations."""

from __future__ import annotations

import json
import logging
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.ingestion.base_dataset import DatasetManifest

logger = logging.getLogger(__name__)


def _new_run_directory(dataset_output_root: Path) -> tuple[Path, str]:
    dataset_output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_name = f"run-{timestamp}"
    candidate = dataset_output_root / base_name
    suffix = 1
    while candidate.exists():
        candidate = dataset_output_root / f"{base_name}-{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate, candidate.name


def _quality_csv_rows(
    manifest: DatasetManifest, quality_results: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image in manifest.images:
        result = quality_results.get(
            image.relative_path,
            {
                "valid": False,
                "error": (
                    f"No quality-analysis result was recorded for "
                    f"{image.relative_path}"
                ),
            },
        )
        row: dict[str, Any] = {
            "source_image_id": image.source_image_id,
            "filename": image.path.name,
            "image_path": image.relative_path,
            "valid": result.get("valid") is True,
            "error": result.get("error"),
            "width": result.get("width"),
            "height": result.get("height"),
            "aspect_ratio": result.get("aspect_ratio"),
            "format": result.get("format"),
            "source_mode": result.get("source_mode"),
            "color_category": result.get("color_category"),
            "file_size_bytes": result.get("file_size_bytes"),
        }
        for metric_name, metric in result.get("measurements", {}).items():
            row[f"{metric_name}_value"] = metric.get("value")
            row[f"{metric_name}_interpretation"] = metric.get("interpretation")
            row[f"{metric_name}_confidence"] = metric.get("confidence")
            row[f"{metric_name}_warnings"] = " | ".join(metric.get("warnings", []))
            if metric_name == "brightness":
                row["brightness_standard_deviation"] = metric.get("standard_deviation")
                row["brightness_distribution_16_bins"] = json.dumps(
                    metric.get("distribution_16_bins", [])
                )
        row["measurement_warnings"] = " | ".join(result.get("warnings", []))
        rows.append(row)
    return rows


def _annotation_csv_rows(manifest: DatasetManifest) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image in manifest.images:
        if not image.annotations:
            rows.append(
                {
                    "source_image_id": image.source_image_id,
                    "filename": image.path.name,
                    "image_path": image.relative_path,
                    "annotation_id": None,
                    "parent_annotation_id": None,
                    "granularity": None,
                    "transcription": None,
                    "transcription_length_characters": None,
                    "language": None,
                    "script": None,
                    "bbox": None,
                    "polygon": None,
                    "polygons": None,
                    "board_type": None,
                    "split": None,
                    "crop_width": None,
                    "crop_height": None,
                    "metadata": None,
                }
            )
        for annotation in image.annotations:
            rows.append(
                {
                    "source_image_id": (
                        annotation.source_image_id or image.source_image_id
                    ),
                    "filename": image.path.name,
                    "image_path": image.relative_path,
                    "annotation_id": annotation.annotation_id,
                    "parent_annotation_id": annotation.parent_annotation_id,
                    "granularity": annotation.granularity,
                    "transcription": annotation.transcription,
                    "transcription_length_characters": (
                        len(annotation.transcription)
                        if annotation.transcription is not None
                        else None
                    ),
                    "language": annotation.language,
                    "script": annotation.script,
                    "bbox": json.dumps(annotation.bbox) if annotation.bbox else None,
                    "polygon": json.dumps(annotation.polygon) if annotation.polygon else None,
                    "polygons": (
                        json.dumps(annotation.polygons)
                        if annotation.polygons
                        else None
                    ),
                    "board_type": (annotation.metadata or {}).get("board_type"),
                    "split": (annotation.metadata or {}).get("split"),
                    "crop_width": (annotation.metadata or {}).get("crop_width"),
                    "crop_height": (annotation.metadata or {}).get("crop_height"),
                    "metadata": json.dumps(annotation.metadata, ensure_ascii=False),
                }
            )
    return rows


def _numeric_metric(profile: dict[str, Any], key: str) -> float | None:
    value = profile.get(key, {}).get("mean")
    return float(value) if value is not None else None


def _comparison_profiles(
    profiling_root: Path, current_profile: dict[str, Any]
) -> list[dict[str, Any]]:
    profiles = [current_profile]
    if profiling_root.is_dir():
        for path in profiling_root.glob("*/run-*/dataset_profile.json"):
            try:
                with path.open("r", encoding="utf-8") as source:
                    candidate = json.load(source)
            except (OSError, json.JSONDecodeError) as error:
                logger.warning("Skipping unreadable prior profile %s: %s", path, error)
                continue
            if candidate.get("dataset") != current_profile.get("dataset"):
                profiles.append(candidate)

    latest_by_dataset: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        name = str(profile.get("dataset", "unknown"))
        prior = latest_by_dataset.get(name)
        if prior is None or str(profile.get("generated_at", "")) > str(
            prior.get("generated_at", "")
        ):
            latest_by_dataset[name] = profile
    return [latest_by_dataset[name] for name in sorted(latest_by_dataset)]


def _write_comparison(
    profiling_root: Path, profile: dict[str, Any], run_directory: Path
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for candidate in _comparison_profiles(profiling_root, profile):
        dimensions = candidate.get("image_dimensions", {})
        annotations = candidate.get("annotations", {})
        comparisons.append(
            {
                "dataset": candidate.get("dataset"),
                "total_images": candidate.get("total_images"),
                "valid_images": candidate.get("valid_images"),
                "unreadable_images": candidate.get("unreadable_images"),
                "mean_width": _numeric_metric(dimensions, "width_pixels"),
                "mean_height": _numeric_metric(dimensions, "height_pixels"),
                "mean_aspect_ratio": _numeric_metric(dimensions, "aspect_ratio"),
                "mean_brightness": _numeric_metric(
                    candidate, "brightness_mean_grayscale"
                ),
                "mean_contrast": _numeric_metric(
                    candidate, "contrast_grayscale_standard_deviation"
                ),
                "mean_blur_laplacian_variance": _numeric_metric(
                    candidate, "blur_laplacian_variance"
                ),
                "annotation_status": annotations.get("annotation_status"),
            }
        )
    pd.DataFrame(comparisons).to_csv(
        run_directory / "dataset_comparison.csv", index=False
    )
    return comparisons


def _plot_histogram(
    values: list[float], title: str, xlabel: str, output_path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    if values:
        axis.hist(values, bins=min(30, max(5, int(len(values) ** 0.5))), color="#4472C4")
    else:
        axis.text(0.5, 0.5, "No valid measurements", ha="center", va="center")
    axis.set(title=title, xlabel=xlabel, ylabel="Image count")
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def _write_visualizations(
    manifest: DatasetManifest,
    quality_results: dict[str, dict[str, Any]],
    output_directory: Path,
) -> None:
    valid = [
        quality_results.get(
            image.relative_path,
            {"valid": False},
        )
        for image in manifest.images
        if quality_results.get(image.relative_path, {}).get("valid") is True
    ]
    widths = [float(result["width"]) for result in valid]
    heights = [float(result["height"]) for result in valid]
    aspects = [float(result["aspect_ratio"]) for result in valid]
    brightness = [
        float(result["measurements"]["brightness"]["value"]) for result in valid
    ]
    contrast = [
        float(result["measurements"]["grayscale_standard_deviation"]["value"])
        for result in valid
    ]
    blur = [
        float(result["measurements"]["blur"]["value"])
        for result in valid
        if result["measurements"]["blur"]["value"] is not None
    ]

    figure, axis = plt.subplots(figsize=(6, 5))
    if widths:
        axis.scatter(widths, heights, alpha=0.65, s=18)
    else:
        axis.text(0.5, 0.5, "No valid dimensions", ha="center", va="center")
    axis.set(title="Image dimensions", xlabel="Width (pixels)", ylabel="Height (pixels)")
    figure.tight_layout()
    figure.savefig(output_directory / "image_dimensions.png", dpi=140)
    plt.close(figure)

    _plot_histogram(
        aspects, "Aspect-ratio distribution", "Width / height",
        output_directory / "aspect_ratio_distribution.png",
    )
    _plot_histogram(
        brightness, "Brightness distribution", "Mean grayscale intensity (0-255)",
        output_directory / "brightness_distribution.png",
    )
    _plot_histogram(
        contrast, "Contrast distribution", "Grayscale standard deviation",
        output_directory / "contrast_distribution.png",
    )
    _plot_histogram(
        blur, "Blur indicator distribution", "Variance of Laplacian",
        output_directory / "blur_distribution.png",
    )

    available_samples = [
        image
        for image in manifest.images
        if quality_results.get(image.relative_path, {}).get("valid") is True
    ]
    seed = manifest.sample_seed or 0
    candidates = sorted(
        random.Random(seed).sample(
            available_samples, min(8, len(available_samples))
        ),
        key=lambda image: image.relative_path.casefold(),
    )
    columns = 2
    rows = max(1, (len(candidates) + columns - 1) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(10, 4 * rows), squeeze=False)
    for axis, image in zip(axes.flat, candidates):
        try:
            with Image.open(image.path) as source:
                axis.imshow(source.convert("RGB"))
            axis.set_title(image.relative_path)
            axis.axis("off")
        except (OSError, UnidentifiedImageError) as error:
            logger.warning("Skipping representative sample %s: %s", image.path, error)
            axis.text(0.5, 0.5, "Could not render source image", ha="center")
            axis.axis("off")
    for axis in list(axes.flat)[len(candidates):]:
        axis.axis("off")
    if not candidates:
        axes.flat[0].text(0.5, 0.5, "No valid source images", ha="center", va="center")
        axes.flat[0].axis("off")
    figure.suptitle("Representative source-image samples")
    figure.tight_layout()
    figure.savefig(output_directory / "representative_samples.png", dpi=140)
    plt.close(figure)


def _write_summary(
    profile: dict[str, Any],
    comparisons: list[dict[str, Any]],
    output_path: Path,
) -> None:
    annotation = profile.get("annotations", {})
    dimensions = profile.get("image_dimensions", {})
    domain = profile.get("domain", {"metadata_domain_labels": []})
    dataset_metadata = profile.get("dataset_metadata", {})
    source_overlap = profile.get("source_image_overlap", {})
    reference_validation = dataset_metadata.get("text_reference_validation", {})
    sampling = profile.get("sampling", {})
    total_images = profile.get("total_images", 0)
    valid_images = profile.get("valid_images", 0)
    unreadable_images = profile.get("unreadable_images", 0)
    lines = [
        f"# Dataset profile: {profile.get('dataset', 'unknown')}",
        "",
        f"- Generated (UTC): {profile.get('generated_at', 'unknown')}",
        f"- Mode: {profile.get('mode', 'automatic')}",
        f"- Sampling: {sampling.get('method', 'unsampled')}; "
        f"seed={sampling.get('seed', 'n/a')}; limit={sampling.get('sample_limit', 'none')}",
        f"- Images: {total_images} total; {valid_images} valid; {unreadable_images} unreadable",
        f"- Annotation status: {annotation.get('annotation_status', 'unknown')}",
        f"- Annotations: {annotation.get('annotation_count', 0)}",
        f"- Width (mean): {dimensions.get('width_pixels', {}).get('mean', 'n/a')}",
        f"- Height (mean): {dimensions.get('height_pixels', {}).get('mean', 'n/a')}",
        f"- Aspect ratio (mean): {dimensions.get('aspect_ratio', {}).get('mean', 'n/a')}",
        f"- Format distribution: {profile.get('image_format_distribution', {})}",
        f"- Grayscale/colour distribution: {profile.get('grayscale_vs_colour_distribution', {})}",
        f"- Annotation granularity: {annotation.get('granularity_distribution', {})}",
        f"- Language distribution: {profile.get('language_distribution', {})}",
        f"- Script distribution: {profile.get('script_distribution', {})}",
        f"- Board-type distribution: {profile.get('board_type_distribution', {})}",
        f"- Split distribution: {profile.get('split_distribution', {})}",
        f"- Source domain/content/granularity: "
        f"{domain.get('source_domain', 'n/a')} / "
        f"{domain.get('source_content', 'n/a')} / "
        f"{domain.get('image_granularity', 'n/a')}",
        f"- Source-image split overlaps: {source_overlap.get('overlap_pairs', {})}",
        f"- Text-reference validation: {reference_validation}",
        f"- Metadata domain labels: {domain.get('metadata_domain_labels', [])}",
        "- Image-based domain classification: unavailable (no classifier implemented).",
        "",
        "Quality values are explainable indicators, not OCR performance claims. "
        "Noise residuals can include text strokes and texture. No source images were modified.",
        "",
        "## Dataset comparisons",
        "",
    ]
    if comparisons:
        lines.extend(
            [
                "| Dataset | Images | Valid | Mean width | Mean height | Mean brightness | Mean contrast |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in comparisons:
            lines.append(
                "| {dataset} | {total_images} | {valid_images} | {mean_width} | "
                "{mean_height} | {mean_brightness} | {mean_contrast} |".format(
                    dataset=item.get("dataset", "unknown"),
                    total_images=item.get("total_images", "n/a"),
                    valid_images=item.get("valid_images", "n/a"),
                    mean_width=item.get("mean_width", "n/a"),
                    mean_height=item.get("mean_height", "n/a"),
                    mean_brightness=item.get("mean_brightness", "n/a"),
                    mean_contrast=item.get("mean_contrast", "n/a"),
                )
            )
    else:
        lines.append("No other dataset profiles are available for comparison.")
    lines.extend(["", "## Visualizations", ""])
    for filename in (
        "image_dimensions.png",
        "aspect_ratio_distribution.png",
        "brightness_distribution.png",
        "contrast_distribution.png",
        "blur_distribution.png",
        "representative_samples.png",
    ):
        lines.append(f"- [{filename}](./{filename})")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_reports(
    profile: dict[str, Any],
    manifest: DatasetManifest,
    quality_results: dict[str, dict[str, Any]],
    output_root: Path,
) -> Path:
    """Write all required report artifacts in a new, non-overwriting run folder."""
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", manifest.name).strip("._")
    if not safe_name:
        raise ValueError(f"Dataset name cannot be used as an output directory: {manifest.name!r}")
    dataset_output_root = output_root / safe_name
    run_directory, run_id = _new_run_directory(dataset_output_root)
    report_profile = {
        **profile,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
    }
    with (run_directory / "dataset_profile.json").open("w", encoding="utf-8") as target:
        json.dump(report_profile, target, indent=2, ensure_ascii=False, allow_nan=False)

    quality_columns = [
        "source_image_id",
        "filename",
        "image_path",
        "valid",
        "error",
        "width",
        "height",
        "aspect_ratio",
        "format",
        "source_mode",
        "color_category",
        "file_size_bytes",
    ]
    for metric_name in (
        "blur",
        "grayscale_standard_deviation",
        "local_contrast",
        "brightness",
        "illumination",
        "noise",
        "skew_angle_degrees",
    ):
        quality_columns.extend(
            [
                f"{metric_name}_value",
                f"{metric_name}_interpretation",
                f"{metric_name}_confidence",
                f"{metric_name}_warnings",
            ]
        )
    quality_columns.extend(
        ["brightness_standard_deviation", "brightness_distribution_16_bins", "measurement_warnings"]
    )
    pd.DataFrame(
        _quality_csv_rows(manifest, quality_results), columns=quality_columns
    ).to_csv(run_directory / "image_quality.csv", index=False)
    annotation_columns = [
        "source_image_id",
        "filename",
        "image_path",
        "annotation_id",
        "parent_annotation_id",
        "granularity",
        "transcription",
        "transcription_length_characters",
        "language",
        "script",
        "bbox",
        "polygon",
        "polygons",
        "board_type",
        "split",
        "crop_width",
        "crop_height",
        "metadata",
    ]
    pd.DataFrame(
        _annotation_csv_rows(manifest), columns=annotation_columns
    ).to_csv(run_directory / "annotation_profile.csv", index=False)
    _write_visualizations(manifest, quality_results, run_directory)
    comparisons = _write_comparison(output_root, report_profile, run_directory)
    _write_summary(
        report_profile, comparisons, run_directory / "dataset_summary.md"
    )
    return run_directory
