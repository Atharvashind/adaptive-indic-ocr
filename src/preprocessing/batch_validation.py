"""Reproducible, source-image-safe batch validation for adaptive decisions."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError

from src.ingestion.base_dataset import DatasetImage, DatasetManifest
from src.preprocessing.adaptive_selector import AdaptivePreprocessorSelector
from src.profiling.domain_profiler import SUPPORTED_DOMAINS
from src.run_profiling import _allocate_sample_counts, _image_language, _image_split
from src.quality.quality_analyzer import QualityAnalyzer


def _annotation_value(image: DatasetImage, field: str) -> str | None:
    for annotation in image.annotations:
        if field == "language" and annotation.language:
            return str(annotation.language)
        if field == "script" and annotation.script:
            return str(annotation.script)
        value = (annotation.metadata or {}).get(field)
        if value not in (None, ""):
            return str(value)
    return None


def _source_group(image: DatasetImage) -> str:
    return image.source_image_id or f"image:{image.relative_path.casefold()}"


def sample_manifest_by_source(
    manifest: DatasetManifest, sample_limit: int, seed: int
) -> DatasetManifest:
    """Sample at most one line crop per source image, stratified by split/language."""
    if sample_limit <= 0:
        raise ValueError("sample_limit must be greater than zero")
    rng = random.Random(seed)
    grouped_images: dict[str, list[DatasetImage]] = defaultdict(list)
    for image in manifest.images:
        grouped_images[_source_group(image)].append(image)

    strata: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for source_id, images in grouped_images.items():
        representative = min(images, key=lambda item: item.relative_path.casefold())
        stratum = f"{_image_split(representative)}|{_image_language(representative)}"
        strata[stratum]["source_ids"].append(source_id)

    stratum_lists = {
        key: sorted(values["source_ids"])
        for key, values in strata.items()
    }
    allocations = _allocate_sample_counts(
        min(sample_limit, len(grouped_images)),
        stratum_lists,
        ensure_each=sample_limit >= len(stratum_lists),
    )
    selected: list[DatasetImage] = []
    for stratum in sorted(stratum_lists):
        source_ids = stratum_lists[stratum]
        for source_id in rng.sample(source_ids, allocations[stratum]):
            selected.append(rng.choice(grouped_images[source_id]))
    selected.sort(key=lambda item: item.relative_path.casefold())

    sampled = replace(
        manifest,
        images=selected,
        annotations=[
            annotation
            for image in selected
            for annotation in image.annotations
        ],
        sample_limit=sample_limit,
        sample_seed=seed,
    )
    return sampled


def _load_array(path: Path) -> tuple[np.ndarray, str]:
    with Image.open(path) as source:
        if source.mode in {"1", "L", "I", "F"}:
            return np.asarray(source.convert("L"), dtype=np.uint8).copy(), "RGB"
        if source.mode == "RGBA":
            return np.asarray(source, dtype=np.uint8).copy(), "RGB"
        return np.asarray(source.convert("RGB"), dtype=np.uint8).copy(), "RGB"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_run_directory(root: Path, dataset: str) -> Path:
    """Create a unique UTC timestamped run directory without overwriting."""
    dataset_root = root / dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = dataset_root / f"run-{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = dataset_root / f"run-{timestamp}-{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                + "\n"
            )


def _csv_write(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False, allow_nan=False)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _failed_decision(
    image: DatasetImage, mode: str, domain: str, error: str
) -> dict[str, Any]:
    return {
        "image_identifier": image.source_image_id or image.path.name,
        "operating_mode": mode,
        "domain_metadata": {
            "selected_domain": domain,
            "user_specified_domain": domain if mode == "user_guided" else None,
            "source_domain": None,
            "metadata_domain": None,
            "domain_labels": [],
            "domain_source": "user" if mode == "user_guided" else "metadata",
        },
        "measured_quality_values": {},
        "selected_operation": "keep_original",
        "operation_parameters": {},
        "rejected_candidates": [],
        "rule_evaluations": [],
        "reasons": ["Quality analysis failed; source image was not changed."],
        "confidence": 0.0,
        "uncertainty": ["Quality measurements are unavailable because image analysis failed."],
        "rule_version": AdaptivePreprocessorSelector.RULE_VERSION,
        "configuration_version": "unavailable",
        "evaluation_time_ms": 0.0,
        "output_path": None,
        "warnings": [error],
        "status": "failure",
        "error": error,
    }


def _language_label(image: DatasetImage) -> str:
    return _annotation_value(image, "language") or "unknown"


def _manifest_rows(manifest: DatasetManifest) -> list[dict[str, Any]]:
    rows = []
    for index, image in enumerate(manifest.images, start=1):
        rows.append(
            {
                "sample_index": index,
                "image_identifier": image.source_image_id or image.path.name,
                "filename": image.path.name,
                "relative_path": image.relative_path,
                "source_path": str(image.path.resolve()),
                "source_image_id": image.source_image_id,
                "language": _language_label(image),
                "script": _annotation_value(image, "script"),
                "board_type": _annotation_value(image, "board_type"),
                "split": _image_split(image),
                "status": "pending",
                "sha256_before": None,
                "sha256_after": None,
                "source_unchanged": None,
            }
        )
    return rows


def _quality_csv_row(
    sample: dict[str, Any],
    profile: dict[str, Any],
    analysis_time_ms: float,
    error: str | None,
) -> dict[str, Any]:
    measurements = profile.get("measurements", {})

    def value(section: str, key: str = "value") -> Any:
        record = measurements.get(section, {})
        return record.get(key) if isinstance(record, dict) else None

    return {
        "sample_index": sample["sample_index"],
        "image_identifier": sample["image_identifier"],
        "filename": sample["filename"],
        "source_image_id": sample["source_image_id"],
        "language": sample["language"],
        "script": sample["script"],
        "split": sample["split"],
        "valid": profile.get("valid", False),
        "error": error,
        "width": profile.get("width"),
        "height": profile.get("height"),
        "aspect_ratio": profile.get("aspect_ratio"),
        "blur_variance_laplacian": value("blur"),
        "grayscale_standard_deviation": value("grayscale_standard_deviation"),
        "local_contrast": value("local_contrast"),
        "brightness_mean": value("brightness"),
        "brightness_standard_deviation": value("brightness", "standard_deviation"),
        "illumination_spatial_variation": value("illumination"),
        "noise_high_frequency_residual": value("noise"),
        "noise_confidence": value("noise", "confidence"),
        "skew_angle_degrees": value("skew_angle_degrees"),
        "skew_confidence": value("skew_angle_degrees", "confidence"),
        "quality_analysis_time_ms": analysis_time_ms,
        "warnings": profile.get("warnings", []),
    }


def _summary(
    manifest: DatasetManifest,
    automatic: list[dict[str, Any]],
    user_guided: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    config_sha256: str,
) -> dict[str, Any]:
    modes = {"automatic": automatic, "user_guided": user_guided}
    operation_counts: dict[str, dict[str, int]] = {}
    reason_counts: dict[str, dict[str, int]] = {}
    confidence_stats: dict[str, dict[str, float | int]] = {}
    confidence_bins: dict[str, dict[str, int]] = {}
    for mode, decisions in modes.items():
        operation_counts[mode] = dict(
            sorted(Counter(row["selected_operation"] for row in decisions).items())
        )
        reasons: Counter[str] = Counter()
        for row in decisions:
            for reason in row.get("reasons", []):
                reasons[reason] += 1
        reason_counts[mode] = dict(reasons.most_common())
        confidences = [
            float(row["confidence"])
            for row in decisions
            if row.get("confidence") is not None
        ]
        confidence_stats[mode] = {
            "count": len(confidences),
            "minimum": min(confidences) if confidences else 0.0,
            "median": statistics.median(confidences) if confidences else 0.0,
            "mean": statistics.fmean(confidences) if confidences else 0.0,
            "maximum": max(confidences) if confidences else 0.0,
        }
        boundaries = (0, 0.2, 0.4, 0.6, 0.8, 1.0000001)
        bins: dict[str, int] = {}
        for lower, upper in zip(boundaries, boundaries[1:]):
            label = f"{lower:.1f}-{min(upper, 1.0):.1f}"
            bins[label] = sum(lower <= score < upper for score in confidences)
        confidence_bins[mode] = bins

    agreements = [
        left.get("selected_operation") == right.get("selected_operation")
        for left, right in zip(automatic, user_guided)
    ]
    rules_triggered = []
    for decision in automatic:
        rules_triggered.append(
            sum(
                "=triggered" in str(item.get("result", ""))
                for item in decision.get("rule_evaluations", [])
            )
        )
    language_counts: dict[str, dict[str, dict[str, int]]] = {}
    for mode, decisions in modes.items():
        by_identifier = {
            row["image_identifier"]: row["selected_operation"] for row in decisions
        }
        language_counts[mode] = {}
        for sample in manifest_rows:
            language = str(sample.get("language") or "unknown")
            operation = by_identifier.get(sample["image_identifier"], "keep_original")
            language_counts[mode].setdefault(language, {})
            language_counts[mode][language][operation] = (
                language_counts[mode][language].get(operation, 0) + 1
            )
    times_by_mode = {
        mode: [
            float(row.get("evaluation_time_ms", 0.0))
            for row in decisions
            if row.get("status", "success") == "success"
        ]
        for mode, decisions in modes.items()
    }
    all_times = [
        float(row["quality_analysis_time_ms"])
        + float(automatic[index].get("evaluation_time_ms", 0.0))
        + float(user_guided[index].get("evaluation_time_ms", 0.0))
        for index, row in enumerate(quality_rows)
    ]
    failed = sum(not bool(row.get("valid")) for row in quality_rows)
    missing_metrics = {
        field: sum(row.get(field) is None for row in quality_rows)
        for field in (
            "blur_variance_laplacian",
            "grayscale_standard_deviation",
            "brightness_mean",
            "illumination_spatial_variation",
            "noise_high_frequency_residual",
            "skew_angle_degrees",
        )
    }
    valid_time = [value for value in all_times if math.isfinite(value)]

    return {
        "dataset": manifest.name,
        "sample_size": len(manifest.images),
        "population_image_count": manifest.discovered_image_count or len(manifest.images),
        "sampling": {
            "method": "stratified_by_original_split_and_language_then_one_image_per_source_image_id",
            "seed": manifest.sample_seed,
            "requested_limit": manifest.sample_limit,
            "selected_count": len(manifest.images),
            "source_image_ids_unique": len(
                {sample["source_image_id"] for sample in manifest_rows if sample["source_image_id"]}
            )
            == len([sample for sample in manifest_rows if sample["source_image_id"]]),
            "source_ids_are_not_repartitioned": True,
            "split_distribution": dict(
                sorted(Counter(row["split"] for row in manifest_rows).items())
            ),
            "language_distribution": dict(
                sorted(Counter(row["language"] for row in manifest_rows).items())
            ),
        },
        "configuration_sha256": config_sha256,
        "operation_recommendation_counts": operation_counts,
        "keep_original_percentage": {
            mode: (
                100.0 * count / len(decisions) if decisions else 0.0
            )
            for mode, decisions in modes.items()
            for count in [
                sum(row["selected_operation"] == "keep_original" for row in decisions)
            ]
        },
        "quality_failures": failed,
        "missing_quality_measurements": missing_metrics,
        "confidence_statistics": confidence_stats,
        "confidence_distribution": confidence_bins,
        "images_with_multiple_triggered_rules": sum(count > 1 for count in rules_triggered),
        "triggered_rule_count_distribution": dict(
            sorted(Counter(rules_triggered).items())
        ),
        "language_recommendation_counts": language_counts,
        "mode_agreement": {
            "matching_images": sum(agreements),
            "total_images": len(agreements),
            "agreement_percentage": (
                100.0 * sum(agreements) / len(agreements) if agreements else 0.0
            ),
        },
        "recommendation_reasons": reason_counts,
        "processing_time_ms_per_image": {
            "count": len(valid_time),
            "mean": statistics.fmean(valid_time) if valid_time else 0.0,
            "median": statistics.median(valid_time) if valid_time else 0.0,
            "maximum": max(valid_time) if valid_time else 0.0,
            "mode_evaluation_time_mean": {
                mode: (
                    statistics.fmean(values) if values else 0.0
                )
                for mode, values in times_by_mode.items()
            },
            "quality_analysis_time_mean": (
                statistics.fmean(
                    float(row["quality_analysis_time_ms"])
                    for row in quality_rows
                    if row.get("valid")
                )
                if any(row.get("valid") for row in quality_rows)
                else 0.0
            ),
        },
        "source_file_integrity": {
            "checked": len(manifest_rows),
            "unchanged": sum(row.get("source_unchanged") is True for row in manifest_rows),
            "missing_or_unverified": sum(
                row.get("source_unchanged") is not True for row in manifest_rows
            ),
        },
    }


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for mode, counts in summary["operation_recommendation_counts"].items():
        for operation, count in counts.items():
            rows.append(
                {
                    "section": "operation_recommendation",
                    "mode": mode,
                    "category": operation,
                    "count": count,
                    "percentage": (
                        100 * count / summary["sample_size"]
                        if summary["sample_size"]
                        else 0.0
                    ),
                    "value": None,
                }
            )
    for mode, bins in summary["confidence_distribution"].items():
        for bin_name, count in bins.items():
            rows.append(
                {
                    "section": "confidence_distribution",
                    "mode": mode,
                    "category": bin_name,
                    "count": count,
                    "percentage": (
                        100 * count / summary["sample_size"]
                        if summary["sample_size"]
                        else 0.0
                    ),
                    "value": None,
                }
            )
    for mode, languages in summary["language_recommendation_counts"].items():
        for language, counts in languages.items():
            for operation, count in counts.items():
                rows.append(
                    {
                        "section": "language_recommendation",
                        "mode": mode,
                        "category": f"{language}:{operation}",
                        "count": count,
                        "percentage": None,
                        "value": None,
                    }
                )
    for name, value in summary["missing_quality_measurements"].items():
        rows.append(
            {
                "section": "missing_quality_measurements",
                "mode": "",
                "category": name,
                "count": value,
                "percentage": None,
                "value": None,
            }
        )
    _csv_write(
        path,
        rows,
        ["section", "mode", "category", "count", "percentage", "value"],
    )


def _write_visualizations(
    output_directory: Path,
    summary: dict[str, Any],
    automatic: list[dict[str, Any]],
    user_guided: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
) -> list[str]:
    generated: list[str] = []
    modes = {"automatic": automatic, "user_guided": user_guided}

    operations = sorted(
        {
            operation
            for counts in summary["operation_recommendation_counts"].values()
            for operation in counts
        }
    )
    fig, axis = plt.subplots(figsize=(10, 5))
    x = np.arange(len(operations))
    width = 0.38
    for index, (mode, decisions) in enumerate(modes.items()):
        counts = Counter(row["selected_operation"] for row in decisions)
        axis.bar(x + (index - 0.5) * width, [counts[item] for item in operations], width, label=mode)
    axis.set_xticks(x, operations, rotation=30, ha="right")
    axis.set_ylabel("Images")
    axis.set_title("Operation recommendations by mode")
    axis.legend()
    fig.tight_layout()
    filename = "operation_recommendation_distribution.png"
    fig.savefig(output_directory / filename, dpi=140)
    plt.close(fig)
    generated.append(filename)

    fig, axis = plt.subplots(figsize=(8, 5))
    for mode, decisions in modes.items():
        axis.hist(
            [row["confidence"] for row in decisions],
            bins=np.linspace(0, 1, 11),
            alpha=0.55,
            label=mode,
        )
    axis.set_xlabel("Decision confidence")
    axis.set_ylabel("Images")
    axis.set_title("Decision confidence distribution")
    axis.legend()
    fig.tight_layout()
    filename = "confidence_distribution.png"
    fig.savefig(output_directory / filename, dpi=140)
    plt.close(fig)
    generated.append(filename)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    automatic_by_id = {row["image_identifier"]: row for row in automatic}
    fields = (
        ("blur_variance_laplacian", "Variance of Laplacian"),
        ("grayscale_standard_deviation", "Grayscale standard deviation"),
        ("noise_high_frequency_residual", "High-frequency residual"),
    )
    for axis, (field, title) in zip(axes, fields):
        values_by_operation: dict[str, list[float]] = defaultdict(list)
        quality_by_id = {row["image_identifier"]: row for row in quality_rows}
        for image_id, decision in automatic_by_id.items():
            value = quality_by_id.get(image_id, {}).get(field)
            if value is not None and math.isfinite(float(value)):
                values_by_operation[decision["selected_operation"]].append(float(value))
        labels = sorted(values_by_operation)
        if labels:
            axis.boxplot(
                [values_by_operation[label] for label in labels], tick_labels=labels
            )
            axis.tick_params(axis="x", labelrotation=35)
        axis.set_title(title)
        axis.set_ylabel("Measured value")
    fig.suptitle("Quality measurements by automatic recommendation")
    fig.tight_layout()
    filename = "quality_metrics_by_selected_operation.png"
    fig.savefig(output_directory / filename, dpi=140)
    plt.close(fig)
    generated.append(filename)

    language_counts = summary["language_recommendation_counts"]
    languages = sorted(
        {
            language
            for mode_data in language_counts.values()
            for language in mode_data
            if language != "unknown"
        }
    )
    if languages:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        operations_lang = sorted(
            {
                operation
                for mode_data in language_counts.values()
                for counts in mode_data.values()
                for operation in counts
            }
        )
        for axis, (mode, _) in zip(axes, modes.items()):
            bottom = np.zeros(len(languages))
            for operation in operations_lang:
                heights = [
                    language_counts[mode].get(language, {}).get(operation, 0)
                    for language in languages
                ]
                axis.bar(languages, heights, bottom=bottom, label=operation)
                bottom += np.asarray(heights)
            axis.set_title(f"Language recommendations: {mode}")
            axis.set_ylabel("Images")
            axis.legend(fontsize="small")
        fig.tight_layout()
        filename = "language_recommendation_distribution.png"
        fig.savefig(output_directory / filename, dpi=140)
        plt.close(fig)
        generated.append(filename)

    matches = summary["mode_agreement"]["matching_images"]
    total = summary["mode_agreement"]["total_images"]
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.bar(["Agree", "Differ"], [matches, total - matches], color=["#4c956c", "#d1495b"])
    axis.set_ylabel("Images")
    axis.set_title("Automatic vs user-guided agreement")
    fig.tight_layout()
    filename = "mode_agreement.png"
    fig.savefig(output_directory / filename, dpi=140)
    plt.close(fig)
    generated.append(filename)
    return generated


def _write_report(
    path: Path,
    summary: dict[str, Any],
    failed_samples: list[dict[str, Any]],
    visualization_files: list[str],
) -> None:
    lines = [
        "# Adaptive selector batch validation",
        "",
        f"- Dataset: `{summary['dataset']}`",
        f"- Images evaluated: {summary['sample_size']}",
        f"- Sampling: {summary['sampling']['method']}; seed={summary['sampling']['seed']}",
        f"- Source IDs are unique in sample: {summary['sampling']['source_image_ids_unique']}",
        f"- Configuration SHA-256: `{summary['configuration_sha256']}`",
        f"- Failed quality analyses: {summary['quality_failures']}",
        f"- Source file integrity: {summary['source_file_integrity']['unchanged']}/"
        f"{summary['source_file_integrity']['checked']} unchanged",
        "",
        "## Recommendations",
        "",
    ]
    for mode, counts in summary["operation_recommendation_counts"].items():
        keep_pct = summary["keep_original_percentage"][mode]
        lines.append(f"### {mode}")
        lines.append(f"- `keep_original`: {keep_pct:.2f}%")
        lines.extend(f"- `{name}`: {count}" for name, count in counts.items())
        lines.append("")
    lines.extend(
        [
            "## Measurement coverage and confidence",
            "",
            f"- Missing measurements: `{json.dumps(summary['missing_quality_measurements'], sort_keys=True)}`",
            f"- Confidence statistics: `{json.dumps(summary['confidence_statistics'], sort_keys=True)}`",
            f"- Images with more than one triggered rule: {summary['images_with_multiple_triggered_rules']}",
            f"- Automatic/user-guided agreement: {summary['mode_agreement']['matching_images']}/"
            f"{summary['mode_agreement']['total_images']} "
            f"({summary['mode_agreement']['agreement_percentage']:.2f}%)",
            "",
            "## Processing time",
            "",
            f"- Mean total measurement plus two selection evaluations: "
            f"{summary['processing_time_ms_per_image']['mean']:.3f} ms/image",
            f"- Median: {summary['processing_time_ms_per_image']['median']:.3f} ms/image",
            "",
            "## Reasons",
            "",
        ]
    )
    for mode, reasons in summary["recommendation_reasons"].items():
        lines.append(f"### {mode}")
        lines.extend(f"- {reason}: {count}" for reason, count in reasons.items())
        lines.append("")
    lines.extend(["## Language-wise recommendations", ""])
    for mode, languages in summary["language_recommendation_counts"].items():
        lines.append(f"### {mode}")
        for language, counts in languages.items():
            lines.append(f"- `{language}`: {json.dumps(counts, sort_keys=True)}")
        lines.append("")
    lines.extend(["## Failed samples", ""])
    if failed_samples:
        lines.extend(
            f"- `{row['filename']}` (`{row['source_path']}`): {row['error']}"
            for row in failed_samples
        )
    else:
        lines.append("None.")
    lines.extend(["", "## Visualizations", ""])
    lines.extend(f"- [{name}]({name})" for name in visualization_files)
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "Recommendations are produced by provisional configurable rules, not a trained selector. "
            "The sample preserves original split labels and selects at most one line crop per "
            "source image ID; it does not repartition data. Noise and skew metrics have known "
            "content-dependent limitations. No preprocessing operation was executed and no OCR "
            "improvement is claimed. Wall-clock timings vary between runs even when sampling, "
            "measurements, and recommendations are deterministic.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


class BatchAdaptiveValidation:
    """Evaluate two recommendation modes and write reproducible artifacts."""

    def __init__(
        self,
        selector: AdaptivePreprocessorSelector,
        quality_analyzer: QualityAnalyzer,
        user_domain: str = "scene_text",
    ) -> None:
        if user_domain not in SUPPORTED_DOMAINS:
            raise ValueError(f"unsupported domain: {user_domain}")
        self.selector = selector
        self.quality_analyzer = quality_analyzer
        self.user_domain = user_domain

    def run(self, manifest: DatasetManifest, output_directory: Path) -> dict[str, Any]:
        output_directory.mkdir(parents=True, exist_ok=True)
        manifest_rows = _manifest_rows(manifest)
        before_hashes: dict[str, str | None] = {}
        for row in manifest_rows:
            path = Path(row["source_path"])
            try:
                before_hashes[row["relative_path"]] = _sha256(path)
                row["sha256_before"] = before_hashes[row["relative_path"]]
            except OSError:
                before_hashes[row["relative_path"]] = None

        automatic: list[dict[str, Any]] = []
        user_guided: list[dict[str, Any]] = []
        quality_rows: list[dict[str, Any]] = []
        failed_samples: list[dict[str, Any]] = []
        sample_by_id: dict[str, dict[str, Any]] = {}
        for row in manifest_rows:
            sample_by_id[row["image_identifier"]] = row
            image = next(
                item for item in manifest.images
                if item.relative_path == row["relative_path"]
            )
            analysis_started = time.perf_counter()
            error: str | None = None
            try:
                pixels, color_order = _load_array(image.path)
                quality = self.quality_analyzer.analyze_array(
                    pixels,
                    image_identifier=row["image_identifier"],
                    color_order=color_order,
                )
                quality["path"] = str(image.path)
            except (OSError, ValueError, UnidentifiedImageError) as caught:
                error = f"Unable to decode or measure image: {caught}"
                quality = {
                    "path": str(image.path),
                    "valid": False,
                    "error": error,
                    "warnings": [error],
                    "measurements": {},
                }
            analysis_time = (time.perf_counter() - analysis_started) * 1000
            row["status"] = "success" if error is None else "failure"
            quality_rows.append(
                _quality_csv_row(row, quality, analysis_time, error)
            )

            if error is None:
                metadata = {
                    "source_domain": manifest.dataset_metadata.get("source_domain"),
                    "domain_labels": manifest.dataset_metadata.get("domain_labels", []),
                    "language": row["language"],
                    "script": row["script"],
                    "split": row["split"],
                    "board_type": row["board_type"],
                }
                auto_decision = self.selector.evaluate(
                    pixels,
                    image_identifier=row["image_identifier"],
                    mode="automatic",
                    metadata=metadata,
                    quality_profile=quality,
                    color_order=color_order,
                ).to_dict()
                user_decision = self.selector.evaluate(
                    pixels,
                    image_identifier=row["image_identifier"],
                    mode="user_guided",
                    domain=self.user_domain,
                    metadata=metadata,
                    quality_profile=quality,
                    color_order=color_order,
                ).to_dict()
                auto_decision["status"] = "success"
                user_decision["status"] = "success"
            else:
                auto_decision = _failed_decision(
                    image, "automatic", "unknown", error
                )
                user_decision = _failed_decision(
                    image, "user_guided", self.user_domain, error
                )
                failed_samples.append(
                    {
                        "sample_index": row["sample_index"],
                        "filename": row["filename"],
                        "source_path": row["source_path"],
                        "error": error,
                    }
                )
            automatic.append(auto_decision)
            user_guided.append(user_decision)

        for row in manifest_rows:
            path = Path(row["source_path"])
            try:
                after = _sha256(path)
                row["sha256_after"] = after
                row["source_unchanged"] = before_hashes[row["relative_path"]] == after
            except OSError:
                row["source_unchanged"] = False

        _csv_write(
            output_directory / "sample_manifest.csv",
            manifest_rows,
            [
                "sample_index", "image_identifier", "filename", "relative_path",
                "source_path", "source_image_id", "language", "script", "board_type",
                "split", "status", "sha256_before", "sha256_after", "source_unchanged",
            ],
        )
        _jsonl_write(output_directory / "automatic_decisions.jsonl", automatic)
        _jsonl_write(output_directory / "user_guided_decisions.jsonl", user_guided)
        _csv_write(
            output_directory / "quality_measurements.csv",
            quality_rows,
            list(quality_rows[0]) if quality_rows else [
                "sample_index", "image_identifier", "filename", "valid", "error"
            ],
        )
        config_digest = hashlib.sha256(
            json.dumps(
                self.selector.configuration,
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        summary = _summary(
            manifest,
            automatic,
            user_guided,
            quality_rows,
            manifest_rows,
            config_digest,
        )
        (output_directory / "decision_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _write_summary_csv(output_directory / "decision_summary.csv", summary)
        visualizations = _write_visualizations(
            output_directory, summary, automatic, user_guided, quality_rows
        )
        _write_report(
            output_directory / "validation_report.md",
            summary,
            failed_samples,
            visualizations,
        )
        return summary
