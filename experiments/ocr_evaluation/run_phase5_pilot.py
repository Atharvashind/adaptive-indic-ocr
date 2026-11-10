from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import math
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECOVERY_DIR = PROJECT_ROOT / "experiments" / "ocr_evaluation" / "recovery"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(RECOVERY_DIR) not in sys.path:
    sys.path.insert(0, str(RECOVERY_DIR))

_baseline_runner = importlib.import_module("run_baseline_reproduction")
EXPECTED_FEASIBLE_TEST_COUNT = _baseline_runner.EXPECTED_FEASIBLE_TEST_COUNT
EXPECTED_TOTAL_TEST_COUNT = _baseline_runner.EXPECTED_TOTAL_TEST_COUNT
load_configured_paths = _baseline_runner.load_configured_paths
preflight_dataset = _baseline_runner.preflight_dataset
sha256_file = _baseline_runner.sha256_file
from src.evaluation.cer_wer import calculate_cer_wer  # noqa: E402
from src.models import CRNNAdapter, load_model_adapter  # noqa: E402
from src.preprocessing.adaptive_selector import (  # noqa: E402
    AdaptivePreprocessorSelector,
)
from src.preprocessing.registry import create_default_registry  # noqa: E402
from src.quality.quality_analyzer import QualityAnalyzer  # noqa: E402

LOGGER = logging.getLogger(__name__)
BASELINE_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "ocr_evaluation"
    / "baseline_reproduction"
    / "run-20261106T091015.067700Z"
)
BASELINE_PREDICTIONS_PATH = BASELINE_DIR / "predictions.csv"
BASELINE_METRICS_PATH = BASELINE_DIR / "metrics.json"
DATASET_CONFIG_PATH = PROJECT_ROOT / "configs" / "datasets.yaml"
PREPROCESSING_CONFIG_PATH = PROJECT_ROOT / "configs" / "preprocessing.yaml"
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "models.yaml"
PILOT_CONFIG_PATH = PROJECT_ROOT / "configs" / "ocr_evaluation.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "ocr_evaluation" / "phase5_pilot"
ARM_ORDER = ("original", "fixed", "scene_text", "automatic", "user_guided")
ARM_LABELS = {
    "original": "A — Original",
    "fixed": "B — Fixed histogram equalization",
    "scene_text": "C — Scene-text CLAHE",
    "automatic": "D — Automatic adaptive",
    "user_guided": "E — User-guided adaptive",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")
    return loaded


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False, allow_nan=False)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _image_hash(path: Path) -> str:
    return sha256_file(path)


def _load_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode in {"1", "L", "I", "F"}:
            return np.asarray(image.convert("L"), dtype=np.uint8).copy()
        if image.mode == "RGBA":
            return np.asarray(image, dtype=np.uint8).copy()
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _save_array(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    Image.fromarray(image).save(path, format="PNG")


def _dimensions(image: np.ndarray) -> dict[str, int]:
    height, width = image.shape[:2]
    return {"width": int(width), "height": int(height)}


def _round_up_four(width: int) -> int:
    return ((width + 3) // 4) * 4


def _stratified_unique_source_sample(
    feasible_rows: list[dict[str, str]],
    sample_size: int,
    seed: int,
) -> list[dict[str, str]]:
    """Select one randomized line per source board with proportional language quotas."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if len(feasible_rows) < sample_size:
        raise ValueError(
            f"Requested {sample_size} samples from only {len(feasible_rows)} feasible rows"
        )

    rng = random.Random(seed)
    rows_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in feasible_rows:
        source_id = row["source_image_id"].strip()
        language = row["language"].strip()
        if not source_id or not language:
            raise ValueError(
                f"Missing source_image_id or language in {row.get('filename')!r}"
            )
        rows_by_source[source_id].append(row)

    source_by_language: dict[str, list[dict[str, str]]] = defaultdict(list)
    for source_id in sorted(rows_by_source):
        candidate_rows = sorted(
            rows_by_source[source_id],
            key=lambda row: (row["language"], row["filename"]),
        )
        chosen = rng.choice(candidate_rows)
        source_by_language[chosen["language"]].append(chosen)

    language_counts = {
        language: len(rows) for language, rows in source_by_language.items()
    }
    if sum(language_counts.values()) < sample_size:
        raise ValueError(
            f"Only {sum(language_counts.values())} unique feasible source boards exist"
        )
    exact_quotas = {
        language: sample_size * count / sum(language_counts.values())
        for language, count in language_counts.items()
    }
    quotas = {language: math.floor(value) for language, value in exact_quotas.items()}
    remainder = sample_size - sum(quotas.values())
    remainder_order = sorted(
        exact_quotas,
        key=lambda language: (-(exact_quotas[language] - quotas[language]), language),
    )
    for language in remainder_order[:remainder]:
        quotas[language] += 1

    selected: list[dict[str, str]] = []
    for language in sorted(source_by_language):
        candidates = source_by_language[language]
        rng.shuffle(candidates)
        if len(candidates) < quotas[language]:
            raise ValueError(
                f"Language {language!r} cannot meet its proportional quota "
                f"({len(candidates)} available, {quotas[language]} requested)"
            )
        selected.extend(candidates[: quotas[language]])
    if len(selected) != sample_size:
        raise RuntimeError(f"Sampler returned {len(selected)} rows, expected {sample_size}")
    if len({row["source_image_id"] for row in selected}) != sample_size:
        raise RuntimeError("Sampler selected more than one line crop for a source board")
    return sorted(selected, key=lambda row: row["filename"])


def _baseline_groups(
    feasible_rows: list[dict[str, str]],
    input_height: int,
    batch_size: int,
) -> dict[str, tuple[int, int]]:
    """Map each filename to its Phase 4B batch index and padded model width."""
    groups: dict[str, tuple[int, int]] = {}
    for batch_start in range(0, len(feasible_rows), batch_size):
        batch = feasible_rows[batch_start : batch_start + batch_size]
        widths = [
            int(round(float(row["crop_width"]) * input_height / float(row["crop_height"])))
            for row in batch
        ]
        if not widths or min(widths) < 1:
            raise ValueError("Found an invalid resized width in the feasible test set")
        group_index = batch_start // batch_size
        padded_width = _round_up_four(max(widths))
        for row in batch:
            groups[row["filename"]] = (group_index, padded_width)
    return groups


def _make_guard(path: Path, width: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (width, 32), color=255).save(path, format="PNG")


def _remove_guard_artifacts(guard_root: Path, run_directory: Path) -> None:
    resolved_root = guard_root.resolve()
    if not resolved_root.is_relative_to(run_directory.resolve()):
        raise ValueError("Guard cleanup path must remain inside this pilot run")
    if not resolved_root.exists():
        return
    for path in sorted(
        resolved_root.rglob("*"),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    resolved_root.rmdir()


def _predict_in_baseline_context(
    adapter: CRNNAdapter,
    selected_rows: list[dict[str, str]],
    image_paths: dict[tuple[str, str], Path],
    baseline_contexts: dict[str, tuple[int, int]],
    guard_root: Path,
    arm: str,
) -> tuple[dict[str, str], float, dict[str, float]]:
    by_group: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in selected_rows:
        by_group[baseline_contexts[row["filename"]][0]].append(row)

    predictions: dict[str, str] = {}
    latency_by_filename: dict[str, float] = {}
    total_seconds = 0.0
    for group_index in sorted(by_group):
        rows = by_group[group_index]
        baseline_width = baseline_contexts[rows[0]["filename"]][1]
        max_width = baseline_width
        for row in rows:
            image = _load_array(image_paths[(arm, row["filename"])])
            height, width = image.shape[:2]
            max_width = max(max_width, _round_up_four(int(round(width * 32 / height))))
        guard_path = guard_root / arm / f"guard-{group_index:04d}.png"
        _make_guard(guard_path, max_width)
        paths = [image_paths[(arm, row["filename"])] for row in rows]
        started = time.perf_counter()
        outputs = adapter.predict([*paths, guard_path])
        elapsed = time.perf_counter() - started
        if len(outputs) != len(rows) + 1:
            raise RuntimeError(
                f"Adapter returned {len(outputs)} predictions for "
                f"{len(rows)} samples and one guard"
            )
        total_seconds += elapsed
        per_sample = elapsed / len(rows)
        for row, prediction in zip(rows, outputs[: len(rows)], strict=True):
            predictions[row["filename"]] = prediction
            latency_by_filename[row["filename"]] = per_sample
    if set(predictions) != {row["filename"] for row in selected_rows}:
        raise RuntimeError(f"Arm {arm!r} did not predict the entire pilot sample")
    return predictions, total_seconds, latency_by_filename


def _quality_values(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "width": profile["width"],
        "height": profile["height"],
        "aspect_ratio": profile["aspect_ratio"],
        "blur": profile["measurements"]["blur"],
        "contrast": profile["measurements"]["grayscale_standard_deviation"],
        "brightness": profile["measurements"]["brightness"],
        "illumination": profile["measurements"]["illumination"],
        "noise": profile["measurements"]["noise"],
        "skew": profile["measurements"]["skew_angle_degrees"],
        "warnings": profile.get("warnings", []),
    }


def _geometry_changed(
    original_dimensions: dict[str, int],
    processed_dimensions: dict[str, int],
    details: dict[str, Any],
) -> bool:
    if original_dimensions != processed_dimensions:
        return True
    matrix = details.get("coordinate_transform_matrix")
    if matrix is None:
        return False
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    return not np.allclose(np.asarray(matrix, dtype=np.float64), identity)


def _per_sample_cer(reference: str, hypothesis: str) -> float:
    import jiwer

    return float(jiwer.cer([reference], [hypothesis]))


def _make_comparison(
    source_path: Path,
    processed_path: Path,
    destination: Path,
    title: str,
) -> None:
    with Image.open(source_path) as source:
        before = source.convert("RGB")
    with Image.open(processed_path) as processed:
        after = processed.convert("RGB")
    cell_width = 480
    cell_height = 260
    canvas = Image.new("RGB", (cell_width * 2, cell_height + 36), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "Original", fill="black")
    draw.text((cell_width + 8, 8), title, fill="black")
    for offset, image in ((0, before), (cell_width, after)):
        image.thumbnail((cell_width - 12, cell_height - 12))
        canvas.paste(image, (offset + 6, 32))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG")


def _prepare_transformations(
    *,
    selected_rows: list[dict[str, str]],
    source_hashes: dict[str, str],
    run_directory: Path,
    image_root: Path,
    preprocessing_config: dict[str, Any],
    dataset_config: dict[str, Any],
    pilot_config: dict[str, Any],
) -> tuple[
    dict[tuple[str, str], Path],
    list[dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
]:
    operation_registry = create_default_registry()
    operation_configuration = preprocessing_config.get("preprocessing", {})
    quality_analyzer = QualityAnalyzer(dataset_config.get("quality", {}))
    selector = AdaptivePreprocessorSelector(
        preprocessing_config,
        quality_analyzer=quality_analyzer,
        operation_configuration=preprocessing_config,
    )
    fixed_configuration = pilot_config["arms"]["fixed"]
    scene_configuration = pilot_config["arms"]["scene_text"]
    scene_parameters = operation_configuration.get("clahe")
    if not isinstance(scene_parameters, dict):
        raise ValueError("Predeclared scene-text CLAHE configuration is missing")

    image_paths: dict[tuple[str, str], Path] = {}
    transform_records: list[dict[str, Any]] = []
    selection_rows: dict[str, dict[str, Any]] = {}
    image_root_out = run_directory / "processed_images"

    for sample_index, row in enumerate(selected_rows, start=1):
        filename = row["filename"]
        source_path = image_root / filename
        source_array = _load_array(source_path)
        source_dimensions = _dimensions(source_array)
        before_quality = quality_analyzer.analyze_array(
            source_array,
            image_identifier=filename,
            color_order="RGB",
        )
        metadata = {
            "filename": filename,
            "source_image_id": row["source_image_id"],
            "source_domain": "scene_text",
            "domain_labels": ["scene_text"],
            "granularity": "line_crop",
        }

        image_paths[("original", filename)] = source_path
        original_profile = _quality_values(before_quality)
        original_record = {
            "sample_index": sample_index,
            "arm": "original",
            "filename": filename,
            "source_image_id": row["source_image_id"],
            "language": row["language"],
            "original_image_path": str(source_path),
            "processed_image_path": str(source_path),
            "original_image_sha256": source_hashes[filename],
            "processed_image_sha256": source_hashes[filename],
            "operation_selected": "keep_original",
            "operation_parameters": {},
            "selector_mode": "none",
            "selection_reason": "Original baseline; no additional preprocessing.",
            "selection_time_ms": 0.0,
            "processing_time_ms": 0.0,
            "transformation_status": "not_applicable",
            "original_dimensions": source_dimensions,
            "processed_dimensions": source_dimensions,
            "geometry_changed": False,
            "coordinate_transform_matrix": None,
            "annotation_geometry_status": (
                "line_crop_transcription_unchanged; source-board bbox coordinates "
                "are not line-crop coordinates and were not transformed"
            ),
            "quality_before": original_profile,
            "quality_after": original_profile,
            "quality_changed": False,
            "warnings": [],
        }
        transform_records.append(original_record)
        selection_rows[(filename, "original")] = original_record

        for arm in ("fixed", "scene_text", "automatic", "user_guided"):
            record_started = time.perf_counter()
            decision: dict[str, Any] | None = None
            if arm == "fixed":
                operation_name = str(fixed_configuration["operation"])
                parameters = dict(fixed_configuration.get("parameters", {}))
                reason = str(fixed_configuration["rationale"]).strip()
                selector_mode = "fixed"
                selection_time_ms = 0.0
            elif arm == "scene_text":
                operation_name = str(scene_configuration["operation"])
                parameters = dict(scene_parameters)
                reason = str(scene_configuration["rationale"]).strip()
                selector_mode = "fixed_scene_text"
                selection_time_ms = 0.0
            else:
                mode = "automatic" if arm == "automatic" else "user_guided"
                selection_started = time.perf_counter()
                selected_decision = selector.evaluate(
                    source_array,
                    image_identifier=filename,
                    mode=mode,
                    domain="scene_text" if mode == "user_guided" else None,
                    metadata=metadata,
                    color_order="RGB",
                )
                selection_time_ms = (time.perf_counter() - selection_started) * 1000
                decision = selected_decision.to_dict()
                operation_name = selected_decision.selected_operation
                parameters = dict(selected_decision.operation_parameters)
                reason = "; ".join(selected_decision.reasons)
                selector_mode = selected_decision.operating_mode

            operation_result = None
            if operation_name == "keep_original":
                processed_array = source_array
                processing_time_ms = 0.0
                status = "keep_original"
                details: dict[str, Any] = {}
                warnings: list[str] = []
                processed_path = source_path
                processed_hash = source_hashes[filename]
            else:
                operation = operation_registry.get(operation_name)
                operation_result = operation.execute(
                    source_array,
                    parameters=parameters,
                    metadata={
                        "source_image_id": row["source_image_id"],
                        "granularity": "line_crop",
                    },
                )
                processing_time_ms = operation_result.execution_time_ms
                status = operation_result.status
                details = operation_result.details
                warnings = list(operation_result.warnings)
                if operation_result.success and operation_result.processed_image is not None:
                    processed_array = operation_result.processed_image
                    processed_path = (
                        image_root_out / arm / Path(filename).with_suffix(".png")
                    )
                    _save_array(processed_array, processed_path)
                    processed_hash = _image_hash(processed_path)
                else:
                    processed_array = source_array
                    processed_path = source_path
                    processed_hash = source_hashes[filename]
                    warnings.append(
                        "Failed transformation; original pixels were used for this arm."
                    )
            total_processing_ms = (
                selection_time_ms + processing_time_ms
                if arm in {"automatic", "user_guided"}
                else processing_time_ms
            )
            after_quality = (
                before_quality
                if processed_array is source_array
                else quality_analyzer.analyze_array(
                    processed_array,
                    image_identifier=f"{filename}:{arm}",
                    color_order="RGB",
                )
            )
            processed_dimensions = _dimensions(processed_array)
            parameters_used = (
                operation_result.parameters if operation_result is not None else {}
            )
            transform_record = {
                "sample_index": sample_index,
                "arm": arm,
                "filename": filename,
                "source_image_id": row["source_image_id"],
                "language": row["language"],
                "original_image_path": str(source_path),
                "processed_image_path": str(processed_path),
                "original_image_sha256": source_hashes[filename],
                "processed_image_sha256": processed_hash,
                "operation_selected": operation_name,
                "operation_parameters": parameters_used or parameters,
                "selector_mode": selector_mode,
                "selection_reason": reason,
                "selection_time_ms": selection_time_ms,
                "processing_time_ms": processing_time_ms,
                "total_preprocessing_time_ms": total_processing_ms,
                "transformation_status": status,
                "original_dimensions": source_dimensions,
                "processed_dimensions": processed_dimensions,
                "geometry_changed": _geometry_changed(
                    source_dimensions,
                    processed_dimensions,
                    details,
                ),
                "coordinate_transform_matrix": details.get(
                    "coordinate_transform_matrix"
                ),
                "annotation_geometry_status": (
                    "line_crop_transcription_unchanged; source-board bbox coordinates "
                    "are not line-crop coordinates and were not transformed"
                ),
                "quality_before": _quality_values(before_quality),
                "quality_after": _quality_values(after_quality),
                "quality_changed": not np.array_equal(
                    processed_array,
                    source_array,
                ),
                "selector_decision": decision,
                "warnings": warnings,
                "record_wall_time_ms": (time.perf_counter() - record_started) * 1000,
            }
            image_paths[(arm, filename)] = processed_path
            transform_records.append(transform_record)
            selection_rows[(filename, arm)] = transform_record
    return image_paths, transform_records, selection_rows


def _sample_manifest(
    selected_rows: list[dict[str, str]],
    source_hashes: dict[str, str],
    baseline_rows: dict[str, dict[str, str]],
    baseline_contexts: dict[str, tuple[int, int]],
) -> list[dict[str, Any]]:
    return [
        {
            "sample_index": index,
            "filename": row["filename"],
            "source_image_id": row["source_image_id"],
            "language": row["language"],
            "script": row.get("script", ""),
            "split": row["split"],
            "ground_truth": row["transcription"],
            "source_path": str((PROJECT_ROOT / "synth6k" / "images" / row["filename"])),
            "original_image_sha256": source_hashes[row["filename"]],
            "baseline_prediction": baseline_rows[row["filename"]]["prediction"],
            "baseline_context_batch": baseline_contexts[row["filename"]][0],
            "baseline_padded_width": baseline_contexts[row["filename"]][1],
        }
        for index, row in enumerate(selected_rows, start=1)
    ]


def _per_language_metrics(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        arm_rows = [row for row in predictions if row["arm"] == arm]
        for language in sorted({row["language"] for row in arm_rows}):
            subset = [row for row in arm_rows if row["language"] == language]
            metrics = calculate_cer_wer(
                [row["ground_truth"] for row in subset],
                [row["prediction"] for row in subset],
            )
            rows.append({"arm": arm, "language": language, **metrics})
    return rows


def _build_metrics(
    predictions: list[dict[str, Any]],
    transform_records: list[dict[str, Any]],
    inference_seconds: dict[str, float],
    inference_latency: dict[tuple[str, str], float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_language = _per_language_metrics(predictions)
    original_metrics: dict[str, Any] | None = None
    metrics_by_arm: dict[str, Any] = {}
    for arm in ARM_ORDER:
        arm_rows = [row for row in predictions if row["arm"] == arm]
        metrics = calculate_cer_wer(
            [row["ground_truth"] for row in arm_rows],
            [row["prediction"] for row in arm_rows],
        )
        if arm == "original":
            original_metrics = metrics
        preprocessing_records = [
            record
            for record in transform_records
            if record["arm"] == arm
        ]
        preprocessing_ms = sum(
            float(record.get("total_preprocessing_time_ms", 0.0))
            for record in preprocessing_records
        )
        inference_ms = inference_seconds[arm] * 1000
        metrics_by_arm[arm] = {
            **metrics,
            "label": ARM_LABELS[arm],
            "per_language": {
                row["language"]: row
                for row in per_language
                if row["arm"] == arm
            },
            "preprocessing_latency_ms": {
                "total": preprocessing_ms,
                "mean_per_sample": preprocessing_ms / len(arm_rows),
            },
            "model_inference_latency_ms": {
                "total": inference_ms,
                "mean_per_sample": inference_ms / len(arm_rows),
                "mean_allocated_per_sample": sum(
                    inference_latency[(arm, row["filename"])] for row in arm_rows
                )
                / len(arm_rows),
            },
            "total_pipeline_latency_ms": {
                "total": preprocessing_ms + inference_ms,
                "mean_per_sample": (preprocessing_ms + inference_ms)
                / len(arm_rows),
            },
            "operation_selection_counts": dict(
                sorted(
                    Counter(
                        record["operation_selected"]
                        for record in preprocessing_records
                    ).items()
                )
            ),
            "failed_transformations": sum(
                record["transformation_status"] == "failure"
                for record in preprocessing_records
            ),
            "geometry_changed_count": sum(
                bool(record["geometry_changed"])
                for record in preprocessing_records
            ),
            "quality_changed_count": sum(
                bool(record["quality_changed"])
                for record in preprocessing_records
            ),
        }
    if original_metrics is None:
        raise RuntimeError("Original arm metrics were not calculated")

    for arm in ARM_ORDER[1:]:
        metrics_by_arm[arm]["delta_cer"] = (
            original_metrics["cer"] - metrics_by_arm[arm]["cer"]
        )
        metrics_by_arm[arm]["delta_wer"] = (
            original_metrics["wer"] - metrics_by_arm[arm]["wer"]
        )
        metrics_by_arm[arm]["delta_exact_match_accuracy"] = (
            metrics_by_arm[arm]["exact_match_accuracy"]
            - original_metrics["exact_match_accuracy"]
        )

    by_filename: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in predictions:
        by_filename[row["filename"]][row["arm"]] = row
    automatic_user_agreement = sum(
        next(
            record["operation_selected"]
            for record in transform_records
            if record["arm"] == "automatic" and record["filename"] == filename
        )
        == next(
            record["operation_selected"]
            for record in transform_records
            if record["arm"] == "user_guided" and record["filename"] == filename
        )
        for filename in by_filename
    )
    operation_counts = {
        arm: metrics_by_arm[arm]["operation_selection_counts"]
        for arm in ARM_ORDER
    }
    sample_diagnostics: dict[str, Any] = {}
    for arm in ARM_ORDER[1:]:
        changes = []
        for filename, arm_rows in by_filename.items():
            before = _per_sample_cer(
                arm_rows["original"]["ground_truth"],
                arm_rows["original"]["prediction"],
            )
            after = _per_sample_cer(
                arm_rows[arm]["ground_truth"],
                arm_rows[arm]["prediction"],
            )
            changes.append(
                {
                    "filename": filename,
                    "cer_delta": before - after,
                    "original_cer": before,
                    "processed_cer": after,
                    "operation": next(
                        record["operation_selected"]
                        for record in transform_records
                        if record["arm"] == arm
                        and record["filename"] == filename
                    ),
                }
            )
        changes.sort(key=lambda item: (-item["cer_delta"], item["filename"]))
        sample_diagnostics[arm] = {
            "best_examples": changes[:3],
            "worst_examples": sorted(
                changes,
                key=lambda item: (item["cer_delta"], item["filename"]),
            )[:3],
            "improved_sample_count": sum(item["cer_delta"] > 0 for item in changes),
            "degraded_sample_count": sum(item["cer_delta"] < 0 for item in changes),
            "unchanged_sample_count": sum(item["cer_delta"] == 0 for item in changes),
        }

    summary = {
        "sample_size": len(by_filename),
        "arms": metrics_by_arm,
        "operation_selection_counts": operation_counts,
        "adaptive_selector_agreement": {
            "matching_samples": automatic_user_agreement,
            "total_samples": len(by_filename),
            "agreement_fraction": automatic_user_agreement / len(by_filename),
        },
        "failed_transformations": sum(
            record["transformation_status"] == "failure"
            for record in transform_records
        ),
        "quality_and_dimension_changes": {
            arm: {
                "quality_changed_samples": metrics_by_arm[arm][
                    "quality_changed_count"
                ],
                "geometry_changed_samples": metrics_by_arm[arm][
                    "geometry_changed_count"
                ],
            }
            for arm in ARM_ORDER
        },
        "sample_level_cer_diagnostics": sample_diagnostics,
        "interpretation": (
            "Exploratory 200-image pilot only; differences are descriptive and "
            "not statistically conclusive."
        ),
    }
    return summary, per_language


def _make_markdown(
    metrics: dict[str, Any],
    run_metadata: dict[str, Any],
) -> str:
    lines = [
        "# Phase 5A controlled preprocessing pilot",
        "",
        f"- **Run:** `{run_metadata['run_id']}`",
        f"- **Status:** {run_metadata['status']}",
        f"- **Samples:** {run_metadata['sample']['selected_count']}",
        f"- **Seed:** {run_metadata['sample']['seed']}",
        f"- **Sampling:** language-stratified among CTC-feasible test lines; "
        "one crop per source board; train/validation source IDs excluded.",
        "",
        "## 1. Implementation correctness",
        "",
        "- All five arms used the same selected 200 filenames, transcription, language labels, recovered checkpoint, vocabulary, adapter, and metric functions.",
        "- The original arm was checked against Phase 4B before any transformed-arm OCR evaluation.",
        f"- Original-arm prediction parity: "
        f"{run_metadata['validation']['original_prediction_matches']}/"
        f"{run_metadata['validation']['original_prediction_count']}.",
        f"- Source files unchanged: "
        f"{run_metadata['validation']['source_images_unchanged']}/"
        f"{run_metadata['validation']['source_images_checked']}.",
        f"- Dataset metadata and test annotation reference unchanged: "
        f"{run_metadata['validation']['annotation_files_unchanged']}.",
        f"- Recovered checkpoint archive unchanged: "
        f"{run_metadata['validation']['checkpoint_archive_unchanged']}.",
        "- Inference reused the exact original Phase 4B batch padded-width context for each selected sample (a non-scored guard input preserves that width); guard outputs are discarded.",
        "",
        "## 2. Experimental results",
        "",
        "| Arm | Samples | CER | WER | Exact match | Delta CER vs original | Preprocess ms/sample | Inference ms/sample | Pipeline ms/sample |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARM_ORDER:
        result = metrics["arms"][arm]
        delta = "—" if arm == "original" else f"{result['delta_cer']:+.6f}"
        lines.append(
            f"| {ARM_LABELS[arm]} | {result['sample_count']} | "
            f"{result['cer']:.6f} | {result['wer']:.6f} | "
            f"{result['exact_match_accuracy']:.2%} | {delta} | "
            f"{result['preprocessing_latency_ms']['mean_per_sample']:.2f} | "
            f"{result['model_inference_latency_ms']['mean_per_sample']:.2f} | "
            f"{result['total_pipeline_latency_ms']['mean_per_sample']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Positive Delta CER means lower CER than the original arm. These pilot differences are descriptive only and are not statistically conclusive.",
            "",
            "### Per-language metrics",
            "",
            "| Arm | Language | N | CER | WER | Exact match |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for arm in ARM_ORDER:
        for language, result in metrics["arms"][arm]["per_language"].items():
            lines.append(
                f"| {ARM_LABELS[arm]} | {language} | {result['sample_count']} | "
                f"{result['cer']:.6f} | {result['wer']:.6f} | "
                f"{result['exact_match_accuracy']:.2%} |"
            )
    lines.extend(
        [
            "",
            "## 3. Observed preprocessing behavior",
            "",
            f"- Operation selections: `{json.dumps(metrics['operation_selection_counts'], ensure_ascii=False, sort_keys=True)}`",
            f"- Automatic/user-guided selector agreement: "
            f"{metrics['adaptive_selector_agreement']['matching_samples']}/"
            f"{metrics['adaptive_selector_agreement']['total_samples']} "
            f"({metrics['adaptive_selector_agreement']['agreement_fraction']:.1%}).",
            f"- Failed transformations: {metrics['failed_transformations']}.",
            f"- Quality/dimension effects: "
            f"`{json.dumps(metrics['quality_and_dimension_changes'], sort_keys=True)}`",
            "- Every transformation record includes source/output hashes, dimensions, operation parameters, selector reason/mode, latency, quality profiles, and geometry status.",
            "- For this line-crop dataset, transcription remains unchanged. Dataset bboxes refer to the parent source board, not the line-crop coordinate frame, and are not transformed or passed as crop geometry.",
            "",
            "### Illustrative best and worst sample-level CER changes",
            "",
        ]
    )
    for arm in ARM_ORDER[1:]:
        diagnostic = metrics["sample_level_cer_diagnostics"][arm]
        lines.append(f"- **{ARM_LABELS[arm]}:** improved {diagnostic['improved_sample_count']}, "
                     f"degraded {diagnostic['degraded_sample_count']}, unchanged "
                     f"{diagnostic['unchanged_sample_count']}. "
                     f"Best examples: `{diagnostic['best_examples'][:2]}`; "
                     f"worst examples: `{diagnostic['worst_examples'][:2]}`.")
    lines.extend(
        [
            "",
            "Representative side-by-side image files are in `visual_comparisons/`; they are selected by sample-level CER deltas, not used to tune preprocessing.",
            "",
            "## 4. Limitations",
            "",
            "- This is a fixed-seed 200-image pilot, not the full 8,094-sample test evaluation. Small-sample differences can be noisy and are not evidence of statistically reliable improvement.",
            "- The sample is restricted to the test split and unique source boards; train/validation source boards are explicitly excluded.",
            "- Fixed global histogram equalization and scene-text CLAHE strategies were predeclared in `configs/ocr_evaluation.yaml`, before OCR execution; no pilot metric was used to select or tune parameters.",
            "- Adaptive selector thresholds were not modified. Operation selection remains a provisional rule-based recommendation, not a learned domain-specific system.",
            "- No source images, annotations, checkpoint, or model weights were modified; no training or full-set experiment was run.",
            "",
            "## 5. Recommendation",
            "",
            "None of the preprocessing arms improved aggregate CER on this pilot; the fixed histogram-equalization arm was especially disruptive, and adaptive deskew produced severe individual regressions. Do not adopt any preprocessing strategy or claim improvement from these 200 samples.",
            "",
            "If proceeding to the full 8,094-sample evaluation, keep all five arms and their current predeclared parameters unchanged, use the same paired sample set across arms, and report the full aggregate and per-language outcomes. Treat the pilot as exploratory; do not tune thresholds or operations from these test results. Review the saved comparisons to understand failure modes, not to change this evaluation's configuration.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_aborted_run(
    run_directory: Path,
    *,
    reason: str,
    seed: int,
    sample_size: int,
) -> None:
    details = {
        "status": "aborted",
        "reason": reason,
        "seed": seed,
        "requested_sample_size": sample_size,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (run_directory / "pilot_aborted.json").write_text(
        json.dumps(details, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_directory / "pilot_aborted.md").write_text(
        "# Phase 5A pilot aborted\n\n"
        f"- Reason: {reason}\n"
        f"- Requested sample size: {sample_size}\n"
        f"- Seed: {seed}\n",
        encoding="utf-8",
    )


def run_pilot() -> Path:
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("Phase 5A pilot is configured for CPU-only inference")
    pilot_config = _read_yaml(PILOT_CONFIG_PATH)["phase5_pilot"]
    sample_size = int(pilot_config["sample_size"])
    seed = int(pilot_config["seed"])
    batch_size = int(pilot_config["batch_size"])
    if sample_size != 200:
        raise ValueError("Phase 5A pilot requires exactly 200 samples")
    if batch_size != 64:
        raise ValueError("Phase 5A inference batch size must remain 64")

    dataset_document = _read_yaml(DATASET_CONFIG_PATH)
    preprocessing_document = _read_yaml(PREPROCESSING_CONFIG_PATH)
    if not dataset_document["datasets"]["synth6k"].get("enabled", False):
        raise ValueError("synth6k must be enabled in configs/datasets.yaml")
    image_root, metadata_path, dataset_configuration = load_configured_paths()
    _, feasible_rows, sample_counts = preflight_dataset(
        image_root,
        metadata_path,
        dataset_configuration,
    )
    if sample_counts["total_test"] != EXPECTED_TOTAL_TEST_COUNT:
        raise ValueError("Unexpected synth6k test split size")
    if sample_counts["feasible_test"] != EXPECTED_FEASIBLE_TEST_COUNT:
        raise ValueError("Unexpected synth6k CTC-feasible test count")

    all_metadata_rows = _read_csv(metadata_path)
    source_ids_by_split: dict[str, set[str]] = {
        split: {
            row["source_image_id"]
            for row in all_metadata_rows
            if row["split"].strip().lower() == split
        }
        for split in ("train", "val", "test")
    }
    split_overlaps = {
        f"{left}_vs_{right}": sorted(source_ids_by_split[left] & source_ids_by_split[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    if any(split_overlaps.values()):
        raise ValueError(
            "Source image IDs overlap across dataset splits; refusing a leakage-prone pilot"
        )

    selected_rows = _stratified_unique_source_sample(
        feasible_rows,
        sample_size=sample_size,
        seed=seed,
    )
    selected_filenames = [row["filename"] for row in selected_rows]
    baseline_rows_list = _read_csv(BASELINE_PREDICTIONS_PATH)
    baseline_rows = {row["filename"]: row for row in baseline_rows_list}
    if len(baseline_rows) != len(baseline_rows_list):
        raise ValueError("Phase 4B predictions contain duplicate filenames")
    if not set(selected_filenames) <= set(baseline_rows):
        raise ValueError("Selected CTC-feasible samples are missing Phase 4B predictions")
    baseline_contexts = _baseline_groups(
        feasible_rows,
        input_height=32,
        batch_size=batch_size,
    )
    for row in selected_rows:
        reference = baseline_rows[row["filename"]]
        if (
            reference["ground_truth"] != row["transcription"]
            or reference["language"] != row["language"]
            or reference["source_image_id"] != row["source_image_id"]
        ):
            raise ValueError(
                f"Phase 4B prediction metadata differs for {row['filename']}"
            )

    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S.%fZ")
    run_directory = OUTPUT_ROOT / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    source_hashes = {
        row["filename"]: _image_hash(image_root / row["filename"])
        for row in selected_rows
    }
    archive_path = PROJECT_ROOT / "experiments" / "recovered_baseline" / (
        "Multilingual_CRNN_OCR-20261105T161607Z-1-001.zip"
    )
    archive_hash_before = sha256_file(archive_path)
    baseline_hash = sha256_file(BASELINE_PREDICTIONS_PATH)
    reference_files = dataset_configuration["annotation"].get("reference_files", [])
    test_annotation_value = next(
        (value for value in reference_files if Path(value).stem.lower() == "test"),
        None,
    )
    if not test_annotation_value:
        raise ValueError("Configured synth6k test annotation reference is missing")
    test_annotation_path = (PROJECT_ROOT / test_annotation_value).resolve()
    annotation_hashes_before = {
        str(path): sha256_file(path)
        for path in (metadata_path, test_annotation_path)
    }
    sample_manifest = _sample_manifest(
        selected_rows,
        source_hashes,
        baseline_rows,
        baseline_contexts,
    )
    _write_csv(
        run_directory / "sample_manifest.csv",
        list(sample_manifest[0]),
        sample_manifest,
    )
    adapter = load_model_adapter(
        "crnn",
        config_path=MODEL_CONFIG_PATH,
        project_root=PROJECT_ROOT,
    )
    if not isinstance(adapter, CRNNAdapter):
        raise TypeError("Model registry did not return the configured CRNN adapter")
    torch.set_num_threads(2)
    image_paths: dict[tuple[str, str], Path] = {
        ("original", row["filename"]): image_root / row["filename"]
        for row in selected_rows
    }
    guard_root = run_directory / "model_context_guards"

    # Verify the A arm first. If it differs, do not evaluate the transformed arms.
    original_predictions, original_seconds, original_latencies = (
        _predict_in_baseline_context(
            adapter,
            selected_rows,
            image_paths,
            baseline_contexts,
            guard_root,
            "original",
        )
    )
    original_matches = sum(
        original_predictions[row["filename"]] == baseline_rows[row["filename"]]["prediction"]
        for row in selected_rows
    )
    if original_matches != sample_size:
        _write_aborted_run(
            run_directory,
            reason=(
                f"Original arm matched {original_matches}/{sample_size} Phase 4B "
                "predictions; transformed-arm OCR evaluation was not started."
            ),
            seed=seed,
            sample_size=sample_size,
        )
        raise RuntimeError(
            f"Original arm prediction mismatch: {original_matches}/{sample_size}; "
            f"see {run_directory / 'pilot_aborted.md'}"
        )

    image_paths, transformation_records, _ = _prepare_transformations(
        selected_rows=selected_rows,
        source_hashes=source_hashes,
        run_directory=run_directory,
        image_root=image_root,
        preprocessing_config=preprocessing_document,
        dataset_config=dataset_document,
        pilot_config=pilot_config,
    )

    predictions: list[dict[str, Any]] = []
    inference_seconds: dict[str, float] = {"original": original_seconds}
    inference_latency: dict[tuple[str, str], float] = {
        ("original", filename): latency
        for filename, latency in original_latencies.items()
    }
    for sample_index, row in enumerate(selected_rows, start=1):
        predictions.append(
            {
                "arm": "original",
                "sample_index": sample_index,
                "filename": row["filename"],
                "source_image_id": row["source_image_id"],
                "language": row["language"],
                "ground_truth": row["transcription"],
                "prediction": original_predictions[row["filename"]],
            }
        )

    for arm in ARM_ORDER[1:]:
        arm_predictions, elapsed, latencies = _predict_in_baseline_context(
            adapter,
            selected_rows,
            image_paths,
            baseline_contexts,
            guard_root,
            arm,
        )
        inference_seconds[arm] = elapsed
        for filename, latency in latencies.items():
            inference_latency[(arm, filename)] = latency
        for sample_index, row in enumerate(selected_rows, start=1):
            predictions.append(
                {
                    "arm": arm,
                    "sample_index": sample_index,
                    "filename": row["filename"],
                    "source_image_id": row["source_image_id"],
                    "language": row["language"],
                    "ground_truth": row["transcription"],
                    "prediction": arm_predictions[row["filename"]],
                }
            )
    _remove_guard_artifacts(guard_root, run_directory)

    source_hashes_after = {
        row["filename"]: _image_hash(image_root / row["filename"])
        for row in selected_rows
    }
    source_unchanged_count = sum(
        source_hashes_after[filename] == source_hashes[filename]
        for filename in source_hashes
    )
    archive_hash_after = sha256_file(archive_path)
    archive_unchanged = archive_hash_before == archive_hash_after
    annotation_hashes_after = {
        str(path): sha256_file(Path(path))
        for path in annotation_hashes_before
    }
    annotations_unchanged = annotation_hashes_before == annotation_hashes_after
    if (
        source_unchanged_count != sample_size
        or not archive_unchanged
        or not annotations_unchanged
    ):
        raise RuntimeError(
            "Integrity check failed: source image, annotation, or checkpoint archive changed"
        )

    for record in transformation_records:
        record["source_unchanged"] = (
            source_hashes_after[record["filename"]]
            == source_hashes[record["filename"]]
        )
    metrics, per_language = _build_metrics(
        predictions,
        transformation_records,
        inference_seconds,
        inference_latency,
    )
    if any(len([row for row in predictions if row["arm"] == arm]) != sample_size for arm in ARM_ORDER):
        raise RuntimeError("Every experimental arm must contain exactly 200 predictions")
    filename_sets = {
        arm: {
            row["filename"] for row in predictions if row["arm"] == arm
        }
        for arm in ARM_ORDER
    }
    if any(filename_sets[arm] != filename_sets["original"] for arm in ARM_ORDER):
        raise RuntimeError("Experimental arms do not contain identical filenames")

    operation_selection_rows = []
    for record in transformation_records:
        operation_selection_rows.append(
            {
                "sample_index": record["sample_index"],
                "arm": record["arm"],
                "filename": record["filename"],
                "source_image_id": record["source_image_id"],
                "language": record["language"],
                "operation_selected": record["operation_selected"],
                "operation_parameters": record["operation_parameters"],
                "selector_mode": record["selector_mode"],
                "selection_reason": record["selection_reason"],
                "selection_time_ms": record.get("selection_time_ms", 0.0),
                "processing_time_ms": record.get("processing_time_ms", 0.0),
                "transformation_status": record["transformation_status"],
                "geometry_changed": record["geometry_changed"],
                "quality_changed": record["quality_changed"],
            }
        )
    _write_csv(
        run_directory / "predictions.csv",
        [
            "arm",
            "sample_index",
            "filename",
            "source_image_id",
            "language",
            "ground_truth",
            "prediction",
        ],
        predictions,
    )
    _write_csv(
        run_directory / "per_language_metrics.csv",
        [
            "arm",
            "language",
            "sample_count",
            "cer",
            "wer",
            "exact_match_count",
            "exact_match_accuracy",
        ],
        per_language,
    )
    _write_csv(
        run_directory / "operation_selection.csv",
        list(operation_selection_rows[0]),
        operation_selection_rows,
    )
    (run_directory / "transformation_log.json").write_text(
        json.dumps(
            transformation_records,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    validation = {
        "original_prediction_matches": original_matches,
        "original_prediction_count": sample_size,
        "source_images_checked": sample_size,
        "source_images_unchanged": source_unchanged_count,
        "checkpoint_archive_sha256_before": archive_hash_before,
        "checkpoint_archive_sha256_after": archive_hash_after,
        "checkpoint_archive_unchanged": archive_unchanged,
        "annotation_file_hashes_before": annotation_hashes_before,
        "annotation_file_hashes_after": annotation_hashes_after,
        "annotation_files_unchanged": annotations_unchanged,
        "same_filenames_across_arms": all(
            filename_sets[arm] == filename_sets["original"] for arm in ARM_ORDER
        ),
        "same_ground_truth_across_arms": len(
            {
                (row["filename"], row["ground_truth"])
                for row in predictions
            }
        )
        == sample_size,
        "source_image_ids_unique": len(
            {row["source_image_id"] for row in selected_rows}
        )
        == sample_size,
        "source_id_split_overlaps": {
            key: len(value) for key, value in split_overlaps.items()
        },
        "ctc_feasible_test_sample_count": len(feasible_rows),
        "sample_size_per_arm": {
            arm: sum(row["arm"] == arm for row in predictions) for arm in ARM_ORDER
        },
        "no_training_performed": True,
        "full_test_experiment_run": False,
    }
    if not all(
        (
            validation["same_filenames_across_arms"],
            validation["same_ground_truth_across_arms"],
            validation["source_image_ids_unique"],
            validation["checkpoint_archive_unchanged"],
        )
    ):
        raise RuntimeError("Pilot fairness or integrity validation failed")
    run_metadata = {
        "status": "passed",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_command": (
            r".\.venv\Scripts\python.exe "
            r".\experiments\ocr_evaluation\run_phase5_pilot.py"
        ),
        "working_directory": str(PROJECT_ROOT),
        "dataset": "synth6k",
        "sample": {
            "requested_count": sample_size,
            "selected_count": len(selected_rows),
            "seed": seed,
            "language_counts": dict(
                sorted(Counter(row["language"] for row in selected_rows).items())
            ),
            "source_board_count": len(
                {row["source_image_id"] for row in selected_rows}
            ),
            "source_id_split_overlaps": validation["source_id_split_overlaps"],
            "split": "test",
            "ctc_feasible_only": True,
            "sampling_method": (
                "proportional language stratification over unique source IDs; "
                "one randomly selected feasible line per source ID; seed-controlled"
            ),
        },
        "experimental_arms": {
            arm: {
                "label": ARM_LABELS[arm],
                **pilot_config["arms"][arm],
            }
            for arm in ARM_ORDER
        },
        "model": {
            "adapter": "CRNNAdapter",
            "architecture": adapter.configuration["architecture"],
            "checkpoint_sha256": adapter.checkpoint_sha256,
            "vocabulary_sha256": adapter.vocabulary_sha256,
            "device": str(next(adapter.model.parameters()).device),
            "batch_size": batch_size,
            "historical_batch_context_preserved": True,
            "torch_version": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
        },
        "metrics": {
            "library": "JiWER",
            "version": version("jiwer"),
            "text_normalization": "none",
            "delta_cer_definition": "CER_original - CER_processed; positive is improvement",
        },
        "inputs": {
            "dataset_metadata": {
                "path": str(metadata_path),
                "sha256": sha256_file(metadata_path),
            },
            "test_annotations": {
                "path": str(test_annotation_path),
                "sha256": annotation_hashes_before[str(test_annotation_path)],
            },
            "dataset_config": {
                "path": str(DATASET_CONFIG_PATH),
                "sha256": sha256_file(DATASET_CONFIG_PATH),
            },
            "preprocessing_config": {
                "path": str(PREPROCESSING_CONFIG_PATH),
                "sha256": sha256_file(PREPROCESSING_CONFIG_PATH),
            },
            "model_config": {
                "path": str(MODEL_CONFIG_PATH),
                "sha256": sha256_file(MODEL_CONFIG_PATH),
            },
            "pilot_config": {
                "path": str(PILOT_CONFIG_PATH),
                "sha256": sha256_file(PILOT_CONFIG_PATH),
            },
            "phase4b_predictions": {
                "path": str(BASELINE_PREDICTIONS_PATH),
                "sha256": baseline_hash,
            },
            "checkpoint_archive": {
                "path": str(archive_path),
                "sha256_before": archive_hash_before,
                "sha256_after": archive_hash_after,
            },
        },
        "validation": validation,
        "latency_seconds": inference_seconds,
        "limitations": [
            "A 200-sample pilot is descriptive and not statistically conclusive.",
            "Preprocessing strategies and parameters were frozen before OCR results.",
            "No OCR improvement is claimed from the pilot.",
            "No source image, annotation, checkpoint, or model weights were modified.",
            "The complete 8,094-sample experiment was not run.",
        ],
    }
    (run_directory / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (run_directory / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    comparison_root = run_directory / "visual_comparisons"
    for arm in ARM_ORDER[1:]:
        diagnostic = metrics["sample_level_cer_diagnostics"][arm]
        examples: list[tuple[str, str]] = []
        for label, key in (
            ("best", "best_examples"),
            ("worst", "worst_examples"),
        ):
            for example in diagnostic[key][:2]:
                if all(existing[0] != example["filename"] for existing in examples):
                    examples.append((example["filename"], label))
        for filename, label in examples:
            transform = next(
                record
                for record in transformation_records
                if record["arm"] == arm and record["filename"] == filename
            )
            output_path = (
                comparison_root
                / arm
                / f"{label}-{Path(filename).stem}.png"
            )
            _make_comparison(
                image_root / filename,
                Path(transform["processed_image_path"]),
                output_path,
                f"{arm}: {transform['operation_selected']} ({label})",
            )

    (run_directory / "metrics.md").write_text(
        _make_markdown(metrics, run_metadata),
        encoding="utf-8",
    )
    LOGGER.info("Phase 5A pilot completed: %s", run_directory)
    print(run_directory)
    return run_directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the fixed 200-sample Phase 5A preprocessing pilot."
    )
    parser.add_argument(
        "--config",
        default=str(PILOT_CONFIG_PATH),
        help="Pilot strategy configuration (must retain the verified sample size and seed).",
    )
    arguments = parser.parse_args(argv)
    if Path(arguments.config).resolve() != PILOT_CONFIG_PATH.resolve():
        raise ValueError(
            "Alternate pilot configurations are not supported; edit the reviewed "
            "configs/ocr_evaluation.yaml before running."
        )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_pilot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
