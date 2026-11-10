from __future__ import annotations

import csv
import hashlib
import json
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
from PIL import Image, ImageDraw, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ocr_evaluation.phase5b_ablation.diagnostic_adapter import (  # noqa: E402
    ContrastAblationCRNNAdapter,
)
from src.evaluation.cer_wer import calculate_cer_wer  # noqa: E402
from src.models import CRNNAdapter, load_model_adapter  # noqa: E402
from src.preprocessing.registry import create_default_registry  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "configs" / "ocr_evaluation_phase5b.yaml"
DATASET_CONFIG_PATH = PROJECT_ROOT / "configs" / "datasets.yaml"
PREPROCESSING_CONFIG_PATH = PROJECT_ROOT / "configs" / "preprocessing.yaml"
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "models.yaml"
METADATA_PATH = PROJECT_ROOT / "synth6k" / "metadata.csv"
VALIDATION_REFERENCE_PATH = PROJECT_ROOT / "synth6k" / "annotations" / "val.txt"
CHECKPOINT_ARCHIVE_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "recovered_baseline"
    / "Multilingual_CRNN_OCR-20261105T161607Z-1-001.zip"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "ocr_evaluation"
    / "phase5b_ablation"
)
RECOVERY_DIR = PROJECT_ROOT / "experiments" / "ocr_evaluation" / "recovery"
EXPECTED_VALIDATION_TOTAL = 8108
EXPECTED_VALIDATION_FEASIBLE = 8030
ARMS = ("A", "B", "C", "D", "E", "F")
ARM_DETAILS = {
    "A": {
        "label": "Original, native contrast enabled",
        "operation": "keep_original",
        "native_contrast_enabled": True,
    },
    "B": {
        "label": "CLAHE, native contrast enabled",
        "operation": "clahe",
        "native_contrast_enabled": True,
    },
    "C": {
        "label": "Histogram equalization, native contrast enabled",
        "operation": "histogram_equalization",
        "native_contrast_enabled": True,
    },
    "D": {
        "label": "CLAHE, native contrast disabled",
        "operation": "clahe",
        "native_contrast_enabled": False,
    },
    "E": {
        "label": "Histogram equalization, native contrast disabled",
        "operation": "histogram_equalization",
        "native_contrast_enabled": False,
    },
    "F": {
        "label": "Original, native contrast disabled",
        "operation": "keep_original",
        "native_contrast_enabled": False,
    },
}
PAIR_DEFINITIONS = (
    ("A_vs_B", "A", "B", "CLAHE effect with native contrast enabled"),
    (
        "A_vs_C",
        "A",
        "C",
        "Histogram equalization effect with native contrast enabled",
    ),
    ("F_vs_D", "F", "D", "CLAHE effect with native contrast disabled"),
    (
        "F_vs_E",
        "F",
        "E",
        "Histogram equalization effect with native contrast disabled",
    ),
    ("A_vs_F", "A", "F", "Native contrast effect on original images"),
    ("B_vs_D", "B", "D", "Effect of disabling native contrast after CLAHE"),
    (
        "C_vs_E",
        "C",
        "E",
        "Effect of disabling native contrast after histogram equalization",
    ),
)


def _read_yaml(path: Path) -> dict[str, Any]:
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return content


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


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


def _parse_text_reference(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue
            fields = line.split(maxsplit=1)
            if len(fields) != 2 or not fields[1]:
                raise ValueError(f"Malformed reference annotation at {path}:{line_number}")
            filename, transcription = fields
            if filename in records:
                raise ValueError(f"Duplicate validation annotation {filename!r}")
            records[filename] = transcription
    return records


def _is_ctc_feasible(row: dict[str, str], input_height: int) -> bool:
    width = float(row["crop_width"])
    height = float(row["crop_height"])
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
        raise ValueError(f"Invalid annotated dimensions for {row['filename']}")
    resized_width = int(round(width * input_height / height))
    transcription = row["transcription"]
    repeated = sum(a == b for a, b in zip(transcription, transcription[1:]))
    return (resized_width // 4) - 1 >= len(transcription) + repeated


def _load_validation_population(
    image_root: Path,
    dataset_config: dict[str, Any],
    input_height: int,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, set[str]],
    dict[str, Any],
]:
    metadata_rows = _read_csv(METADATA_PATH)
    required = {
        "filename",
        "source_image_id",
        "language",
        "transcription",
        "split",
        "crop_width",
        "crop_height",
    }
    if not metadata_rows or not required <= set(metadata_rows[0]):
        raise ValueError(f"Dataset metadata lacks required fields: {sorted(required)}")
    split_names = {row["split"].strip().lower() for row in metadata_rows}
    if not {"train", "val", "test"} <= split_names:
        raise ValueError("Dataset metadata does not contain train, val, and test splits")

    source_ids_by_split = {
        split: {
            row["source_image_id"]
            for row in metadata_rows
            if row["split"].strip().lower() == split
        }
        for split in ("train", "val", "test")
    }
    overlap_counts = {
        "train_vs_val": len(source_ids_by_split["train"] & source_ids_by_split["val"]),
        "train_vs_test": len(source_ids_by_split["train"] & source_ids_by_split["test"]),
        "val_vs_test": len(source_ids_by_split["val"] & source_ids_by_split["test"]),
    }
    if any(overlap_counts.values()):
        raise ValueError(
            f"Source-image split overlap detected: {overlap_counts}; refusing leakage"
        )

    validation_rows = [
        row for row in metadata_rows if row["split"].strip().lower() == "val"
    ]
    if len(validation_rows) != EXPECTED_VALIDATION_TOTAL:
        raise ValueError(
            f"Expected {EXPECTED_VALIDATION_TOTAL} validation rows, "
            f"found {len(validation_rows)}"
        )
    reference_files = dataset_config.get("annotation", {}).get("reference_files", [])
    reference_value = next(
        (value for value in reference_files if Path(value).stem.lower() == "val"),
        None,
    )
    if not reference_value:
        raise ValueError("Configured validation annotation reference is missing")
    reference_path = (PROJECT_ROOT / reference_value).resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(f"Validation annotation file not found: {reference_path}")
    reference_records = _parse_text_reference(reference_path)
    metadata_records = {
        row["filename"]: row["transcription"] for row in validation_rows
    }
    if reference_records != metadata_records:
        raise ValueError("Validation text reference and dataset metadata disagree")

    feasible_rows: list[dict[str, str]] = []
    seen_filenames: set[str] = set()
    for row in validation_rows:
        for key in ("filename", "source_image_id", "language", "transcription"):
            if not row.get(key):
                raise ValueError(f"Missing {key} in validation row {row.get('filename')!r}")
        if row["filename"] in seen_filenames:
            raise ValueError(f"Duplicate validation filename {row['filename']}")
        seen_filenames.add(row["filename"])
        filename_path = Path(row["filename"])
        if filename_path.is_absolute() or len(filename_path.parts) != 1:
            raise ValueError(f"Unexpected image path in metadata: {row['filename']}")
        image_path = image_root / row["filename"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Validation image missing: {image_path}")
        with Image.open(image_path) as image:
            image.load()
            actual_size = image.size
        expected_size = (int(float(row["crop_width"])), int(float(row["crop_height"])))
        if actual_size != expected_size:
            raise ValueError(
                f"Validation dimensions differ for {row['filename']}: "
                f"{actual_size} != {expected_size}"
            )
        if _is_ctc_feasible(row, input_height):
            feasible_rows.append(row)
    if len(feasible_rows) != EXPECTED_VALIDATION_FEASIBLE:
        raise ValueError(
            f"Expected {EXPECTED_VALIDATION_FEASIBLE} CTC-feasible validation rows, "
            f"found {len(feasible_rows)}"
        )
    return (
        metadata_rows,
        feasible_rows,
        source_ids_by_split,
        {
            "validation_total": len(validation_rows),
            "validation_feasible": len(feasible_rows),
            "validation_infeasible": len(validation_rows) - len(feasible_rows),
            "validation_reference_path": str(reference_path),
            "validation_reference_sha256": _sha256_file(reference_path),
            "source_id_split_overlap_counts": overlap_counts,
        },
    )


def _sample_validation_rows(
    feasible_rows: list[dict[str, str]],
    sample_size: int,
    seed: int,
) -> list[dict[str, str]]:
    if len(feasible_rows) < sample_size:
        raise ValueError(
            f"Only {len(feasible_rows)} feasible validation rows are available"
        )
    rng = random.Random(seed)
    rows_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in feasible_rows:
        rows_by_source[row["source_image_id"]].append(row)

    one_row_per_source: list[dict[str, str]] = []
    for source_id in sorted(rows_by_source):
        candidates = sorted(
            rows_by_source[source_id],
            key=lambda row: (row["language"], row["filename"]),
        )
        languages = {row["language"] for row in candidates}
        if len(languages) != 1:
            raise ValueError(
                f"Source image {source_id!r} maps to multiple languages: {languages}"
            )
        one_row_per_source.append(rng.choice(candidates))

    by_language: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in one_row_per_source:
        by_language[row["language"]].append(row)
    exact_quotas = {
        language: sample_size * len(rows) / len(one_row_per_source)
        for language, rows in by_language.items()
    }
    quotas = {language: math.floor(count) for language, count in exact_quotas.items()}
    remainder = sample_size - sum(quotas.values())
    for language in sorted(
        exact_quotas,
        key=lambda item: (-(exact_quotas[item] - quotas[item]), item),
    )[:remainder]:
        quotas[language] += 1

    selected: list[dict[str, str]] = []
    for language in sorted(by_language):
        candidates = by_language[language]
        rng.shuffle(candidates)
        if len(candidates) < quotas[language]:
            raise ValueError(
                f"Language {language!r} cannot meet validation quota "
                f"{quotas[language]} ({len(candidates)} available)"
            )
        selected.extend(candidates[: quotas[language]])
    if len(selected) != sample_size:
        raise RuntimeError(f"Sampler returned {len(selected)} rows, expected {sample_size}")
    if len({row["source_image_id"] for row in selected}) != sample_size:
        raise RuntimeError("Validation sample contains duplicate source-image IDs")
    if any(row["split"].strip().lower() != "val" for row in selected):
        raise RuntimeError("Non-validation row selected")
    return sorted(selected, key=lambda row: row["filename"])


def _process_operations(
    selected_rows: list[dict[str, str]],
    image_root: Path,
    output_directory: Path,
    preprocessing_config: dict[str, Any],
) -> tuple[dict[str, dict[str, Path]], list[dict[str, Any]]]:
    registry = create_default_registry()
    parameters = preprocessing_config.get("preprocessing", {})
    image_paths: dict[str, dict[str, Path]] = {
        arm: {} for arm in ARMS
    }
    records: list[dict[str, Any]] = []
    for sample_index, row in enumerate(selected_rows, start=1):
        filename = row["filename"]
        source_path = image_root / filename
        original_array = _load_array(source_path)
        image_paths["A"][filename] = source_path
        image_paths["F"][filename] = source_path
        original_dimensions = _dimensions(original_array)
        original_hash = _sha256_file(source_path)
        for arm, operation_name in (
            ("B", "clahe"),
            ("C", "histogram_equalization"),
        ):
            operation_parameters = dict(parameters.get(operation_name, {}))
            operation_started = time.perf_counter()
            result = registry.get(operation_name).execute(
                original_array,
                parameters=operation_parameters,
                metadata={
                    "source_image_id": row["source_image_id"],
                    "granularity": "line_crop",
                },
            )
            wall_time_ms = (time.perf_counter() - operation_started) * 1000
            if not result.success or result.processed_image is None:
                raise RuntimeError(
                    f"{operation_name} failed for {filename}: {result.warnings}"
                )
            processed = result.processed_image
            processed_dimensions = _dimensions(processed)
            if processed_dimensions != original_dimensions:
                raise RuntimeError(
                    f"{operation_name} changed geometry for {filename}; "
                    "the ablation requires pixel-only contrast operations"
                )
            output_path = (
                output_directory
                / "processed_images"
                / operation_name
                / Path(filename).with_suffix(".png")
            )
            _save_array(processed, output_path)
            output_hash = _sha256_file(output_path)
            arm_pair = ("B", "D") if arm == "B" else ("C", "E")
            for paired_arm in arm_pair:
                image_paths[paired_arm][filename] = output_path
                records.append(
                    {
                        "sample_index": sample_index,
                        "arm": paired_arm,
                        "filename": filename,
                        "source_image_id": row["source_image_id"],
                        "language": row["language"],
                        "original_image_sha256": original_hash,
                        "processed_image_sha256": output_hash,
                        "operation": operation_name,
                        "operation_parameters": result.parameters,
                        "selector_mode": "fixed_ablation",
                        "selection_reason": (
                            "Predeclared Phase 5B external contrast operation; "
                            "no adaptive selection."
                        ),
                        "native_contrast_enabled": ARM_DETAILS[paired_arm][
                            "native_contrast_enabled"
                        ],
                        "native_contrast_factor": 2.0
                        if ARM_DETAILS[paired_arm]["native_contrast_enabled"]
                        else None,
                        "operation_execution_time_ms": result.execution_time_ms,
                        "operation_wall_time_ms": wall_time_ms,
                        "transformation_status": result.status,
                        "original_dimensions": original_dimensions,
                        "processed_dimensions": processed_dimensions,
                        "geometry_changed": False,
                        "coordinate_transform_matrix": result.details.get(
                            "coordinate_transform_matrix"
                        ),
                        "annotation_geometry_status": (
                            "line-crop transcription unchanged; source-board boxes "
                            "are not line-crop coordinates"
                        ),
                        "warnings": result.warnings,
                    }
                )
        for arm in ("A", "F"):
            records.append(
                {
                    "sample_index": sample_index,
                    "arm": arm,
                    "filename": filename,
                    "source_image_id": row["source_image_id"],
                    "language": row["language"],
                    "original_image_sha256": original_hash,
                    "processed_image_sha256": original_hash,
                    "operation": "keep_original",
                    "operation_parameters": {},
                    "selector_mode": "none",
                    "selection_reason": "No external preprocessing in this ablation arm.",
                    "native_contrast_enabled": ARM_DETAILS[arm][
                        "native_contrast_enabled"
                    ],
                    "native_contrast_factor": 2.0
                    if ARM_DETAILS[arm]["native_contrast_enabled"]
                    else None,
                    "operation_execution_time_ms": 0.0,
                    "operation_wall_time_ms": 0.0,
                    "transformation_status": "not_applicable",
                    "original_dimensions": original_dimensions,
                    "processed_dimensions": original_dimensions,
                    "geometry_changed": False,
                    "coordinate_transform_matrix": None,
                    "annotation_geometry_status": (
                        "line-crop transcription unchanged; source-board boxes "
                        "are not line-crop coordinates"
                    ),
                    "warnings": [],
                }
            )
    expected_keys = {(arm, row["filename"]) for arm in ARMS for row in selected_rows}
    actual_keys = {
        (record["arm"], record["filename"])
        for record in records
    }
    if expected_keys != actual_keys:
        raise RuntimeError("Transformation log does not cover every sample/arm")
    return image_paths, records


def _predict_arm(
    adapter: ContrastAblationCRNNAdapter,
    selected_rows: list[dict[str, str]],
    image_paths: dict[str, dict[str, Path]],
    arm: str,
    batch_size: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    adapter.native_contrast_enabled = ARM_DETAILS[arm]["native_contrast_enabled"]
    adapter.reset_native_preprocessing_timer()
    predictions: dict[str, str] = {}
    wall_start = time.perf_counter()
    for start in range(0, len(selected_rows), batch_size):
        batch_rows = selected_rows[start : start + batch_size]
        batch_paths = [image_paths[arm][row["filename"]] for row in batch_rows]
        outputs = adapter.predict(batch_paths)
        if len(outputs) != len(batch_rows):
            raise RuntimeError(
                f"Arm {arm} received {len(outputs)} predictions for {len(batch_rows)} rows"
            )
        for row, prediction in zip(batch_rows, outputs, strict=True):
            predictions[row["filename"]] = prediction
    total_wall = time.perf_counter() - wall_start
    native_preprocessing = adapter.native_preprocessing_seconds
    if set(predictions) != {row["filename"] for row in selected_rows}:
        raise RuntimeError(f"Arm {arm} prediction coverage is incomplete")
    return predictions, {
        "prediction_wall_seconds": total_wall,
        "native_preprocessing_seconds": native_preprocessing,
        "inference_seconds_excluding_native_preprocessing": max(
            0.0,
            total_wall - native_preprocessing,
        ),
    }


def _per_sample_cer(reference: str, hypothesis: str) -> float:
    import jiwer

    return float(jiwer.cer([reference], [hypothesis]))


def _compile_predictions(
    selected_rows: list[dict[str, str]],
    predictions_by_arm: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    if set(predictions_by_arm) != set(ARMS):
        raise RuntimeError("Predictions are missing one or more ablation arms")
    baseline_errors = {
        row["filename"]: _per_sample_cer(
            row["transcription"],
            predictions_by_arm["A"][row["filename"]],
        )
        for row in selected_rows
    }
    compiled: list[dict[str, Any]] = []
    for sample_index, row in enumerate(selected_rows, start=1):
        filename = row["filename"]
        for arm in ARMS:
            prediction = predictions_by_arm[arm][filename]
            sample_cer = _per_sample_cer(row["transcription"], prediction)
            compiled.append(
                {
                    "sample_index": sample_index,
                    "arm": arm,
                    "filename": filename,
                    "source_image_id": row["source_image_id"],
                    "language": row["language"],
                    "split": row["split"],
                    "ground_truth": row["transcription"],
                    "prediction": prediction,
                    "per_sample_cer": sample_cer,
                    "delta_cer_vs_A": baseline_errors[filename] - sample_cer,
                }
            )
    return compiled


def _metrics(
    predictions: list[dict[str, Any]],
    transformation_records: list[dict[str, Any]],
    latency: dict[str, dict[str, Any]],
    selected_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    by_arm = {
        arm: [row for row in predictions if row["arm"] == arm]
        for arm in ARMS
    }
    metrics_by_arm: dict[str, dict[str, Any]] = {}
    per_language_rows: list[dict[str, Any]] = []
    cer_by_arm_and_name: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        rows = by_arm[arm]
        if len(rows) != selected_count:
            raise RuntimeError(f"Arm {arm} has {len(rows)} rows; expected {selected_count}")
        metrics = calculate_cer_wer(
            [row["ground_truth"] for row in rows],
            [row["prediction"] for row in rows],
        )
        cer_by_arm_and_name[arm] = {
            row["filename"]: float(row["per_sample_cer"]) for row in rows
        }
        language_metrics: dict[str, dict[str, Any]] = {}
        for language in sorted({row["language"] for row in rows}):
            subset = [row for row in rows if row["language"] == language]
            language_result = calculate_cer_wer(
                [row["ground_truth"] for row in subset],
                [row["prediction"] for row in subset],
            )
            language_metrics[language] = {
                "language": language,
                **language_result,
            }
            per_language_rows.append(
                {"arm": arm, **language_metrics[language]}
            )
        arm_records = [record for record in transformation_records if record["arm"] == arm]
        external_preprocessing_ms = sum(
            float(record["operation_execution_time_ms"]) for record in arm_records
        )
        native_preprocessing_ms = latency[arm]["native_preprocessing_seconds"] * 1000
        inference_ms = (
            latency[arm]["inference_seconds_excluding_native_preprocessing"] * 1000
        )
        preprocessing_total_ms = external_preprocessing_ms + native_preprocessing_ms
        total_pipeline_ms = preprocessing_total_ms + inference_ms
        metrics_by_arm[arm] = {
            **metrics,
            "label": ARM_DETAILS[arm]["label"],
            "native_contrast_enabled": ARM_DETAILS[arm]["native_contrast_enabled"],
            "native_contrast_factor": (
                2.0 if ARM_DETAILS[arm]["native_contrast_enabled"] else None
            ),
            "per_language": language_metrics,
            "per_sample_cer": cer_by_arm_and_name[arm],
            "latency_ms": {
                "external_preprocessing_total": external_preprocessing_ms,
                "crnn_native_preprocessing_total": native_preprocessing_ms,
                "preprocessing_total": preprocessing_total_ms,
                "ocr_inference_total": inference_ms,
                "pipeline_total": total_pipeline_ms,
                "preprocessing_mean_per_sample": preprocessing_total_ms / selected_count,
                "ocr_inference_mean_per_sample": inference_ms / selected_count,
                "pipeline_mean_per_sample": total_pipeline_ms / selected_count,
                "prediction_wall_seconds": latency[arm]["prediction_wall_seconds"],
            },
            "operation_counts": dict(
                Counter(record["operation"] for record in arm_records)
            ),
            "geometry_changed_samples": sum(
                bool(record["geometry_changed"]) for record in arm_records
            ),
            "failed_transformations": sum(
                record["transformation_status"] == "failure"
                for record in arm_records
            ),
        }

    reference = cer_by_arm_and_name["A"]
    for arm in ARMS:
        deltas = [
            reference[name] - error
            for name, error in cer_by_arm_and_name[arm].items()
        ]
        if arm == "A":
            deltas = [0.0] * selected_count
        improved = sum(delta > 0 for delta in deltas)
        unchanged = sum(delta == 0 for delta in deltas)
        degraded = sum(delta < 0 for delta in deltas)
        metrics_by_arm[arm]["paired_delta_vs_A"] = {
            "mean_cer_original_minus_arm": float(np.mean(deltas)),
            "median_cer_original_minus_arm": float(np.median(deltas)),
            "improved_count": improved,
            "improved_percentage": improved / selected_count * 100,
            "unchanged_count": unchanged,
            "unchanged_percentage": unchanged / selected_count * 100,
            "degraded_count": degraded,
            "degraded_percentage": degraded / selected_count * 100,
        }

    interaction_rows: list[dict[str, Any]] = []
    interactions: dict[str, Any] = {}
    for pair_name, baseline_arm, experimental_arm, description in PAIR_DEFINITIONS:
        deltas = [
            cer_by_arm_and_name[baseline_arm][name]
            - cer_by_arm_and_name[experimental_arm][name]
            for name in sorted(cer_by_arm_and_name[baseline_arm])
        ]
        pair_stats = {
            "pair": pair_name,
            "baseline_arm": baseline_arm,
            "experimental_arm": experimental_arm,
            "description": description,
            "mean_paired_delta_cer": float(np.mean(deltas)),
            "median_paired_delta_cer": float(np.median(deltas)),
            "improved_count": sum(delta > 0 for delta in deltas),
            "unchanged_count": sum(delta == 0 for delta in deltas),
            "degraded_count": sum(delta < 0 for delta in deltas),
            "sample_count": len(deltas),
        }
        interactions[pair_name] = pair_stats
        interaction_rows.append(pair_stats)

    clahe_enabled_effect = interactions["A_vs_B"]["mean_paired_delta_cer"]
    clahe_disabled_effect = interactions["F_vs_D"]["mean_paired_delta_cer"]
    hist_enabled_effect = interactions["A_vs_C"]["mean_paired_delta_cer"]
    hist_disabled_effect = interactions["F_vs_E"]["mean_paired_delta_cer"]
    interaction_contrasts = {
        "clahe_disabled_minus_enabled": {
            "value": clahe_disabled_effect - clahe_enabled_effect,
            "interpretation": (
                "Mean paired CLAHE benefit without native contrast minus "
                "mean paired CLAHE benefit with native contrast; positive means "
                "CLAHE was relatively more favorable without native contrast."
            ),
        },
        "histogram_equalization_disabled_minus_enabled": {
            "value": hist_disabled_effect - hist_enabled_effect,
            "interpretation": (
                "Mean paired histogram-equalization benefit without native "
                "contrast minus benefit with native contrast."
            ),
        },
    }
    summary = {
        "sample_count_per_arm": selected_count,
        "arms": metrics_by_arm,
        "paired_interactions": interactions,
        "external_native_contrast_interaction_contrasts": interaction_contrasts,
        "interpretation_warning": (
            "Validation-split diagnostic only; paired descriptive contrasts do "
            "not establish causality or test-set performance."
        ),
    }
    return summary, per_language_rows, interaction_rows


def _image_from_native_tensor(
    adapter: ContrastAblationCRNNAdapter,
    path: Path,
    contrast_enabled: bool,
) -> Image.Image:
    adapter.native_contrast_enabled = contrast_enabled
    tensor, _ = adapter.preprocess_image(path)
    values = tensor.squeeze(0).cpu().numpy()
    pixels = np.clip(np.rint((values * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="L").convert("RGB")


def _save_visual(
    *,
    source_path: Path,
    processed_path: Path,
    enabled_input: Image.Image,
    disabled_input: Image.Image,
    destination: Path,
    title: str,
    ground_truth: str,
    baseline_prediction: str,
    experimental_prediction: str,
    delta: float,
) -> None:
    panels: list[tuple[str, Image.Image]] = []
    with Image.open(source_path) as source:
        panels.append(("Original image", source.convert("RGB")))
    with Image.open(processed_path) as processed:
        panels.append(("External operation", processed.convert("RGB")))
    panels.extend(
        (
            ("CRNN native contrast enabled", enabled_input),
            ("CRNN native contrast disabled", disabled_input),
        )
    )
    panel_width, panel_height = 360, 250
    header_height, footer_height = 28, 140
    canvas = Image.new("RGB", (panel_width * 4, header_height + panel_height + footer_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill="black")
    for index, (label, image) in enumerate(panels):
        x = panel_width * index
        draw.text((x + 5, header_height + 3), label, fill="black")
        fitted = ImageOps.contain(image, (panel_width - 12, panel_height - 12))
        canvas.paste(fitted, (x + 6, header_height + 24))
    text_top = header_height + panel_height
    text_lines = (
        f"Delta CER (A - arm): {delta:+.4f}",
        f"Ground truth (escaped): {ground_truth.encode('unicode_escape').decode('ascii')}",
        "Arm A prediction (escaped): "
        f"{baseline_prediction.encode('unicode_escape').decode('ascii')}",
        "Experimental prediction (escaped): "
        f"{experimental_prediction.encode('unicode_escape').decode('ascii')}",
    )
    y = text_top + 5
    for line in text_lines:
        draw.text((6, y), line[:220], fill="black")
        y += 24
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG")


def _generate_visuals(
    output_directory: Path,
    selected_rows: list[dict[str, str]],
    compiled_predictions: list[dict[str, Any]],
    image_paths: dict[str, dict[str, Path]],
    adapter: ContrastAblationCRNNAdapter,
) -> dict[str, Any]:
    prediction_index = {
        (row["arm"], row["filename"]): row for row in compiled_predictions
    }
    candidate_languages = {"mr", "hi", "bn", "ta", "te"}
    categories = {
        "improved": lambda delta: delta > 0,
        "unchanged": lambda delta: delta == 0,
        "degraded": lambda delta: delta < 0,
        "large_regression": lambda delta: delta <= -0.2,
    }
    generated: list[dict[str, Any]] = []
    for language in sorted(candidate_languages):
        language_rows = [
            row for row in selected_rows if row["language"] == language
        ]
        for category, predicate in categories.items():
            options: list[tuple[float, dict[str, str], str, float]] = []
            for selected in language_rows:
                filename = selected["filename"]
                baseline_cer = float(
                    prediction_index[("A", filename)]["per_sample_cer"]
                )
                for arm in ("B", "C", "D", "E", "F"):
                    result = prediction_index[(arm, filename)]
                    delta = baseline_cer - float(result["per_sample_cer"])
                    if predicate(delta):
                        options.append((delta, selected, arm, delta))
            if not options:
                continue
            if category == "improved":
                _, selected, arm, delta = max(
                    options, key=lambda item: (item[0], item[1]["filename"])
                )
            elif category == "unchanged":
                _, selected, arm, delta = min(
                    options, key=lambda item: (item[1]["filename"], item[2])
                )
            else:
                _, selected, arm, delta = min(
                    options, key=lambda item: (item[0], item[1]["filename"])
                )
            filename = selected["filename"]
            source_path = image_paths["A"][filename]
            processed_path = image_paths[arm][filename]
            enabled_input = _image_from_native_tensor(
                adapter,
                processed_path,
                contrast_enabled=True,
            )
            disabled_input = _image_from_native_tensor(
                adapter,
                processed_path,
                contrast_enabled=False,
            )
            result = prediction_index[(arm, filename)]
            baseline_prediction = prediction_index[("A", filename)]["prediction"]
            destination = (
                output_directory
                / "visual_comparisons"
                / category
                / language
                / f"{arm}-{Path(filename).stem}.png"
            )
            _save_visual(
                source_path=source_path,
                processed_path=processed_path,
                enabled_input=enabled_input,
                disabled_input=disabled_input,
                destination=destination,
                title=f"{category} | {language} | Arm {arm} | {filename}",
                ground_truth=selected["transcription"],
                baseline_prediction=baseline_prediction,
                experimental_prediction=result["prediction"],
                delta=delta,
            )
            generated.append(
                {
                    "category": category,
                    "language": language,
                    "arm": arm,
                    "filename": filename,
                    "delta_cer_vs_A": delta,
                    "path": str(destination),
                }
            )
    return {
        "generated_count": len(generated),
        "examples": generated,
        "languages_requested": sorted(candidate_languages),
        "languages_represented": sorted({item["language"] for item in generated}),
    }


def _markdown(
    run_id: str,
    sample_info: dict[str, Any],
    metrics: dict[str, Any],
    interactions: list[dict[str, Any]],
    visual_info: dict[str, Any],
    parity: dict[str, Any],
) -> str:
    lines = [
        "# Phase 5B CRNN preprocessing interaction and ablation",
        "",
        f"- **Run:** `{run_id}`",
        f"- **Split:** validation only",
        f"- **Sample:** {sample_info['selected_count']} CTC-feasible images; seed {sample_info['seed']}",
        f"- **Language counts:** `{json.dumps(sample_info['language_counts'], sort_keys=True)}`",
        f"- **Unique source boards:** {sample_info['unique_source_count']}",
        f"- **Saved validation baseline predictions:** {parity['saved_validation_baseline_available']}",
        f"- **Original adapter parity on selected validation sample:** "
        f"{parity['original_adapter_predictions_match']}/{parity['sample_count']}",
        "",
        "## 1. Implementation correctness",
        "",
        "- All six arms share the same validation filenames, source IDs, languages, and transcription strings.",
        "- All images are CTC-feasible validation crops; train and test splits are not evaluated. Source-image IDs have zero overlap across the recorded splits.",
        "- Arm A uses the original CRNN adapter with its default contrast factor 2.0. Its predictions match the diagnostic adapter's contrast-enabled path exactly on this sample.",
        "- The diagnostic adapter delegates model loading, architecture, normalization, batch sorting/padding, CTC handling, and decoding to the frozen adapter. Contrast-disabled mode skips only PIL `ImageEnhance.Contrast`.",
        "- CLAHE and histogram equalization use the existing registry defaults from `configs/preprocessing.yaml`; neither operation changes dimensions. No source pixels or annotations were modified.",
        "- No saved validation predictions were found in prior OCR runs; therefore comparison against a previously saved Phase 4D validation output is unavailable.",
        "",
        "## 2. Experimental findings",
        "",
        "| Arm | Condition | N | CER | WER | Exact match | Mean paired Delta CER vs A | Median paired Delta CER | Improved / unchanged / degraded | Preprocess ms/sample | OCR ms/sample | Total ms/sample |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        result = metrics["arms"][arm]
        paired = result["paired_delta_vs_A"]
        latency = result["latency_ms"]
        lines.append(
            f"| {arm} | {ARM_DETAILS[arm]['label']} | {result['sample_count']} | "
            f"{result['cer']:.6f} | {result['wer']:.6f} | "
            f"{result['exact_match_accuracy']:.2%} | "
            f"{paired['mean_cer_original_minus_arm']:+.6f} | "
            f"{paired['median_cer_original_minus_arm']:+.6f} | "
            f"{paired['improved_count']} / {paired['unchanged_count']} / "
            f"{paired['degraded_count']} | "
            f"{latency['preprocessing_mean_per_sample']:.2f} | "
            f"{latency['ocr_inference_mean_per_sample']:.2f} | "
            f"{latency['pipeline_mean_per_sample']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Positive Delta CER means lower CER than Arm A. Paired counts are based on per-sample JiWER CER.",
            "",
            "### Per-language CER/WER",
            "",
            "| Arm | Language | N | CER | WER | Exact match |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        for language, result in metrics["arms"][arm]["per_language"].items():
            lines.append(
                f"| {arm} | {language} | {result['sample_count']} | "
                f"{result['cer']:.6f} | {result['wer']:.6f} | "
                f"{result['exact_match_accuracy']:.2%} |"
            )
    lines.extend(
        [
            "",
            "## 3. Contrast-enhancement interaction",
            "",
            "| Comparison | Question | Mean paired Delta CER | Median | Improved | Unchanged | Degraded |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for pair in interactions:
        lines.append(
            f"| {pair['pair']} | {pair['description']} | "
            f"{pair['mean_paired_delta_cer']:+.6f} | "
            f"{pair['median_paired_delta_cer']:+.6f} | "
            f"{pair['improved_count']} | {pair['unchanged_count']} | "
            f"{pair['degraded_count']} |"
        )
    contrasts = metrics["external_native_contrast_interaction_contrasts"]
    lines.extend(
        [
            "",
            "- CLAHE interaction contrast `(F-D) - (A-B)`: "
            f"{contrasts['clahe_disabled_minus_enabled']['value']:+.6f}.",
            "- Histogram-equalization interaction contrast `(F-E) - (A-C)`: "
            f"{contrasts['histogram_equalization_disabled_minus_enabled']['value']:+.6f}.",
            "- These are descriptive differences of paired mean CER effects. They do not alone establish a causal mechanism or statistical interaction.",
            "",
            "## 4. Visual failure analysis",
            "",
            f"- Saved {visual_info['generated_count']} four-panel comparisons across improved, unchanged, degraded, and large-regression cases.",
            f"- Languages represented: `{', '.join(visual_info['languages_represented'])}`.",
            "- Each comparison shows the original image, external operation result, and the CRNN-native tensor view with contrast enabled and disabled, plus ground truth and predictions.",
            "- Inspect matras/modifiers, thin components, close spacing, background texture, local contrast, stroke merging, and amplified noise. These visual examples are diagnostics, not representative estimates of the population.",
            "",
            "## 5. Limitations",
            "",
            "- This is a fixed-seed 200-sample validation diagnostic, not test-set evaluation and not a final generalization estimate.",
            "- No stored Phase 4D validation prediction file was available. Arm A was cross-checked against the frozen Phase 4D adapter on the selected validation images instead.",
            "- Per-sample CER differences are paired but no confidence interval, hypothesis test, or causal attribution is claimed.",
            "- The contrast-disabled path deliberately removes only the native PIL contrast enhancement. It does not imply a recommended production configuration.",
            "- This experiment does not run OCR on test images, tune selector thresholds, retrain the model, or alter the frozen adapter defaults.",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> Path:
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("Phase 5B diagnostic is restricted to CPU inference")
    config = _read_yaml(CONFIG_PATH)["phase5b_ablation"]
    if (
        config["split"] != "val"
        or int(config["sample_size"]) != 200
        or int(config["seed"]) != 7
        or int(config["batch_size"]) != 64
        or float(config["native_contrast_factor"]) != 2.0
    ):
        raise ValueError("Phase 5B frozen split/sample/seed/batch/contrast settings changed")
    dataset_document = _read_yaml(DATASET_CONFIG_PATH)
    preprocessing_document = _read_yaml(PREPROCESSING_CONFIG_PATH)
    model_document = _read_yaml(MODEL_CONFIG_PATH)
    dataset_configuration = dataset_document["datasets"]["synth6k"]
    if not dataset_configuration.get("enabled"):
        raise ValueError("synth6k is not enabled in datasets.yaml")
    image_root_value = Path(dataset_configuration["image_root"])
    image_root = image_root_value if image_root_value.is_absolute() else PROJECT_ROOT / image_root_value
    image_root = image_root.resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"synth6k image root is missing: {image_root}")

    metadata_before = _sha256_file(METADATA_PATH)
    validation_population, feasible_rows, source_ids_by_split, population_info = (
        _load_validation_population(
            image_root,
            dataset_configuration,
            input_height=32,
        )
    )
    selected_rows = _sample_validation_rows(
        feasible_rows,
        int(config["sample_size"]),
        int(config["seed"]),
    )
    selected_names = [row["filename"] for row in selected_rows]
    test_filenames = {
        row["filename"]
        for row in validation_population
        if row["split"].strip().lower() == "test"
    }
    if len(set(selected_names)) != 200:
        raise RuntimeError("Pilot sample filenames are not unique")
    if any(row["split"].strip().lower() != "val" for row in selected_rows):
        raise RuntimeError("Only validation samples may enter the Phase 5B run")
    if any(row["filename"] in test_filenames for row in selected_rows):
        raise RuntimeError("A test-split image was selected")
    if len({row["source_image_id"] for row in selected_rows}) != 200:
        raise RuntimeError("Sample has repeated source boards")

    timestamp = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S.%fZ")
    output_directory = OUTPUT_ROOT / timestamp
    output_directory.mkdir(parents=True, exist_ok=False)
    original_hashes_before = {
        row["filename"]: _sha256_file(image_root / row["filename"])
        for row in selected_rows
    }
    archive_hash_before = _sha256_file(CHECKPOINT_ARCHIVE_PATH)
    validation_reference_hash_before = _sha256_file(VALIDATION_REFERENCE_PATH)

    sample_manifest = []
    for index, row in enumerate(selected_rows, start=1):
        sample_manifest.append(
            {
                "sample_index": index,
                "filename": row["filename"],
                "source_image_id": row["source_image_id"],
                "language": row["language"],
                "script": row.get("script", ""),
                "split": row["split"],
                "ground_truth": row["transcription"],
                "source_path": str(image_root / row["filename"]),
                "original_image_sha256": original_hashes_before[row["filename"]],
                "ctc_feasible": True,
            }
        )
    _write_csv(
        output_directory / "sample_manifest.csv",
        list(sample_manifest[0]),
        sample_manifest,
    )

    image_paths, transformation_records = _process_operations(
        selected_rows,
        image_root,
        output_directory,
        preprocessing_document,
    )
    if any(
        record["geometry_changed"]
        for record in transformation_records
    ):
        raise RuntimeError("Pixel-only contrast arms unexpectedly changed geometry")

    torch.set_num_threads(2)
    model_configuration = model_document["models"]["crnn"]
    base_adapter = load_model_adapter(
        "crnn",
        config_path=MODEL_CONFIG_PATH,
        project_root=PROJECT_ROOT,
    )
    if not isinstance(base_adapter, CRNNAdapter):
        raise TypeError("Model registry did not return CRNNAdapter")
    diagnostic_adapter = ContrastAblationCRNNAdapter(
        model_configuration,
        PROJECT_ROOT,
        native_contrast_enabled=True,
    )

    batch_size = int(config["batch_size"])
    base_predictions: dict[str, str] = {}
    base_adapter_seconds = 0.0
    for start in range(0, len(selected_rows), batch_size):
        batch_rows = selected_rows[start : start + batch_size]
        started = time.perf_counter()
        batch_predictions = base_adapter.predict(
            [image_paths["A"][row["filename"]] for row in batch_rows]
        )
        base_adapter_seconds += time.perf_counter() - started
        base_predictions.update(
            {
                row["filename"]: prediction
                for row, prediction in zip(
                    batch_rows,
                    batch_predictions,
                    strict=True,
                )
            }
        )
    if set(base_predictions) != set(selected_names):
        raise RuntimeError("Default CRNN adapter did not cover the sample")

    predictions_by_arm: dict[str, dict[str, str]] = {}
    latency_by_arm: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        predictions_by_arm[arm], latency_by_arm[arm] = _predict_arm(
            diagnostic_adapter,
            selected_rows,
            image_paths,
            arm,
            batch_size,
        )
    original_matches = sum(
        predictions_by_arm["A"][filename] == base_predictions[filename]
        for filename in selected_names
    )
    if original_matches != len(selected_names):
        raise RuntimeError(
            f"Arm A differs from frozen CRNNAdapter for {len(selected_names)-original_matches} samples"
        )

    for first, second in (("B", "D"), ("C", "E")):
        if any(
            _sha256_file(image_paths[first][row["filename"]])
            != _sha256_file(image_paths[second][row["filename"]])
            for row in selected_rows
        ):
            raise RuntimeError(
                f"External preprocessing inputs differ between paired arms {first}/{second}"
            )
    for arm in ARMS:
        if any(
            _dimensions(_load_array(image_paths[arm][row["filename"]]))
            != _dimensions(_load_array(image_paths["A"][row["filename"]]))
            for row in selected_rows
        ):
            raise RuntimeError(f"Arm {arm} changed image dimensions")

    compiled_predictions = _compile_predictions(selected_rows, predictions_by_arm)
    metrics, per_language, interaction_rows = _metrics(
        compiled_predictions,
        transformation_records,
        latency_by_arm,
        selected_count=len(selected_rows),
    )
    source_hashes_after = {
        row["filename"]: _sha256_file(image_root / row["filename"])
        for row in selected_rows
    }
    source_images_unchanged = all(
        original_hashes_before[name] == source_hashes_after[name]
        for name in selected_names
    )
    metadata_after = _sha256_file(METADATA_PATH)
    validation_reference_hash_after = _sha256_file(VALIDATION_REFERENCE_PATH)
    archive_hash_after = _sha256_file(CHECKPOINT_ARCHIVE_PATH)
    annotations_unchanged = (
        metadata_before == metadata_after
        and validation_reference_hash_before == validation_reference_hash_after
    )
    checkpoint_unchanged = archive_hash_before == archive_hash_after
    if not source_images_unchanged or not annotations_unchanged or not checkpoint_unchanged:
        raise RuntimeError(
            "Integrity verification failed for source images, annotations, or checkpoint"
        )

    for record in transformation_records:
        record["source_unchanged"] = (
            source_hashes_after[record["filename"]]
            == original_hashes_before[record["filename"]]
        )
        record["annotation_files_unchanged"] = annotations_unchanged

    visual_info = _generate_visuals(
        output_directory,
        selected_rows,
        compiled_predictions,
        image_paths,
        diagnostic_adapter,
    )
    expected_per_arm = {
        arm: sum(row["arm"] == arm for row in compiled_predictions)
        for arm in ARMS
    }
    if any(count != 200 for count in expected_per_arm.values()):
        raise RuntimeError(f"Incorrect arm sample counts: {expected_per_arm}")
    filename_sets = {
        arm: {
            row["filename"] for row in compiled_predictions if row["arm"] == arm
        }
        for arm in ARMS
    }
    truth_sets = {
        arm: {
            (row["filename"], row["ground_truth"])
            for row in compiled_predictions
            if row["arm"] == arm
        }
        for arm in ARMS
    }
    if any(filename_sets[arm] != filename_sets["A"] for arm in ARMS):
        raise RuntimeError("Experimental arms do not share identical filenames")
    if any(truth_sets[arm] != truth_sets["A"] for arm in ARMS):
        raise RuntimeError("Experimental arms do not share identical ground truth")

    _write_csv(
        output_directory / "predictions.csv",
        [
            "sample_index",
            "arm",
            "filename",
            "source_image_id",
            "language",
            "split",
            "ground_truth",
            "prediction",
            "per_sample_cer",
            "delta_cer_vs_A",
        ],
        compiled_predictions,
    )
    _write_csv(
        output_directory / "per_language_metrics.csv",
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
        output_directory / "interaction_analysis.csv",
        list(interaction_rows[0]),
        interaction_rows,
    )
    (output_directory / "transformation_log.json").write_text(
        json.dumps(
            transformation_records,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    sample_info = {
        "requested_count": int(config["sample_size"]),
        "selected_count": len(selected_rows),
        "seed": int(config["seed"]),
        "language_counts": dict(
            sorted(Counter(row["language"] for row in selected_rows).items())
        ),
        "unique_source_count": len(
            {row["source_image_id"] for row in selected_rows}
        ),
        "split": "val",
        "sampling_method": (
            "CTC-feasible validation rows only; choose one random line crop per "
            "source image ID; proportional language quotas; Python random seed 7; "
            "sorted by filename after selection"
        ),
        "population_counts": population_info,
        "validation_split_total": sum(
            row["split"].strip().lower() == "val" for row in validation_population
        ),
        "validation_ctc_feasible_population": len(feasible_rows),
        "source_id_counts_by_split": {
            split: len(values) for split, values in source_ids_by_split.items()
        },
    }
    parity = {
        "saved_validation_baseline_available": False,
        "saved_validation_baseline_search": (
            "No Phase 4D validation predictions were found in existing "
            "baseline_reproduction or Phase 5A run outputs."
        ),
        "original_adapter_predictions_match": original_matches,
        "sample_count": len(selected_rows),
        "reference_adapter_seconds": base_adapter_seconds,
    }
    validation_info = {
        "sample_count_per_arm": expected_per_arm,
        "same_filenames_across_arms": True,
        "same_ground_truth_across_arms": True,
        "all_validation_split": all(
            row["split"].strip().lower() == "val" for row in selected_rows
        ),
        "test_split_filenames_in_sample": sum(
            row["filename"] in test_filenames for row in selected_rows
        ),
        "source_ids_unique": len(
            {row["source_image_id"] for row in selected_rows}
        )
        == 200,
        "source_split_overlap_counts": population_info[
            "source_id_split_overlap_counts"
        ],
        "original_adapter_prediction_matches": original_matches,
        "original_adapter_prediction_count": len(selected_rows),
        "source_images_unchanged": source_images_unchanged,
        "annotations_unchanged": annotations_unchanged,
        "checkpoint_archive_unchanged": checkpoint_unchanged,
        "checkpoint_archive_sha256_before": archive_hash_before,
        "checkpoint_archive_sha256_after": archive_hash_after,
        "metadata_sha256_before": metadata_before,
        "metadata_sha256_after": metadata_after,
        "validation_reference_sha256_before": validation_reference_hash_before,
        "validation_reference_sha256_after": validation_reference_hash_after,
        "contrast_disabled_changes_only_contrast": True,
        "all_external_operations_preserve_geometry": True,
        "no_training": True,
        "test_split_inference": False,
    }
    metadata = {
        "status": "passed",
        "run_id": timestamp,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_command": (
            r".\.venv\Scripts\python.exe "
            r".\experiments\ocr_evaluation\phase5b_ablation\run_ablation.py"
        ),
        "working_directory": str(PROJECT_ROOT),
        "dataset": "synth6k",
        "sample": sample_info,
        "arms": {
            arm: {
                **ARM_DETAILS[arm],
                "native_contrast_factor": (
                    2.0 if ARM_DETAILS[arm]["native_contrast_enabled"] else None
                ),
                "external_operation_parameters": next(
                    (
                        record["operation_parameters"]
                        for record in transformation_records
                        if record["arm"] == arm
                        and record["operation"] != "keep_original"
                    ),
                    {},
                ),
            }
            for arm in ARMS
        },
        "model": {
            "adapter": "ContrastAblationCRNNAdapter delegates to frozen CRNNAdapter",
            "architecture": model_configuration["architecture"],
            "checkpoint_sha256": diagnostic_adapter.checkpoint_sha256,
            "vocabulary_sha256": diagnostic_adapter.vocabulary_sha256,
            "device": str(next(diagnostic_adapter.model.parameters()).device),
            "batch_size": int(config["batch_size"]),
            "input_height": 32,
            "native_preprocessing": (
                "PIL grayscale; round aspect-ratio width; PIL bilinear resize to "
                "height 32; optional Contrast.enhance(2.0); float32 /255; "
                "(x-.5)/.5; one channel"
            ),
            "batching_and_decoder": (
                "unchanged CRNNAdapter descending content-width sort within batch, "
                "right-pad to multiple of four with normalized zero, original greedy "
                "CTC decoder, blank index 0"
            ),
        },
        "operations": {
            "clahe": {
                "operation": "existing preprocessing registry ClaheOperation",
                "parameters": preprocessing_document["preprocessing"]["clahe"],
            },
            "histogram_equalization": {
                "operation": "existing preprocessing registry HistogramEqualizationOperation",
                "parameters": preprocessing_document["preprocessing"][
                    "histogram_equalization"
                ],
            },
            "native_contrast_default": 2.0,
            "contrast_disabled_behavior": "skip only PIL ImageEnhance.Contrast",
        },
        "metrics": {
            "library": "JiWER",
            "version": version("jiwer"),
            "text_normalization": "none",
            "delta_cer_definition": "CER_A - CER_arm; positive is improvement",
        },
        "validation": validation_info,
        "latency_by_arm_seconds": latency_by_arm,
        "visual_analysis": visual_info,
        "inputs": {
            "phase5b_config": {
                "path": str(CONFIG_PATH),
                "sha256": _sha256_file(CONFIG_PATH),
            },
            "dataset_metadata": {
                "path": str(METADATA_PATH),
                "sha256": metadata_before,
            },
            "validation_reference": {
                "path": str(VALIDATION_REFERENCE_PATH),
                "sha256": validation_reference_hash_before,
            },
            "dataset_config": {
                "path": str(DATASET_CONFIG_PATH),
                "sha256": _sha256_file(DATASET_CONFIG_PATH),
            },
            "preprocessing_config": {
                "path": str(PREPROCESSING_CONFIG_PATH),
                "sha256": _sha256_file(PREPROCESSING_CONFIG_PATH),
            },
            "model_config": {
                "path": str(MODEL_CONFIG_PATH),
                "sha256": _sha256_file(MODEL_CONFIG_PATH),
            },
            "checkpoint_archive": {
                "path": str(CHECKPOINT_ARCHIVE_PATH),
                "sha256_before": archive_hash_before,
                "sha256_after": archive_hash_after,
            },
        },
        "limitations": [
            "Validation diagnostic only; no test-split images were used.",
            "No previously saved validation prediction file exists for external historical parity.",
            "Paired pilot differences are descriptive and not statistically conclusive.",
            "No model training, tuning, or checkpoint modification was performed.",
        ],
    }
    (output_directory / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_directory / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_directory / "metrics.md").write_text(
        _markdown(timestamp, sample_info, metrics, interaction_rows, visual_info, parity),
        encoding="utf-8",
    )
    print(output_directory)
    return output_directory


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
