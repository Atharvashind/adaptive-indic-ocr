from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
import time
import zipfile
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ocr_evaluation import run_phase5_pilot as phase5a  # noqa: E402
from src.evaluation.cer_wer import calculate_cer_wer  # noqa: E402
from src.models import CRNNAdapter, load_model_adapter  # noqa: E402
from src.preprocessing.adaptive_selector import (  # noqa: E402
    AdaptivePreprocessorSelector,
)
from src.preprocessing.registry import create_default_registry  # noqa: E402
from src.quality.quality_analyzer import QualityAnalyzer  # noqa: E402

BASELINE_PREDICTIONS_PATH = phase5a.BASELINE_PREDICTIONS_PATH
PILOT_MANIFEST_PATH = (
    phase5a.OUTPUT_ROOT
    / "run-20261106T101321.309609Z"
    / "sample_manifest.csv"
)
DATASET_CONFIG_PATH = phase5a.DATASET_CONFIG_PATH
PREPROCESSING_CONFIG_PATH = phase5a.PREPROCESSING_CONFIG_PATH
MODEL_CONFIG_PATH = phase5a.MODEL_CONFIG_PATH
PILOT_CONFIG_PATH = phase5a.PILOT_CONFIG_PATH
CHECKPOINT_ARCHIVE_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "recovered_baseline"
    / "Multilingual_CRNN_OCR-20261105T161607Z-1-001.zip"
)
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "ocr_evaluation" / "phase5_final"
EXPECTED_REMAINING = 7894
SEED = 7
ARM_ORDER = phase5a.ARM_ORDER
ARM_LABELS = phase5a.ARM_LABELS


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")
    return loaded


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


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _hash_zip_member(archive_path: Path, member: str) -> str:
    with zipfile.ZipFile(archive_path) as archive:
        return hashlib.sha256(archive.read(member)).hexdigest()


def _baseline_sample_manifest(
    selected_rows: list[dict[str, str]],
    baseline_rows: dict[str, dict[str, str]],
    baseline_contexts: dict[str, tuple[int, int]],
    image_root: Path,
    source_hashes: dict[str, str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows, start=1):
        filename = row["filename"]
        context = baseline_contexts[filename]
        result.append(
            {
                "sample_index": index,
                "filename": filename,
                "source_image_id": row["source_image_id"],
                "language": row["language"],
                "script": row.get("script", ""),
                "split": row["split"],
                "ground_truth": row["transcription"],
                "source_path": str(image_root / filename),
                "original_image_sha256": source_hashes[filename],
                "phase4b_prediction": baseline_rows[filename]["prediction"],
                "phase4b_batch_index": context[0],
                "phase4b_padded_width": context[1],
            }
        )
    return result


def _configuration_snapshot(
    *,
    dataset_config: dict[str, Any],
    preprocessing_config: dict[str, Any],
    model_config: dict[str, Any],
    pilot_config: dict[str, Any],
    exclusion_path: Path,
    exclusion_hash: str,
    exclusion_filenames: list[str],
) -> dict[str, Any]:
    adapter_path = PROJECT_ROOT / "src" / "models" / "crnn_adapter.py"
    metric_path = PROJECT_ROOT / "src" / "evaluation" / "cer_wer.py"
    crnn = model_config["models"]["crnn"]
    snapshot = {
        "snapshot_version": 1,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "dataset": {
            "name": "synth6k",
            "configuration": dataset_config,
            "configuration_sha256": _sha256_file(DATASET_CONFIG_PATH),
            "test_annotations": str(
                PROJECT_ROOT / "synth6k" / "annotations" / "test.txt"
            ),
            "test_annotations_sha256": _sha256_file(
                PROJECT_ROOT / "synth6k" / "annotations" / "test.txt"
            ),
        },
        "preprocessing": {
            "configuration": preprocessing_config,
            "configuration_sha256": _sha256_file(PREPROCESSING_CONFIG_PATH),
            "selector_thresholds": preprocessing_config.get(
                "adaptive_selection", {}
            ).get("thresholds", {}),
        },
        "model": {
            "configuration": crnn,
            "configuration_sha256": _sha256_file(MODEL_CONFIG_PATH),
            "checkpoint_sha256": crnn["checkpoint_sha256"],
            "vocabulary_sha256": crnn["vocabulary_sha256"],
            "adapter": "src.models.crnn_adapter.CRNNAdapter",
            "adapter_source_sha256": _sha256_file(adapter_path),
            "decoder": {
                "type": crnn["decoder"],
                "blank_index": crnn["blank_index"],
                "unicode_normalization": None,
                "whitespace_transformations": None,
            },
        },
        "metrics": {
            "implementation": "src.evaluation.cer_wer.calculate_cer_wer",
            "implementation_sha256": _sha256_file(metric_path),
            "jiwer_version": version("jiwer"),
            "rapidfuzz_version": version("rapidfuzz"),
            "numpy_version": np.__version__,
            "python_version": sys.version,
            "exact_match": "raw Python string equality",
            "text_normalization": "none",
            "paired_delta": "CER_original - CER_experimental; positive is improvement",
            "confidence_interval": {
                "method": "95% percentile cluster bootstrap",
                "cluster": "source_image_id",
                "statistic": "mean paired per-sample CER difference",
                "replicates": 5000,
                "seed": SEED,
            },
        },
        "sample_selection": {
            "split": "test",
            "ctc_feasible_only": True,
            "population_policy": (
                "All Phase 4B CTC-feasible test samples except every Phase 5A "
                "pilot filename; retain Phase 4B order and language distribution."
            ),
            "expected_sample_count": EXPECTED_REMAINING,
            "one_crop_per_source_board": False,
            "source_board_note": (
                "Not feasible for this complete remainder: the test split contains "
                "only 612 source boards for 7894 required crops."
            ),
            "exclusion_manifest": {
                "path": str(exclusion_path),
                "sha256": exclusion_hash,
                "filenames": exclusion_filenames,
            },
        },
        "experiment_arms": {
            arm: {"label": ARM_LABELS[arm], **pilot_config["arms"][arm]}
            for arm in ARM_ORDER
        },
        "source_hashes": {
            "phase5a_configuration_sha256": _sha256_file(PILOT_CONFIG_PATH),
            "phase4b_predictions_sha256": _sha256_file(BASELINE_PREDICTIONS_PATH),
            "checkpoint_archive_sha256": _sha256_file(CHECKPOINT_ARCHIVE_PATH),
            "phase5a_sample_manifest_sha256": _sha256_file(PILOT_MANIFEST_PATH),
        },
        "batch_context": {
            "batch_size": int(crnn["batch_size"]),
            "preserve_phase4b_group_and_padded_width": True,
            "right_padding_multiple": 4,
            "guard_images": (
                "Use one non-scored guard in each sub-batch to preserve the "
                "Phase 4B padded width; guard consumes one batch slot."
            ),
        },
        "transformation_failure_policy": (
            "Log the failure, use the unmodified original image for that sample "
            "and arm, and still emit one prediction per arm."
        ),
    }
    return snapshot


def _group_by_baseline(
    rows: list[dict[str, str]],
    contexts: dict[str, tuple[int, int]],
) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[contexts[row["filename"]][0]].append(row)
    return grouped


def _operation_spec(
    arm: str,
    image: np.ndarray,
    row: dict[str, str],
    selector: AdaptivePreprocessorSelector,
    pilot_config: dict[str, Any],
    preprocessing_config: dict[str, Any],
) -> tuple[str, dict[str, Any], str, str, float, dict[str, Any] | None]:
    if arm == "fixed":
        config = pilot_config["arms"]["fixed"]
        return (
            str(config["operation"]),
            dict(config.get("parameters", {})),
            str(config["rationale"]).strip(),
            "fixed",
            0.0,
            None,
        )
    if arm == "scene_text":
        config = pilot_config["arms"]["scene_text"]
        operation = str(config["operation"])
        parameters = preprocessing_config.get("preprocessing", {}).get(operation)
        if not isinstance(parameters, dict):
            raise ValueError(f"Frozen {operation!r} parameters are missing")
        return (
            operation,
            dict(parameters),
            str(config["rationale"]).strip(),
            "fixed_scene_text",
            0.0,
            None,
        )
    mode = "automatic" if arm == "automatic" else "user_guided"
    started = time.perf_counter()
    decision = selector.evaluate(
        image,
        image_identifier=row["filename"],
        mode=mode,
        domain="scene_text" if mode == "user_guided" else None,
        metadata={
            "filename": row["filename"],
            "source_image_id": row["source_image_id"],
            "source_domain": "scene_text",
            "domain_labels": ["scene_text"],
            "granularity": "line_crop",
        },
        color_order="RGB",
    )
    selection_ms = (time.perf_counter() - started) * 1000
    return (
        decision.selected_operation,
        dict(decision.operation_parameters),
        "; ".join(decision.reasons),
        decision.operating_mode,
        selection_ms,
        decision.to_dict(),
    )


def _run_arm(
    *,
    arm: str,
    adapter: CRNNAdapter,
    selected_rows: list[dict[str, str]],
    baseline_contexts: dict[str, tuple[int, int]],
    image_root: Path,
    run_directory: Path,
    source_hashes: dict[str, str],
    pilot_config: dict[str, Any],
    preprocessing_config: dict[str, Any],
    dataset_config: dict[str, Any],
    visual_samples: set[str],
) -> tuple[dict[str, str], float, dict[str, float], list[dict[str, Any]]]:
    image_root_work = run_directory / ".working" / arm
    image_root_work.mkdir(parents=True, exist_ok=True)
    guard_root = run_directory / ".working" / "guards"
    guard_root.mkdir(parents=True, exist_ok=True)
    quality_analyzer = QualityAnalyzer(dataset_config.get("quality", {}))
    operation_registry = create_default_registry()
    selector = AdaptivePreprocessorSelector(
        preprocessing_config,
        quality_analyzer=quality_analyzer,
        operation_configuration=preprocessing_config,
    )
    grouped = _group_by_baseline(selected_rows, baseline_contexts)
    predictions: dict[str, str] = {}
    inference_seconds = 0.0
    inference_latency: dict[str, float] = {}
    records: list[dict[str, Any]] = []
    batch_size = adapter.batch_size
    sample_indices = {
        row["filename"]: index for index, row in enumerate(selected_rows, start=1)
    }

    for group_index in sorted(grouped):
        group_rows = grouped[group_index]
        baseline_width = baseline_contexts[group_rows[0]["filename"]][1]
        prepared: list[tuple[dict[str, str], Path, dict[str, Any]]] = []
        max_width = baseline_width
        for row in group_rows:
            filename = row["filename"]
            source_path = image_root / filename
            source_array = phase5a._load_array(source_path)
            source_dimensions = phase5a._dimensions(source_array)
            if arm == "original":
                record = {
                    "sample_index": sample_indices[filename],
                    "arm": arm,
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
                    "selection_reason": "Original baseline; no external preprocessing.",
                    "selection_time_ms": 0.0,
                    "processing_time_ms": 0.0,
                    "total_preprocessing_time_ms": 0.0,
                    "transformation_status": "not_applicable",
                    "original_dimensions": source_dimensions,
                    "processed_dimensions": source_dimensions,
                    "geometry_changed": False,
                    "coordinate_transform_matrix": None,
                    "annotation_geometry_status": (
                        "line_crop_transcription_unchanged; source-board bbox "
                        "coordinates are not line-crop coordinates"
                    ),
                    "warnings": [],
                    "selector_decision": None,
                }
                prepared.append((row, source_path, record))
                max_width = max(
                    max_width,
                    phase5a._round_up_four(
                        int(round(source_dimensions["width"] * 32 / source_dimensions["height"]))
                    ),
                )
                continue

            started = time.perf_counter()
            (
                operation_name,
                parameters,
                reason,
                selector_mode,
                selection_ms,
                decision,
            ) = _operation_spec(
                arm,
                source_array,
                row,
                selector,
                pilot_config,
                preprocessing_config,
            )
            operation_ms = 0.0
            warnings: list[str] = []
            details: dict[str, Any] = {}
            status = "keep_original"
            processed_array = source_array
            output_path = source_path
            processed_hash = source_hashes[filename]
            actual_parameters: dict[str, Any] = parameters
            if operation_name != "keep_original":
                operation_result = operation_registry.get(operation_name).execute(
                    source_array,
                    parameters=parameters,
                    metadata={
                        "source_image_id": row["source_image_id"],
                        "granularity": "line_crop",
                    },
                )
                operation_ms = float(operation_result.execution_time_ms)
                warnings = list(operation_result.warnings)
                details = operation_result.details
                actual_parameters = operation_result.parameters or parameters
                if operation_result.success and operation_result.processed_image is not None:
                    processed_array = operation_result.processed_image
                    status = operation_result.status
                    output_path = image_root_work / Path(filename).with_suffix(".png")
                    phase5a._save_array(processed_array, output_path)
                    processed_hash = _sha256_file(output_path)
                    if filename in visual_samples:
                        visual_staging = (
                            run_directory
                            / "visual_comparisons"
                            / ".staging"
                            / arm
                            / Path(filename).with_suffix(".png")
                        )
                        phase5a._save_array(processed_array, visual_staging)
                else:
                    status = "fallback_original"
                    warnings.append(
                        "Transformation failed; original input used under the frozen fallback policy."
                    )
            processed_dimensions = phase5a._dimensions(processed_array)
            preprocessing_ms = selection_ms + operation_ms
            record = {
                "sample_index": sample_indices[filename],
                "arm": arm,
                "filename": filename,
                "source_image_id": row["source_image_id"],
                "language": row["language"],
                "original_image_path": str(source_path),
                "processed_image_path": (
                    str(output_path)
                    if output_path == source_path
                    else f"temporary:{arm}/{filename}"
                ),
                "original_image_sha256": source_hashes[filename],
                "processed_image_sha256": processed_hash,
                "operation_selected": operation_name,
                "operation_parameters": actual_parameters,
                "selector_mode": selector_mode,
                "selection_reason": reason,
                "selection_time_ms": selection_ms,
                "processing_time_ms": operation_ms,
                "total_preprocessing_time_ms": preprocessing_ms,
                "transformation_status": status,
                "original_dimensions": source_dimensions,
                "processed_dimensions": processed_dimensions,
                "geometry_changed": phase5a._geometry_changed(
                    source_dimensions,
                    processed_dimensions,
                    details,
                ),
                "coordinate_transform_matrix": details.get(
                    "coordinate_transform_matrix"
                ),
                "annotation_geometry_status": (
                    "line_crop transcription unchanged; source-board bbox "
                    "coordinates are not line-crop coordinates"
                ),
                "warnings": warnings,
                "selector_decision": decision,
                "record_wall_time_ms": (time.perf_counter() - started) * 1000,
            }
            content_width = int(
                round(
                    processed_dimensions["width"]
                    * 32
                    / processed_dimensions["height"]
                )
            )
            max_width = max(max_width, phase5a._round_up_four(content_width))
            prepared.append((row, output_path, record))

        guard_path = guard_root / f"{arm}-{group_index:04d}.png"
        outputs_created = [
            path
            for _, path, record in prepared
            if path != image_root / record["filename"]
        ]
        chunk_size = max(1, batch_size - 1)
        for start in range(0, len(prepared), chunk_size):
            chunk = prepared[start : start + chunk_size]
            phase5a._make_guard(guard_path, max_width)
            started = time.perf_counter()
            outputs = adapter.predict(
                [path for _, path, _ in chunk] + [guard_path]
            )
            elapsed = time.perf_counter() - started
            if len(outputs) != len(chunk) + 1:
                raise RuntimeError(
                    f"Arm {arm} returned {len(outputs)} predictions for "
                    f"{len(chunk)} samples plus a guard"
                )
            inference_seconds += elapsed
            allocated = elapsed / len(chunk)
            for (row, _, record), prediction in zip(
                chunk, outputs[:-1], strict=True
            ):
                filename = row["filename"]
                predictions[filename] = prediction
                inference_latency[filename] = allocated
                records.append(record)
            guard_path.unlink()
        for path in outputs_created:
            path.unlink()
        if len(prepared) == 0:
            raise RuntimeError(f"Empty baseline context group {group_index}")

    if set(predictions) != {row["filename"] for row in selected_rows}:
        raise RuntimeError(f"Arm {arm} did not predict every selected filename")
    if len(records) != len(selected_rows):
        raise RuntimeError(f"Arm {arm} transformation log count is incomplete")
    return predictions, inference_seconds, inference_latency, records


def _per_sample_cer(reference: str, hypothesis: str) -> float:
    import jiwer

    return float(jiwer.cer([reference], [hypothesis]))


def _cluster_bootstrap_ci(
    deltas_by_source: dict[str, list[float]],
    *,
    seed: int,
    replicates: int = 5000,
) -> dict[str, float]:
    source_ids = sorted(deltas_by_source)
    if len(source_ids) < 2:
        raise ValueError("Cluster bootstrap requires at least two source boards")
    cluster_sums = np.asarray(
        [sum(deltas_by_source[source_id]) for source_id in source_ids],
        dtype=np.float64,
    )
    cluster_sizes = np.asarray(
        [len(deltas_by_source[source_id]) for source_id in source_ids],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(source_ids), size=(replicates, len(source_ids)))
    sums = cluster_sums[draws].sum(axis=1)
    sizes = cluster_sizes[draws].sum(axis=1)
    bootstrap_means = sums / sizes
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "lower_95_percent": float(low),
        "upper_95_percent": float(high),
        "replicates": replicates,
        "cluster_count": len(source_ids),
    }


def _build_metrics(
    selected_rows: list[dict[str, str]],
    predictions_by_arm: dict[str, dict[str, str]],
    transform_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    records_by_arm_filename = {
        (record["arm"], record["filename"]): record for record in transform_records
    }
    all_predictions: list[dict[str, Any]] = []
    sample_deltas: dict[str, dict[str, float]] = defaultdict(dict)
    original_by_filename = predictions_by_arm["original"]
    per_language: list[dict[str, Any]] = []
    arm_metrics: dict[str, Any] = {}
    operation_summary: list[dict[str, Any]] = []
    sample_count = len(selected_rows)

    for arm in ARM_ORDER:
        arm_rows: list[dict[str, Any]] = []
        for sample_index, row in enumerate(selected_rows, start=1):
            filename = row["filename"]
            prediction = predictions_by_arm[arm][filename]
            cer_value = _per_sample_cer(row["transcription"], prediction)
            paired_delta = (
                0.0
                if arm == "original"
                else _per_sample_cer(
                    row["transcription"], original_by_filename[filename]
                )
                - cer_value
            )
            sample_deltas[filename][arm] = paired_delta
            result = {
                "arm": arm,
                "sample_index": sample_index,
                "filename": filename,
                "source_image_id": row["source_image_id"],
                "language": row["language"],
                "ground_truth": row["transcription"],
                "prediction": prediction,
                "sample_cer": cer_value,
                "paired_delta_cer_vs_original": paired_delta,
                "operation_selected": records_by_arm_filename[
                    (arm, filename)
                ]["operation_selected"],
            }
            all_predictions.append(result)
            arm_rows.append(result)

        aggregate = calculate_cer_wer(
            [row["ground_truth"] for row in arm_rows],
            [row["prediction"] for row in arm_rows],
        )
        language_metrics: dict[str, Any] = {}
        for language in sorted({row["language"] for row in arm_rows}):
            subset = [row for row in arm_rows if row["language"] == language]
            metrics = calculate_cer_wer(
                [row["ground_truth"] for row in subset],
                [row["prediction"] for row in subset],
            )
            language_metrics[language] = metrics
            per_language.append(
                {"arm": arm, "language": language, **metrics}
            )
        total_preprocess_ms = sum(
            float(records_by_arm_filename[(arm, row["filename"])].get(
                "total_preprocessing_time_ms", 0.0
            ))
            for row in selected_rows
        )
        arm_metrics[arm] = {
            **aggregate,
            "label": ARM_LABELS[arm],
            "per_language": language_metrics,
            "preprocessing_latency_ms": {
                "total": total_preprocess_ms,
                "mean_per_sample": total_preprocess_ms / sample_count,
            },
            "operation_selection_counts": dict(
                sorted(
                    Counter(
                        records_by_arm_filename[(arm, row["filename"])]
                        ["operation_selected"]
                        for row in selected_rows
                    ).items()
                )
            ),
            "failed_transformations": sum(
                records_by_arm_filename[(arm, row["filename"])]
                ["transformation_status"] == "fallback_original"
                for row in selected_rows
            ),
        }

    original_cer = arm_metrics["original"]["cer"]
    for arm in ARM_ORDER:
        if arm == "original":
            paired = [0.0] * sample_count
        else:
            paired = [sample_deltas[row["filename"]][arm] for row in selected_rows]
        deltas_by_source: dict[str, list[float]] = defaultdict(list)
        for row, delta in zip(selected_rows, paired, strict=True):
            deltas_by_source[row["source_image_id"]].append(delta)
        if arm == "original":
            ci = {
                "lower_95_percent": 0.0,
                "upper_95_percent": 0.0,
                "replicates": 5000,
                "cluster_count": len(deltas_by_source),
            }
        else:
            ci = _cluster_bootstrap_ci(
                deltas_by_source,
                seed=SEED,
            )
        values = np.asarray(paired, dtype=np.float64)
        metrics = arm_metrics[arm]
        metrics["aggregate_delta_cer_vs_original"] = (
            0.0 if arm == "original" else original_cer - metrics["cer"]
        )
        metrics["paired_sample_cer_difference"] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "improved_count": int(np.count_nonzero(values > 0)),
            "unchanged_count": int(np.count_nonzero(values == 0)),
            "degraded_count": int(np.count_nonzero(values < 0)),
            "improved_percent": float(np.count_nonzero(values > 0) * 100 / sample_count),
            "unchanged_percent": float(np.count_nonzero(values == 0) * 100 / sample_count),
            "degraded_percent": float(np.count_nonzero(values < 0) * 100 / sample_count),
            "mean_95_percent_cluster_bootstrap_ci": ci,
        }

    operation_buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_prediction = {
        (row["arm"], row["filename"]): row for row in all_predictions
    }
    for record in transform_records:
        operation_buckets[
            (record["arm"], record["language"], record["operation_selected"])
        ].append(record)
    for (arm, language, operation), records in sorted(operation_buckets.items()):
        subset = [
            by_prediction[(arm, record["filename"])] for record in records
        ]
        condition_metrics = calculate_cer_wer(
            [row["ground_truth"] for row in subset],
            [row["prediction"] for row in subset],
        )
        deltas = [
            by_prediction[(arm, record["filename"])]
            ["paired_delta_cer_vs_original"]
            for record in records
        ]
        operation_summary.append(
            {
                "arm": arm,
                "language": language,
                "operation": operation,
                "sample_count": len(records),
                "selection_fraction_within_arm_language": len(records)
                / sum(
                    len(v)
                    for (candidate_arm, candidate_language, _), v
                    in operation_buckets.items()
                    if candidate_arm == arm and candidate_language == language
                ),
                **condition_metrics,
                "mean_paired_delta_cer_vs_original": float(np.mean(deltas)),
                "improved_count": sum(delta > 0 for delta in deltas),
                "unchanged_count": sum(delta == 0 for delta in deltas),
                "degraded_count": sum(delta < 0 for delta in deltas),
            }
        )

    summary = {
        "sample_count": sample_count,
        "arms": arm_metrics,
        "operation_outcomes": operation_summary,
        "operation_outcomes_overall": _overall_operation_outcomes(
            transform_records,
            all_predictions,
        ),
        "bootstrap_method": (
            "Percentile bootstrap of mean paired per-sample CER difference; "
            "resample source_image_id clusters with replacement (5000 replicates)."
        ),
    }
    return summary, per_language, all_predictions


def _operation_selection_rows(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result_by_key = {
        (row["arm"], row["filename"]): row for row in predictions
    }
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (record["arm"], record["language"], record["operation_selected"])
        ].append(record)
    rows: list[dict[str, Any]] = []
    for (arm, language, operation), group in sorted(grouped.items()):
        subset_predictions = [
            result_by_key[(arm, record["filename"])] for record in group
        ]
        operation_metrics = calculate_cer_wer(
            [row["ground_truth"] for row in subset_predictions],
            [row["prediction"] for row in subset_predictions],
        )
        count_in_stratum = sum(
            len(items)
            for (candidate_arm, candidate_language, _), items in grouped.items()
            if candidate_arm == arm and candidate_language == language
        )
        deltas = [
            result_by_key[(arm, record["filename"])]
            ["paired_delta_cer_vs_original"]
            for record in group
        ]
        rows.append(
            {
                "arm": arm,
                "language": language,
                "operation_selected": operation,
                "sample_count": len(group),
                "within_language_selection_fraction": len(group) / count_in_stratum,
                "cer": operation_metrics["cer"],
                "wer": operation_metrics["wer"],
                "exact_match_accuracy": operation_metrics["exact_match_accuracy"],
                "mean_paired_delta_cer_vs_original": float(np.mean(deltas)),
                "improved_count": sum(delta > 0 for delta in deltas),
                "unchanged_count": sum(delta == 0 for delta in deltas),
                "degraded_count": sum(delta < 0 for delta in deltas),
            }
        )
    return rows


def _overall_operation_outcomes(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result_by_key = {
        (row["arm"], row["filename"]): row for row in predictions
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["arm"], record["operation_selected"])].append(record)
    outcomes: list[dict[str, Any]] = []
    for (arm, operation), group in sorted(grouped.items()):
        subset = [result_by_key[(arm, record["filename"])] for record in group]
        metrics = calculate_cer_wer(
            [row["ground_truth"] for row in subset],
            [row["prediction"] for row in subset],
        )
        deltas = [
            float(row["paired_delta_cer_vs_original"]) for row in subset
        ]
        outcomes.append(
            {
                "arm": arm,
                "operation": operation,
                "sample_count": len(group),
                **metrics,
                "mean_paired_delta_cer_vs_original": float(np.mean(deltas)),
                "improved_count": sum(delta > 0 for delta in deltas),
                "unchanged_count": sum(delta == 0 for delta in deltas),
                "degraded_count": sum(delta < 0 for delta in deltas),
            }
        )
    return outcomes


def _make_visual_comparison(
    *,
    source_path: Path,
    processed_path: Path,
    destination: Path,
    arm: str,
    row: dict[str, str],
    prediction: dict[str, Any],
) -> None:
    with Image.open(source_path) as image:
        original = image.convert("RGB")
    with Image.open(processed_path) as image:
        processed = image.convert("RGB")
    width = 900
    panel_height = 260
    canvas = Image.new("RGB", (width, panel_height * 2 + 72), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), f"{arm} | {row['language']} | {row['filename']}", fill="black")
    draw.text(
        (10, 28),
        f"sample CER={float(prediction['sample_cer']):.4f}; "
        f"paired Delta CER="
        f"{float(prediction['paired_delta_cer_vs_original']):+.4f}",
        fill="black",
    )
    for image, top, label in (
        (original, 58, "Original image"),
        (processed, 58 + panel_height, "External preprocessing result"),
    ):
        draw.text((10, top), label, fill="black")
        image.thumbnail((width - 20, panel_height - 34))
        canvas.paste(image, (10, top + 20))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG")


def _generate_failure_visuals(
    *,
    run_directory: Path,
    image_root: Path,
    selected_rows: list[dict[str, str]],
    transform_records: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_by_filename = {row["filename"]: row for row in selected_rows}
    prediction_by_filename = {
        row["filename"]: row
        for row in prediction_rows
        if row["arm"] == "automatic"
    }
    records_by_filename = {
        record["filename"]: record
        for record in transform_records
        if record["arm"] == "automatic"
    }
    failures: list[dict[str, Any]] = []
    staging_path = run_directory / ".visual_failure_preview.png"
    operation_registry = create_default_registry()
    for operation in ("clahe", "deskew"):
        candidates = [
            record
            for record in records_by_filename.values()
            if record["operation_selected"] == operation
            and float(
                prediction_by_filename[record["filename"]][
                    "paired_delta_cer_vs_original"
                ]
            )
            < 0
        ]
        if not candidates:
            continue
        worst_record = min(
            candidates,
            key=lambda record: float(
                prediction_by_filename[record["filename"]][
                    "paired_delta_cer_vs_original"
                ]
            ),
        )
        filename = worst_record["filename"]
        row = selected_by_filename[filename]
        source_path = image_root / filename
        source_array = phase5a._load_array(source_path)
        operation_result = operation_registry.get(operation).execute(
            source_array,
            parameters=worst_record["operation_parameters"],
            metadata={
                "source_image_id": row["source_image_id"],
                "granularity": "line_crop",
            },
        )
        if not operation_result.success or operation_result.processed_image is None:
            raise RuntimeError(
                f"Could not reproduce frozen {operation} output for visual analysis: "
                f"{filename}"
            )
        phase5a._save_array(operation_result.processed_image, staging_path)
        if _sha256_file(staging_path) != worst_record["processed_image_sha256"]:
            staging_path.unlink()
            raise RuntimeError(
                f"Visual-analysis transformation hash mismatch for {filename}"
            )
        prediction = prediction_by_filename[filename]
        destination = (
            run_directory
            / "visual_comparisons"
            / "failure_analysis"
            / f"automatic-{operation}-{Path(filename).stem}.png"
        )
        _make_visual_comparison(
            source_path=source_path,
            processed_path=staging_path,
            destination=destination,
            arm=f"automatic/{operation}",
            row=row,
            prediction=prediction,
        )
        staging_path.unlink()
        failures.append(
            {
                "arm": "automatic",
                "operation": operation,
                "filename": filename,
                "language": row["language"],
                "paired_delta_cer_vs_original": prediction[
                    "paired_delta_cer_vs_original"
                ],
                "visual_path": str(destination),
                "selection_note": (
                    "Chosen after evaluation as the largest observed regression "
                    "for visual diagnosis only; not used to tune settings."
                ),
            }
        )
    return failures


def _build_markdown(
    *,
    run_id: str,
    summary: dict[str, Any],
    sample_counts: dict[str, Any],
    exclusion_count: int,
    validation: dict[str, Any],
) -> str:
    lines = [
        "# Phase 5C frozen held-out test evaluation",
        "",
        f"- **Run:** `{run_id}`",
        "- **Status:** passed",
        f"- **Final held-out population:** {summary['sample_count']} CTC-feasible test samples.",
        f"- **Phase 5A test samples excluded:** {exclusion_count}.",
        "- **Population policy:** all remaining feasible test samples, original Phase 4B order.",
        f"- **Language counts:** `{json.dumps(sample_counts, sort_keys=True)}`",
        f"- **Source boards:** {validation['unique_source_board_count']} unique; "
        f"{validation['duplicate_source_board_excess_crops']} extra crops across repeated source IDs.",
        "",
        "## Overall OCR results",
        "",
        "| Arm | N | CER | WER | Exact match | Aggregate Delta CER | Mean paired sample Delta CER | Median paired Delta CER | Improved / unchanged / degraded | Paired mean 95% cluster-bootstrap CI | Preprocess ms/sample | OCR ms/sample | Total ms/sample |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARM_ORDER:
        metrics = summary["arms"][arm]
        paired = metrics["paired_sample_cer_difference"]
        ci = paired["mean_95_percent_cluster_bootstrap_ci"]
        lines.append(
            f"| {ARM_LABELS[arm]} | {metrics['sample_count']} | "
            f"{metrics['cer']:.6f} | {metrics['wer']:.6f} | "
            f"{metrics['exact_match_accuracy']:.2%} | "
            f"{metrics['aggregate_delta_cer_vs_original']:+.6f} | "
            f"{paired['mean']:+.6f} | {paired['median']:+.6f} | "
            f"{paired['improved_count']} / {paired['unchanged_count']} / "
            f"{paired['degraded_count']} | "
            f"[{ci['lower_95_percent']:+.6f}, {ci['upper_95_percent']:+.6f}] | "
            f"{metrics['preprocessing_latency_ms']['mean_per_sample']:.2f} | "
            f"{metrics['inference_latency_ms']['mean_per_sample']:.2f} | "
            f"{metrics['total_pipeline_latency_ms']['mean_per_sample']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Positive Delta CER indicates improvement over Arm A; negative indicates degradation. "
            "Aggregate Delta CER is the difference of corpus CER values. The paired mean is "
            "the mean of per-sample CER differences; they need not be identical.",
            "",
            "## Per-language results",
            "",
            "| Arm | Language | N | CER | WER | Exact match |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for arm in ARM_ORDER:
        for language, metrics in sorted(summary["arms"][arm]["per_language"].items()):
            lines.append(
                f"| {ARM_LABELS[arm]} | {language} | {metrics['sample_count']} | "
                f"{metrics['cer']:.6f} | {metrics['wer']:.6f} | "
                f"{metrics['exact_match_accuracy']:.2%} |"
            )
    lines.extend(
        [
            "",
            "## Adaptive selector behavior",
            "",
            "- Selection counts and paired recognition outcomes conditioned on selected operation are in `operation_selection.csv` and `metrics.json`.",
            "",
            "| Adaptive arm | Keep original | CLAHE | Deskew |",
            "|---|---:|---:|---:|",
        ]
    )
    for arm in ("automatic", "user_guided"):
        counts = summary["arms"][arm]["operation_selection_counts"]
        lines.append(
            f"| {ARM_LABELS[arm]} | {counts.get('keep_original', 0)} "
            f"({counts.get('keep_original', 0) / summary['sample_count']:.2%}) | "
            f"{counts.get('clahe', 0)} "
            f"({counts.get('clahe', 0) / summary['sample_count']:.2%}) | "
            f"{counts.get('deskew', 0)} "
            f"({counts.get('deskew', 0) / summary['sample_count']:.2%}) |"
        )
    lines.extend(
        [
            "",
            "| Adaptive operation | N | CER | WER | Mean paired Delta CER | Improved / unchanged / degraded |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for result in summary["operation_outcomes_overall"]:
        if result["arm"] != "automatic":
            continue
        lines.append(
            f"| {result['operation']} | {result['sample_count']} | "
            f"{result['cer']:.6f} | {result['wer']:.6f} | "
            f"{result['mean_paired_delta_cer_vs_original']:+.6f} | "
            f"{result['improved_count']} / {result['unchanged_count']} / "
            f"{result['degraded_count']} |"
        )
    lines.extend(
        [
            "",
            f"- Automatic/user-guided exact operation agreement: "
            f"{validation['automatic_user_agreement_count']}/"
            f"{summary['sample_count']} "
            f"({validation['automatic_user_agreement_fraction']:.2%}).",
            f"- Transformation failures with original-image fallback: "
            f"{validation['transformation_failure_count']}.",
            "- Conditional outcome differences are observational and do not show that an operation caused an OCR change.",
            "",
            "## Failure analysis",
            "",
            f"- Transformation failures: {validation['transformation_failure_count']}; "
            "the predeclared original-image fallback was not needed.",
            "- Fixed global histogram equalization produced the largest aggregate regression: "
            f"CER {summary['arms']['fixed']['cer']:.6f} versus "
            f"{summary['arms']['original']['cer']:.6f} for the original arm.",
            "- The `visual_comparisons/failure_analysis/` panels show the worst observed "
            "adaptive CLAHE and deskew regressions, selected after evaluation for diagnosis "
            "only. They were not used to change any parameter or threshold.",
            "- A sampled Bengali histogram-equalization panel shows a visibly more mottled "
            "background and altered stroke edges; this is an illustrative failure, not a "
            "population-wide visual classification.",
            "",
            "## Statistical uncertainty",
            "",
            "- 95% percentile confidence intervals use 5,000 paired bootstrap replicates, resampling source-image IDs as clusters to account for multiple crops per board.",
            "- The reported statistic is the mean per-sample CER difference. The bootstrap seed is 7.",
            "- The run is a census of the remaining finite test population, so the reported test metrics are exact descriptive scores for that population. The cluster-bootstrap intervals describe hypothetical source-board sampling variability, not uncertainty in those finite-population scores.",
            "- One held-out dataset and one checkpoint do not establish broad generalization.",
            "",
            "## Implementation and integrity",
            "",
            f"- Original arm matched Phase 4B predictions exactly: "
            f"{validation['original_prediction_matches']}/"
            f"{validation['original_prediction_count']}.",
            f"- Every arm has {summary['sample_count']} predictions with identical filenames and ground truth.",
            f"- Source files unchanged: {validation['source_images_unchanged']}/"
            f"{validation['source_images_checked']}. Annotation, checkpoint, and vocabulary hashes were unchanged.",
            f"- Train/validation source-ID overlaps with test: "
            f"{validation['source_id_split_overlaps']}.",
            "- The complete remainder necessarily includes multiple crops from source boards; one-crop-per-board sampling would not represent the required population because it has only 612 source boards.",
            "",
            "## Limitations and supported claims",
            "",
            "- The 200 Phase 5A test samples were excluded. This evaluation used the remaining 7,894 CTC-feasible held-out test samples.",
            "- Results apply to this synth6k test remainder, the recovered CRNN checkpoint, its vocabulary/decoder, and the frozen preprocessing configuration.",
            "- No universal improvement, model-family generalization, or causal effect is claimed.",
            "- No test-set tuning, retraining, or changes to prior experiment artifacts were performed.",
            "",
            "## Visual comparisons",
            "",
            "Fixed-seed, language-stratified examples are saved under `visual_comparisons/`; these examples were selected independently of OCR outcome. The separate `failure_analysis/` examples were selected post hoc for error diagnosis only.",
            "",
        ]
    )
    return "\n".join(lines)


def run_final_evaluation() -> Path:
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("The frozen CRNN configuration requires CPU-only inference")
    dataset_document = _read_yaml(DATASET_CONFIG_PATH)
    preprocessing_document = _read_yaml(PREPROCESSING_CONFIG_PATH)
    model_document = _read_yaml(MODEL_CONFIG_PATH)
    pilot_document = _read_yaml(PILOT_CONFIG_PATH)
    pilot_config = pilot_document["phase5_pilot"]
    pilot_run_metadata = json.loads(
        (
            phase5a.OUTPUT_ROOT
            / "run-20261106T101321.309609Z"
            / "run_metadata.json"
        ).read_text(encoding="utf-8")
    )
    if int(pilot_config["seed"]) != SEED:
        raise ValueError("Phase 5A seed differs from the frozen Phase 5C seed")
    if int(pilot_config["batch_size"]) != 64:
        raise ValueError("Frozen experiment requires Phase 5A batch size 64")
    if float(model_document["models"]["crnn"]["contrast_factor"]) != 2.0:
        raise ValueError("CRNN native contrast factor must remain 2.0")

    image_root, metadata_path, dataset_configuration = phase5a.load_configured_paths()
    phase5a_inputs = pilot_run_metadata["inputs"]
    frozen_path_hashes = {
        "dataset_metadata": _sha256_file(metadata_path),
        "test_annotations": _sha256_file(
            PROJECT_ROOT / "synth6k" / "annotations" / "test.txt"
        ),
        "dataset_config": _sha256_file(DATASET_CONFIG_PATH),
        "preprocessing_config": _sha256_file(PREPROCESSING_CONFIG_PATH),
        "model_config": _sha256_file(MODEL_CONFIG_PATH),
        "pilot_config": _sha256_file(PILOT_CONFIG_PATH),
        "phase4b_predictions": _sha256_file(BASELINE_PREDICTIONS_PATH),
        "checkpoint_archive": _sha256_file(CHECKPOINT_ARCHIVE_PATH),
    }
    for key, actual_hash in frozen_path_hashes.items():
        expected_hash = (
            phase5a_inputs["checkpoint_archive"]["sha256_after"]
            if key == "checkpoint_archive"
            else phase5a_inputs[key]["sha256"]
        )
        if actual_hash != expected_hash:
            raise ValueError(
                f"Frozen Phase 5A input {key} has changed since the pilot; "
                "refusing to alter the experiment definition."
            )
    _, feasible_rows, sample_counts = phase5a.preflight_dataset(
        image_root,
        metadata_path,
        dataset_configuration,
    )
    if sample_counts["total_test"] != phase5a.EXPECTED_TOTAL_TEST_COUNT:
        raise ValueError(f"Unexpected test count: {sample_counts['total_test']}")
    if sample_counts["feasible_test"] != phase5a.EXPECTED_FEASIBLE_TEST_COUNT:
        raise ValueError(f"Unexpected feasible test count: {sample_counts['feasible_test']}")

    pilot_manifest = _read_csv(PILOT_MANIFEST_PATH)
    excluded_by_filename = {row["filename"]: row for row in pilot_manifest}
    if len(excluded_by_filename) != len(pilot_manifest) or len(pilot_manifest) != 200:
        raise ValueError("Phase 5A exclusion manifest must contain 200 unique filenames")
    feasible_by_filename = {row["filename"]: row for row in feasible_rows}
    if not set(excluded_by_filename) <= set(feasible_by_filename):
        raise ValueError("A Phase 5A exclusion filename is missing from feasible test data")
    for filename, excluded in excluded_by_filename.items():
        source = feasible_by_filename[filename]
        if (
            excluded["ground_truth"] != source["transcription"]
            or excluded["source_image_id"] != source["source_image_id"]
            or excluded["split"] != "test"
        ):
            raise ValueError(f"Phase 5A exclusion metadata mismatch for {filename}")

    selected_rows = [
        row for row in feasible_rows if row["filename"] not in excluded_by_filename
    ]
    if len(selected_rows) != EXPECTED_REMAINING:
        raise ValueError(
            f"Expected {EXPECTED_REMAINING} remaining feasible test samples; "
            f"found {len(selected_rows)}"
        )
    if any(row["split"].strip().lower() != "test" for row in selected_rows):
        raise ValueError("Non-test rows are present in the final sample")
    if len({row["filename"] for row in selected_rows}) != len(selected_rows):
        raise ValueError("Duplicate filenames are present in the final sample")
    for filename, excluded in excluded_by_filename.items():
        current_hash = _sha256_file(image_root / filename)
        if current_hash != excluded["original_image_sha256"]:
            raise ValueError(
                f"Excluded Phase 5A image changed since the pilot: {filename}"
            )

    all_metadata = phase5a._read_csv(metadata_path)
    source_ids_by_split = {
        split: {
            row["source_image_id"]
            for row in all_metadata
            if row["split"].strip().lower() == split
        }
        for split in ("train", "val", "test")
    }
    split_overlaps = {
        f"{left}_vs_{right}": len(source_ids_by_split[left] & source_ids_by_split[right])
        for left, right in (
            ("train", "val"),
            ("train", "test"),
            ("val", "test"),
        )
    }
    if any(split_overlaps.values()):
        raise ValueError(f"Source-board split leakage detected: {split_overlaps}")

    baseline_rows_list = _read_csv(BASELINE_PREDICTIONS_PATH)
    baseline_rows = {row["filename"]: row for row in baseline_rows_list}
    if len(baseline_rows) != len(baseline_rows_list):
        raise ValueError("Phase 4B predictions contain duplicate filenames")
    if set(baseline_rows) != {row["filename"] for row in feasible_rows}:
        raise ValueError("Phase 4B predictions do not match the feasible test population")
    for row in selected_rows:
        reference = baseline_rows[row["filename"]]
        if reference["ground_truth"] != row["transcription"]:
            raise ValueError(f"Phase 4B ground truth mismatch for {row['filename']}")

    baseline_contexts = phase5a._baseline_groups(
        feasible_rows,
        input_height=32,
        batch_size=int(pilot_config["batch_size"]),
    )
    source_hashes = {
        row["filename"]: _sha256_file(image_root / row["filename"])
        for row in selected_rows
    }
    exclusion_hash = _sha256_file(PILOT_MANIFEST_PATH)
    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S.%fZ")
    run_directory = OUTPUT_ROOT / run_id
    run_directory.mkdir(parents=True, exist_ok=False)

    excluded_rows: list[dict[str, Any]] = []
    for filename in sorted(excluded_by_filename):
        row = excluded_by_filename[filename]
        excluded_rows.append(
            {
                "filename": filename,
                "source_image_id": row["source_image_id"],
                "language": row["language"],
                "ground_truth": row["ground_truth"],
                "phase5a_sample_index": row["sample_index"],
                "source_image_sha256": row["original_image_sha256"],
                "excluded_from_phase5c": True,
            }
        )
    _write_csv(
        run_directory / "excluded_pilot_samples.csv",
        list(excluded_rows[0]),
        excluded_rows,
    )
    excluded_set = set(excluded_by_filename)
    if excluded_set & {row["filename"] for row in selected_rows}:
        raise RuntimeError("Pilot exclusion leakage detected in selected rows")

    configuration_snapshot = _configuration_snapshot(
        dataset_config=dataset_document,
        preprocessing_config=preprocessing_document,
        model_config=model_document,
        pilot_config=pilot_config,
        exclusion_path=run_directory / "excluded_pilot_samples.csv",
        exclusion_hash=_sha256_file(run_directory / "excluded_pilot_samples.csv"),
        exclusion_filenames=sorted(excluded_set),
    )
    snapshot_path = run_directory / "configuration_snapshot.yaml"
    snapshot_path.write_text(
        yaml.safe_dump(
            configuration_snapshot,
            allow_unicode=True,
            sort_keys=False,
            width=100,
        ),
        encoding="utf-8",
    )

    sample_manifest = _baseline_sample_manifest(
        selected_rows,
        baseline_rows,
        baseline_contexts,
        image_root,
        source_hashes,
    )
    _write_csv(
        run_directory / "sample_manifest.csv",
        list(sample_manifest[0]),
        sample_manifest,
    )
    if len(_read_csv(run_directory / "sample_manifest.csv")) != EXPECTED_REMAINING:
        raise RuntimeError("Frozen sample manifest was not persisted correctly")

    test_reference = PROJECT_ROOT / "synth6k" / "annotations" / "test.txt"
    integrity_before = {
        "metadata": _sha256_file(metadata_path),
        "test_annotation": _sha256_file(test_reference),
        "checkpoint_archive": _sha256_file(CHECKPOINT_ARCHIVE_PATH),
        "checkpoint_member": _hash_zip_member(
            CHECKPOINT_ARCHIVE_PATH,
            model_document["models"]["crnn"]["checkpoint_member"],
        ),
        "vocabulary_member": _hash_zip_member(
            CHECKPOINT_ARCHIVE_PATH,
            model_document["models"]["crnn"]["vocabulary_member"],
        ),
        "phase4b_predictions": _sha256_file(BASELINE_PREDICTIONS_PATH),
        "dataset_config": _sha256_file(DATASET_CONFIG_PATH),
        "preprocessing_config": _sha256_file(PREPROCESSING_CONFIG_PATH),
        "model_config": _sha256_file(MODEL_CONFIG_PATH),
        "pilot_config": _sha256_file(PILOT_CONFIG_PATH),
    }

    adapter = load_model_adapter(
        "crnn",
        config_path=MODEL_CONFIG_PATH,
        project_root=PROJECT_ROOT,
    )
    if not isinstance(adapter, CRNNAdapter):
        raise TypeError("Model registry did not return CRNNAdapter")
    torch.set_num_threads(2)
    if adapter.checkpoint_sha256 != configuration_snapshot["model"]["checkpoint_sha256"]:
        raise RuntimeError("Loaded checkpoint hash differs from frozen snapshot")
    if adapter.vocabulary_sha256 != configuration_snapshot["model"]["vocabulary_sha256"]:
        raise RuntimeError("Loaded vocabulary hash differs from frozen snapshot")

    visual_rng = random.Random(SEED)
    visual_samples: set[str] = set()
    for language in sorted({row["language"] for row in selected_rows}):
        candidates = [
            row["filename"] for row in selected_rows if row["language"] == language
        ]
        visual_samples.add(visual_rng.choice(candidates))

    predictions_by_arm: dict[str, dict[str, str]] = {}
    inference_seconds: dict[str, float] = {}
    inference_latency: dict[str, dict[str, float]] = {}
    transform_records: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        print(f"Starting arm {arm}: {ARM_LABELS[arm]}", flush=True)
        predictions, elapsed, latency_by_filename, records = _run_arm(
            arm=arm,
            adapter=adapter,
            selected_rows=selected_rows,
            baseline_contexts=baseline_contexts,
            image_root=image_root,
            run_directory=run_directory,
            source_hashes=source_hashes,
            pilot_config=pilot_config,
            preprocessing_config=preprocessing_document,
            dataset_config=dataset_document,
            visual_samples=visual_samples,
        )
        predictions_by_arm[arm] = predictions
        inference_seconds[arm] = elapsed
        inference_latency[arm] = latency_by_filename
        if arm == "original":
            mismatches = [
                row["filename"]
                for row in selected_rows
                if predictions[row["filename"]]
                != baseline_rows[row["filename"]]["prediction"]
            ]
            if mismatches:
                _write_json(
                    run_directory / "run_metadata.json",
                    {
                        "status": "aborted",
                        "reason": "Original arm failed exact Phase 4B prediction parity.",
                        "mismatch_count": len(mismatches),
                        "mismatch_examples": mismatches[:20],
                        "configuration_snapshot_sha256": _sha256_file(snapshot_path),
                    },
                )
                raise RuntimeError(
                    f"Original arm parity failed for {len(mismatches)} samples; "
                    "transformed arms were not started."
                )
        transform_records.extend(records)
        print(
            f"Completed arm {arm}: {len(predictions)} predictions in "
            f"{elapsed:.1f}s",
            flush=True,
        )

    for record in transform_records:
        record["inference_seconds"] = inference_latency[record["arm"]][
            record["filename"]
        ]
    summary, per_language, prediction_rows = _build_metrics(
        selected_rows,
        predictions_by_arm,
        transform_records,
    )
    for arm in ARM_ORDER:
        total_inference = inference_seconds[arm] * 1000
        summary["arms"][arm]["inference_latency_ms"] = {
            "total": total_inference,
            "mean_per_sample": total_inference / len(selected_rows),
        }
        preprocess_ms = summary["arms"][arm]["preprocessing_latency_ms"]["total"]
        summary["arms"][arm]["total_pipeline_latency_ms"] = {
            "total": preprocess_ms + total_inference,
            "mean_per_sample": (preprocess_ms + total_inference)
            / len(selected_rows),
        }

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
            "sample_cer",
            "paired_delta_cer_vs_original",
            "operation_selected",
        ],
        prediction_rows,
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
    operation_rows = _operation_selection_rows(transform_records, prediction_rows)
    _write_csv(
        run_directory / "operation_selection.csv",
        list(operation_rows[0]),
        operation_rows,
    )
    _write_json(run_directory / "transformation_log.json", transform_records)
    _write_json(run_directory / "metrics.json", summary)

    visual_predictions = {
        (row["arm"], row["filename"]): row for row in prediction_rows
    }
    selected_by_filename = {row["filename"]: row for row in selected_rows}
    for arm in ARM_ORDER[1:]:
        for filename in sorted(visual_samples):
            staging_path = (
                run_directory
                / "visual_comparisons"
                / ".staging"
                / arm
                / Path(filename).with_suffix(".png")
            )
            record = next(
                item
                for item in transform_records
                if item["arm"] == arm and item["filename"] == filename
            )
            if record["processed_image_sha256"] == record["original_image_sha256"]:
                processed_path = image_root / filename
            else:
                processed_path = staging_path
            _make_visual_comparison(
                source_path=image_root / filename,
                processed_path=processed_path,
                destination=(
                    run_directory
                    / "visual_comparisons"
                    / selected_by_filename[filename]["language"]
                    / f"{arm}-{filename}.png"
                ),
                arm=arm,
                row=selected_by_filename[filename],
                prediction=visual_predictions[(arm, filename)],
            )
    staging_root = run_directory / "visual_comparisons" / ".staging"
    if staging_root.exists():
        for path in sorted(staging_root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        staging_root.rmdir()
    failure_visuals = _generate_failure_visuals(
        run_directory=run_directory,
        image_root=image_root,
        selected_rows=selected_rows,
        transform_records=transform_records,
        prediction_rows=prediction_rows,
    )

    source_hashes_after = {
        row["filename"]: _sha256_file(image_root / row["filename"])
        for row in selected_rows
    }
    archive_hash_after = _sha256_file(CHECKPOINT_ARCHIVE_PATH)
    vocabulary_hash_after = _hash_zip_member(
        CHECKPOINT_ARCHIVE_PATH,
        model_document["models"]["crnn"]["vocabulary_member"],
    )
    checkpoint_hash_after = _hash_zip_member(
        CHECKPOINT_ARCHIVE_PATH,
        model_document["models"]["crnn"]["checkpoint_member"],
    )
    metadata_hash_after = _sha256_file(metadata_path)
    annotation_hash_after = _sha256_file(test_reference)
    integrity_after = {
        "phase4b_predictions": _sha256_file(BASELINE_PREDICTIONS_PATH),
        "dataset_config": _sha256_file(DATASET_CONFIG_PATH),
        "preprocessing_config": _sha256_file(PREPROCESSING_CONFIG_PATH),
        "model_config": _sha256_file(MODEL_CONFIG_PATH),
        "pilot_config": _sha256_file(PILOT_CONFIG_PATH),
        "pilot_manifest": _sha256_file(PILOT_MANIFEST_PATH),
    }
    source_unchanged_count = sum(
        source_hashes[filename] == source_hashes_after[filename]
        for filename in source_hashes
    )
    automatic_ops = {
        record["filename"]: record["operation_selected"]
        for record in transform_records
        if record["arm"] == "automatic"
    }
    user_ops = {
        record["filename"]: record["operation_selected"]
        for record in transform_records
        if record["arm"] == "user_guided"
    }
    agreement_count = sum(
        automatic_ops[filename] == user_ops[filename]
        for filename in automatic_ops
    )
    source_counts = Counter(row["source_image_id"] for row in selected_rows)
    validation = {
        "original_prediction_matches": len(selected_rows),
        "original_prediction_count": len(selected_rows),
        "same_filenames_across_arms": all(
            set(predictions_by_arm[arm]) == set(predictions_by_arm["original"])
            for arm in ARM_ORDER
        ),
        "same_ground_truth_across_arms": True,
        "pilot_exclusions_absent": not (
            excluded_set & {row["filename"] for row in selected_rows}
        ),
        "no_train_or_validation_samples": all(
            row["split"].strip().lower() == "test" for row in selected_rows
        ),
        "source_id_split_overlaps": split_overlaps,
        "unique_source_board_count": len(source_counts),
        "duplicate_source_board_excess_crops": sum(
            count - 1 for count in source_counts.values()
        ),
        "source_images_checked": len(selected_rows),
        "source_images_unchanged": source_unchanged_count,
        "annotation_files_unchanged": (
            integrity_before["metadata"] == metadata_hash_after
            and integrity_before["test_annotation"] == annotation_hash_after
        ),
        "frozen_configs_and_baselines_unchanged": all(
            integrity_before[key] == integrity_after[key]
            for key in (
                "phase4b_predictions",
                "dataset_config",
                "preprocessing_config",
                "model_config",
                "pilot_config",
            )
        )
        and integrity_after["pilot_manifest"] == exclusion_hash,
        "metadata_sha256_before": integrity_before["metadata"],
        "metadata_sha256_after": metadata_hash_after,
        "test_annotation_sha256_before": integrity_before["test_annotation"],
        "test_annotation_sha256_after": annotation_hash_after,
        "checkpoint_archive_unchanged": (
            integrity_before["checkpoint_archive"] == archive_hash_after
        ),
        "checkpoint_archive_sha256_before": integrity_before["checkpoint_archive"],
        "checkpoint_archive_sha256_after": archive_hash_after,
        "checkpoint_unchanged": (
            integrity_before["checkpoint_member"] == checkpoint_hash_after
            and checkpoint_hash_after == adapter.checkpoint_sha256
        ),
        "vocabulary_unchanged": (
            integrity_before["vocabulary_member"] == vocabulary_hash_after
            and vocabulary_hash_after == adapter.vocabulary_sha256
        ),
        "checkpoint_sha256_after": checkpoint_hash_after,
        "vocabulary_sha256_after": vocabulary_hash_after,
        "sample_count_per_arm": {
            arm: len(predictions_by_arm[arm]) for arm in ARM_ORDER
        },
        "transformation_failure_count": sum(
            record["transformation_status"] == "fallback_original"
            for record in transform_records
        ),
        "automatic_user_agreement_count": agreement_count,
        "automatic_user_agreement_fraction": agreement_count / len(selected_rows),
        "failure_visuals": failure_visuals,
        "no_training_performed": True,
        "pilot_test_samples_excluded": len(excluded_set),
        "final_test_samples": len(selected_rows),
    }
    required_checks = (
        validation["original_prediction_matches"]
        == validation["original_prediction_count"],
        validation["same_filenames_across_arms"],
        validation["same_ground_truth_across_arms"],
        validation["pilot_exclusions_absent"],
        validation["no_train_or_validation_samples"],
        validation["source_images_unchanged"] == validation["source_images_checked"],
        validation["annotation_files_unchanged"],
        validation["frozen_configs_and_baselines_unchanged"],
        validation["checkpoint_archive_unchanged"],
        validation["checkpoint_unchanged"],
        validation["vocabulary_unchanged"],
        all(count == EXPECTED_REMAINING for count in validation["sample_count_per_arm"].values()),
    )
    if not all(required_checks):
        raise RuntimeError(f"Final run integrity checks failed: {validation}")

    language_counts = dict(
        sorted(Counter(row["language"] for row in selected_rows).items())
    )
    snapshot_hash = _sha256_file(snapshot_path)
    run_metadata = {
        "status": "passed",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_command": (
            r".\.venv\Scripts\python.exe "
            r".\experiments\ocr_evaluation\run_phase5_final.py"
        ),
        "working_directory": str(PROJECT_ROOT),
        "dataset": "synth6k",
        "sample": {
            "split": "test",
            "ctc_feasible_population": len(feasible_rows),
            "phase5a_excluded": len(excluded_set),
            "final_sample_count": len(selected_rows),
            "seed": SEED,
            "selection_method": (
                "Full feasible test remainder after exact Phase 5A filename exclusion; "
                "Phase 4B order retained; no OCR-dependent resampling."
            ),
            "language_counts": language_counts,
            "source_board_count": len(source_counts),
            "one_crop_per_source_board_feasible": False,
            "source_id_split_overlaps": split_overlaps,
        },
        "frozen_configuration": {
            "path": str(snapshot_path),
            "sha256": snapshot_hash,
            "exclusion_manifest": str(run_directory / "excluded_pilot_samples.csv"),
            "sample_manifest": str(run_directory / "sample_manifest.csv"),
        },
        "model": {
            "adapter": "CRNNAdapter",
            "adapter_source_sha256": configuration_snapshot["model"][
                "adapter_source_sha256"
            ],
            "architecture": adapter.configuration["architecture"],
            "checkpoint_sha256": adapter.checkpoint_sha256,
            "vocabulary_sha256": adapter.vocabulary_sha256,
            "device": str(next(adapter.model.parameters()).device),
            "batch_size": adapter.batch_size,
            "torch_version": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "decoder": adapter.configuration["decoder"],
            "native_contrast_factor": adapter.contrast_factor,
        },
        "metrics": {
            "library": "JiWER",
            "version": version("jiwer"),
            "text_normalization": "none",
            "delta_cer": "CER_original - CER_experimental; positive is improvement",
            "confidence_interval": configuration_snapshot["metrics"][
                "confidence_interval"
            ],
        },
        "inputs": {
            "dataset_metadata_sha256": integrity_before["metadata"],
            "test_annotation_sha256": integrity_before["test_annotation"],
            "dataset_config_sha256": integrity_before["dataset_config"],
            "preprocessing_config_sha256": integrity_before["preprocessing_config"],
            "model_config_sha256": integrity_before["model_config"],
            "phase5a_config_sha256": integrity_before["pilot_config"],
            "phase5a_manifest_sha256": exclusion_hash,
            "phase4b_predictions_sha256": integrity_before["phase4b_predictions"],
            "checkpoint_archive_sha256": archive_hash_after,
            "frozen_configs_and_baselines_unchanged": validation[
                "frozen_configs_and_baselines_unchanged"
            ],
        },
        "validation": validation,
        "failure_visuals": failure_visuals,
        "latency_seconds": {
            arm: {
                "preprocessing": summary["arms"][arm][
                    "preprocessing_latency_ms"
                ]["total"]
                / 1000,
                "inference": inference_seconds[arm],
                "total_pipeline": (
                    summary["arms"][arm]["preprocessing_latency_ms"]["total"]
                    / 1000
                    + inference_seconds[arm]
                ),
            }
            for arm in ARM_ORDER
        },
        "transformation_failure_policy": (
            "Failed transform is recorded and that sample uses original pixels "
            "for the corresponding arm."
        ),
        "limitations": [
            "One CRNN checkpoint and one held-out synth6k test remainder; not family-wide evidence.",
            "Multiple crops per source board are present because the complete remainder contains 7894 crops from only 612 source boards.",
            "Cluster-bootstrap intervals describe mean per-sample CER differences under the source-board resampling assumptions.",
            "No test-set tuning, retraining, or prior-run modification.",
        ],
    }
    _write_json(run_directory / "run_metadata.json", run_metadata)
    metrics_text = _build_markdown(
        run_id=run_id,
        summary=summary,
        sample_counts=language_counts,
        exclusion_count=len(excluded_set),
        validation=validation,
    )
    (run_directory / "metrics.md").write_text(metrics_text, encoding="utf-8")
    (OUTPUT_ROOT / "final_evaluation_report.md").write_text(
        metrics_text
        + "\nRun artifacts are preserved in "
        + f"`{run_id}/`.\n",
        encoding="utf-8",
    )
    return run_directory


def main() -> int:
    run_directory = run_final_evaluation()
    print(f"Phase 5C evaluation completed: {run_directory}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
