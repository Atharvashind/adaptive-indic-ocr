from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import tempfile
import unicodedata
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import run_baseline_reproduction as baseline


ROOT = baseline.ROOT
RECOVERY_DIR = baseline.RECOVERY_DIR
RUN_DIR = (
    ROOT
    / "experiments"
    / "ocr_evaluation"
    / "baseline_reproduction"
    / "run-20261106T091015.067700Z"
)
ARCHIVE_PATH = baseline.ARCHIVE_PATH
METADATA_PATH = ROOT / "synth6k" / "metadata.csv"
REPORT_PATH = RECOVERY_DIR / "prediction_discrepancy_report.md"
JSON_PATH = RECOVERY_DIR / "prediction_discrepancy_report.json"
CHECKPOINT_AUDIT_PATH = RECOVERY_DIR / "checkpoint_load_test.json"
HISTORICAL_PREDICTIONS_MEMBER = baseline.HISTORICAL_PREDICTIONS_MEMBER
CHECKPOINT_MEMBER = baseline.CHECKPOINT_MEMBER
EXPECTED_CONTENT_DIFFERENCES = 10
EXPECTED_ALL_DIFFERENCES = 14
RUN_COMMAND = (
    r".\.venv\Scripts\python.exe "
    r".\experiments\ocr_evaluation\recovery\diagnose_prediction_discrepancies.py"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def string_edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def levenshtein_alignment(left: str, right: str) -> list[dict[str, Any]]:
    rows = len(left) + 1
    columns = len(right) + 1
    costs = [[0] * columns for _ in range(rows)]
    for i in range(rows):
        costs[i][0] = i
    for j in range(columns):
        costs[0][j] = j
    for i in range(1, rows):
        for j in range(1, columns):
            costs[i][j] = min(
                costs[i - 1][j] + 1,
                costs[i][j - 1] + 1,
                costs[i - 1][j - 1] + (left[i - 1] != right[j - 1]),
            )

    edits: list[dict[str, Any]] = []
    i, j = len(left), len(right)
    while i or j:
        if i and j and left[i - 1] == right[j - 1] and costs[i][j] == costs[i - 1][j - 1]:
            i -= 1
            j -= 1
        elif (
            i
            and j
            and costs[i][j] == costs[i - 1][j - 1] + 1
        ):
            edits.append(
                {
                    "operation": "replace",
                    "historical_index": i - 1,
                    "current_index": j - 1,
                    "historical_character": left[i - 1],
                    "current_character": right[j - 1],
                }
            )
            i -= 1
            j -= 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            edits.append(
                {
                    "operation": "delete_from_current",
                    "historical_index": i - 1,
                    "current_index": None,
                    "historical_character": left[i - 1],
                    "current_character": None,
                }
            )
            i -= 1
        else:
            edits.append(
                {
                    "operation": "insert_in_current",
                    "historical_index": None,
                    "current_index": j - 1,
                    "historical_character": None,
                    "current_character": right[j - 1],
                }
            )
            j -= 1
    edits.reverse()
    return edits


def normalization_views(text: str) -> dict[str, str]:
    return {
        "raw": text,
        "NFC": unicodedata.normalize("NFC", text),
        "strip_only": text.strip(),
        "NFC_then_strip": unicodedata.normalize("NFC", text).strip(),
        "NFC_strip_and_collapse_whitespace": re.sub(
            r"\s+",
            " ",
            unicodedata.normalize("NFC", text).strip(),
        ),
    }


def visible_spaces(text: str) -> str:
    return text.replace(" ", "␠").replace("\t", "⇥").replace("\n", "\\n")


def label_for_class(class_index: int, index_to_character: dict[int, str]) -> str:
    if class_index == baseline.BLANK_INDEX:
        return "<BLANK>"
    character = index_to_character.get(class_index)
    return character if character is not None else f"<CLASS:{class_index}>"


def summarize_one_sample(
    filename: str,
    reference: torch.Tensor,
    replay: torch.Tensor,
    index_to_character: dict[int, str],
    current_prediction: str,
) -> dict[str, Any]:
    if reference.shape != replay.shape:
        raise RuntimeError(f"Replay logits shape changed for {filename}")
    difference = torch.abs(reference - replay)
    indices_a = reference.argmax(dim=-1)
    indices_b = replay.argmax(dim=-1)
    differing_timesteps = torch.nonzero(indices_a != indices_b).flatten().tolist()
    top_values, top_indices = torch.topk(reference, k=2, dim=-1)
    margins = top_values[:, 0] - top_values[:, 1]
    margin_values = margins.cpu().tolist()
    low_margin_count = max(1, math.ceil(len(margin_values) * 0.05))
    low_margin_timesteps = sorted(
        range(len(margin_values)),
        key=lambda time_index: (margin_values[time_index], time_index),
    )[:low_margin_count]
    decoded_characters = []
    previous = -1
    for timestep, raw_class in enumerate(indices_a.tolist()):
        class_index = int(raw_class)
        if class_index != baseline.BLANK_INDEX and class_index != previous:
            decoded_characters.append(
                {
                    "character": index_to_character[class_index],
                    "timestep": timestep,
                    "top1_top2_logit_margin": float(margin_values[timestep]),
                    "margin_percentile_in_sample": (
                        sum(value <= margin_values[timestep] for value in margin_values)
                        / len(margin_values)
                    ),
                }
            )
        previous = class_index
    decoded = "".join(item["character"] for item in decoded_characters)
    if decoded != current_prediction:
        raise RuntimeError(
            f"Determinism replay output for {filename} differs from full-run saved prediction"
        )

    class_sequence = [int(value) for value in indices_a.tolist()]
    blank_transition_timesteps = [
        timestep
        for timestep in range(1, len(class_sequence))
        if (class_sequence[timestep] == baseline.BLANK_INDEX)
        != (class_sequence[timestep - 1] == baseline.BLANK_INDEX)
    ]
    return {
        "filename": filename,
        "logits_shape": list(reference.shape),
        "logits_dtype": str(reference.dtype),
        "replay_comparison": {
            "maximum_absolute_logit_difference": float(difference.max().item()),
            "mean_absolute_logit_difference": float(difference.mean().item()),
            "raw_logits_bitwise_equal": bool(torch.equal(reference, replay)),
            "argmax_sequences_equal": not differing_timesteps,
            "argmax_differing_timesteps": differing_timesteps,
            "argmax_indices_run_1": [int(value) for value in indices_a.tolist()],
            "argmax_indices_run_2": [int(value) for value in indices_b.tolist()],
        },
        "current_logits_margin_analysis": {
            "minimum_top1_top2_logit_margin": float(margins.min().item()),
            "median_top1_top2_logit_margin": float(margins.median().item()),
            "p05_top1_top2_logit_margin": float(
                torch.quantile(margins, 0.05).item()
            ),
            "p95_top1_top2_logit_margin": float(
                torch.quantile(margins, 0.95).item()
            ),
            "lowest_margin_timesteps": [
                {
                    "timestep": timestep,
                    "predicted_class": int(top_indices[timestep, 0].item()),
                    "predicted_label": label_for_class(
                        int(top_indices[timestep, 0].item()),
                        index_to_character,
                    ),
                    "runner_up_class": int(top_indices[timestep, 1].item()),
                    "runner_up_label": label_for_class(
                        int(top_indices[timestep, 1].item()),
                        index_to_character,
                    ),
                    "top1_logit": float(top_values[timestep, 0].item()),
                    "runner_up_logit": float(top_values[timestep, 1].item()),
                    "margin": float(margin_values[timestep]),
                    "blank_transition": timestep in blank_transition_timesteps,
                }
                for timestep in low_margin_timesteps
            ],
        },
        "current_decode_analysis": {
            "decoded_text": decoded,
            "blank_transition_count": len(blank_transition_timesteps),
            "blank_transition_timesteps": blank_transition_timesteps,
            "emitted_characters_and_timesteps": decoded_characters,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose historical vs CPU OCR predictions.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the two discrepancy report files after verifying the same baseline run.",
    )
    arguments = parser.parse_args()
    if not arguments.overwrite and (REPORT_PATH.exists() or JSON_PATH.exists()):
        raise FileExistsError(
            "Discrepancy report already exists; pass --overwrite to replace this diagnostic report."
        )
    if not RUN_DIR.is_dir():
        raise FileNotFoundError(f"Phase 4B run directory missing: {RUN_DIR}")
    if file_sha256(ARCHIVE_PATH) != json.loads(
        (RUN_DIR / "run_metadata.json").read_text(encoding="utf-8")
    )["inputs"]["archive"]["sha256_before"]:
        raise ValueError("Recovered checkpoint archive hash differs from Phase 4B run")

    run_metadata = json.loads((RUN_DIR / "run_metadata.json").read_text(encoding="utf-8"))
    predictions_rows = baseline.read_csv(RUN_DIR / "predictions.csv", encoding="utf-8")
    historical_report_rows: list[dict[str, str]]
    with zipfile.ZipFile(ARCHIVE_PATH, "r") as archive:
        historical_report_rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read(HISTORICAL_PREDICTIONS_MEMBER).decode("utf-8-sig")
                )
            )
        )
    saved_by_filename = {row["filename"]: row for row in predictions_rows}
    historic_by_filename = {
        row["filename"]: row for row in historical_report_rows
    }
    if len(saved_by_filename) != len(predictions_rows):
        raise ValueError("Phase 4B prediction file contains duplicate filenames")
    if set(saved_by_filename) != set(historic_by_filename):
        raise ValueError("Historical and reproduced prediction sample sets differ")

    differences: list[dict[str, Any]] = []
    for filename, current in saved_by_filename.items():
        historic = historic_by_filename[filename]
        if historic["prediction"] == current["prediction"]:
            continue
        historical_prediction = historic["prediction"]
        current_prediction = current["prediction"]
        whitespace_only = historical_prediction.strip() == current_prediction.strip()
        differences.append(
            {
                "filename": filename,
                "language": current["language"],
                "ground_truth": current["ground_truth"],
                "historical_prediction": historical_prediction,
                "current_cpu_prediction": current_prediction,
                "difference_type": "whitespace_only" if whitespace_only else "content",
                "whitespace_only": whitespace_only,
                "character_edit_distance_historical_to_current": string_edit_distance(
                    historical_prediction,
                    current_prediction,
                ),
                "character_edit_distance_ground_truth_to_historical": string_edit_distance(
                    current["ground_truth"],
                    historical_prediction,
                ),
                "character_edit_distance_ground_truth_to_current": string_edit_distance(
                    current["ground_truth"],
                    current_prediction,
                ),
                "historical_to_current_alignment": levenshtein_alignment(
                    historical_prediction,
                    current_prediction,
                ),
                "normalization_comparison": {
                    "historical": normalization_views(historical_prediction),
                    "current": normalization_views(current_prediction),
                    "equal_after": {
                        key: normalization_views(historical_prediction)[key]
                        == normalization_views(current_prediction)[key]
                        for key in normalization_views(historical_prediction)
                    },
                    "diagnostic_only_not_applied_to_metrics": True,
                },
                "model_output_comparison": {
                    "historical_logits_available": False,
                    "historical_argmax_timesteps_available": False,
                    "historical_prediction_is_decoded_string_only": True,
                },
            }
        )
    if len(differences) != EXPECTED_ALL_DIFFERENCES:
        raise ValueError(f"Expected {EXPECTED_ALL_DIFFERENCES} prediction differences")
    content_differences = [
        item for item in differences if item["difference_type"] == "content"
    ]
    whitespace_differences = [
        item for item in differences if item["difference_type"] == "whitespace_only"
    ]
    if len(content_differences) != EXPECTED_CONTENT_DIFFERENCES:
        raise ValueError(
            f"Expected {EXPECTED_CONTENT_DIFFERENCES} content-level differences"
        )

    metadata_rows = baseline.read_csv(METADATA_PATH, encoding="utf-8-sig")
    all_test_rows = [
        row for row in metadata_rows if row["split"].strip().lower() == "test"
    ]
    test_rows = [
        row
        for row in all_test_rows
        if baseline.calculate_feasibility(row)[2]
    ]
    if len(test_rows) != baseline.EXPECTED_FEASIBLE_TEST_COUNT:
        raise ValueError("Feasibility count changed from the audited Phase 4B dataset")
    by_filename = {row["filename"]: row for row in test_rows}
    content_names = {item["filename"] for item in content_differences}
    if not content_names.issubset(by_filename):
        raise ValueError("Content-difference samples are absent from feasible test data")
    image_root = Path(run_metadata["dataset"]["image_root"])

    sample_dimensions: dict[str, dict[str, Any]] = {}
    for item in differences:
        row = by_filename[item["filename"]]
        image_path = image_root / item["filename"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Diagnostic image missing: {image_path}")
        with baseline.Image.open(image_path) as image:
            original_width, original_height = image.size
        resized_width = int(round(original_width * baseline.INPUT_HEIGHT / original_height))
        sample_dimensions[item["filename"]] = {
            "original_width": original_width,
            "original_height": original_height,
            "resized_width": resized_width,
            "resized_height": baseline.INPUT_HEIGHT,
            "metadata_crop_width": int(float(row["crop_width"])),
            "metadata_crop_height": int(float(row["crop_height"])),
        }
        item["dimensions"] = sample_dimensions[item["filename"]]

    checkpoint_audit = json.loads(CHECKPOINT_AUDIT_PATH.read_text(encoding="utf-8"))
    expected_checkpoint = next(
        item
        for item in checkpoint_audit["checkpoints"]
        if item["name"] == "best_model_epoch4.pt"
    )
    with zipfile.ZipFile(ARCHIVE_PATH, "r") as archive:
        with tempfile.TemporaryDirectory(prefix="crnn_discrepancy_diagnostic_") as temp_dir:
            checkpoint_path = Path(temp_dir) / "best_model_epoch4.pt"
            checkpoint_path.write_bytes(archive.read(CHECKPOINT_MEMBER))
            if file_sha256(checkpoint_path) != expected_checkpoint["sha256_before"]:
                raise ValueError("Diagnostic checkpoint hash differs from Phase 4A")
            model = baseline.CRNN(num_classes=baseline.OUTPUT_CLASSES).to("cpu")
            state_dict = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            if next(model.parameters()).dtype != torch.float32:
                raise ValueError("Expected float32 model parameters")

    with zipfile.ZipFile(ARCHIVE_PATH, "r") as archive:
        vocabulary_bytes = archive.read(baseline.VOCABULARY_MEMBER)
    vocabulary = json.loads(vocabulary_bytes.decode("utf-8"))
    _, index_to_character = baseline.verify_vocabulary(
        vocabulary,
        metadata_rows,
        test_rows,
    )
    dataset = baseline.HistoricalCRNNDataset(test_rows, image_root, vocabulary)
    position_by_filename = {row["filename"]: index for index, row in enumerate(test_rows)}
    target_batches = {
        position_by_filename[filename] // baseline.BATCH_SIZE
        for filename in content_names
    }
    cpu_replays: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for batch_index in sorted(target_batches):
            start = batch_index * baseline.BATCH_SIZE
            stop = min(start + baseline.BATCH_SIZE, len(dataset))
            batch_items = [dataset[index] for index in range(start, stop)]
            collated = baseline.historical_collate(batch_items)
            images = collated[0]
            filenames = collated[5]
            selected = content_names.intersection(filenames)
            if not selected:
                continue
            logits_first = model(images)
            if not torch.isfinite(logits_first).all():
                raise ValueError("First CPU inference returned non-finite logits")
            first_target_logits = {
                filename: logits_first[:, filenames.index(filename), :].contiguous()
                for filename in selected
            }
            del logits_first
            logits_second = model(images)
            if not torch.isfinite(logits_second).all():
                raise ValueError("Second CPU inference returned non-finite logits")
            for filename in selected:
                sample_index = filenames.index(filename)
                result = summarize_one_sample(
                    filename,
                    first_target_logits[filename],
                    logits_second[:, sample_index, :].contiguous(),
                    index_to_character,
                    saved_by_filename[filename]["prediction"],
                )
                result["original_batch_context"] = {
                    "batch_index": batch_index,
                    "batch_size": len(filenames),
                    "padded_batch_width": collated[8],
                    "content_width": int(
                        sample_dimensions[filename]["resized_width"]
                    ),
                    "batch_filenames_in_collated_order": filenames,
                }
                cpu_replays[filename] = result
            del first_target_logits, logits_second, images

    if set(cpu_replays) != content_names:
        raise RuntimeError("Repeat inference did not cover every content-level difference")

    for item in content_differences:
        replay = cpu_replays[item["filename"]]
        emitted = replay["current_decode_analysis"]["emitted_characters_and_timesteps"]
        timesteps_by_character_index = {
            index: emitted[index]["timestep"] for index in range(len(emitted))
        }
        margin_by_timestep = {
            entry["timestep"]: entry["top1_top2_logit_margin"]
            for entry in emitted
        }
        for edit in item["historical_to_current_alignment"]:
            current_index = edit["current_index"]
            timestep = timesteps_by_character_index.get(current_index) if current_index is not None else None
            edit["current_cpu_timestep"] = timestep
            edit["current_cpu_top1_top2_logit_margin"] = (
                margin_by_timestep.get(timestep) if timestep is not None else None
            )
            edit["current_cpu_margin_percentile_in_sample"] = (
                emitted[current_index]["margin_percentile_in_sample"]
                if current_index is not None
                else None
            )
            edit["near_blank_transition"] = (
                timestep in replay["current_decode_analysis"]["blank_transition_timesteps"]
                if timestep is not None
                else None
            )

    normalization_counts = {
        key: sum(item["normalization_comparison"]["equal_after"][key] for item in differences)
        for key in normalization_views("")
    }
    deterministic_count = sum(
        item["replay_comparison"]["raw_logits_bitwise_equal"]
        and item["replay_comparison"]["argmax_sequences_equal"]
        for item in cpu_replays.values()
    )
    mapped_text_edits = [
        edit
        for item in content_differences
        for edit in item["historical_to_current_alignment"]
        if edit["current_cpu_timestep"] is not None
    ]
    mapped_low_margin_edits = [
        edit
        for edit in mapped_text_edits
        if edit["current_cpu_margin_percentile_in_sample"] <= 0.05
    ]
    result: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_command": RUN_COMMAND,
        "baseline_run_id": run_metadata["run_id"],
        "summary": {
            "evaluated_samples": 8094,
            "historical_vs_current_prediction_differences": len(differences),
            "content_level_differences": len(content_differences),
            "whitespace_only_differences": len(whitespace_differences),
            "content_difference_languages": dict(
                sorted(Counter(item["language"] for item in content_differences).items())
            ),
            "replayed_content_samples": len(cpu_replays),
            "bitwise_identical_replays": deterministic_count,
            "internally_deterministic_for_all_replayed_samples": (
                deterministic_count == len(cpu_replays)
            ),
            "historical_logits_available": False,
            "normalization_equalities_across_all_differences": normalization_counts,
            "content_edits_mapped_to_current_emissions": len(mapped_text_edits),
            "mapped_edits_within_sample_lowest_5_percent_margin": len(
                mapped_low_margin_edits
            ),
            "mapped_edits_at_blank_transition": sum(
                bool(edit["near_blank_transition"]) for edit in mapped_text_edits
            ),
        },
        "historical_artifact_availability": {
            "available": [
                "decoded predictions in test_predictions_log.csv",
                "ground-truth strings",
                "sample filenames and language labels",
                "aggregate historical CER/WER report",
            ],
            "unavailable": [
                "historical per-sample logits",
                "historical per-timestep argmax sequences",
                "historical top-two logits or margins",
                "historical batch tensors or per-sample batch membership",
                "actual runtime/device library versions for the archived evaluation",
            ],
            "archive_search_for_logits_members": [],
        },
        "numerical_environment": {
            "python_version": __import__("sys").version,
            "torch_version": torch.__version__,
            "torch_build": torch.__config__.show(),
            "device": "cpu",
            "cpu_processor": __import__("platform").processor(),
            "cpu_count": __import__("os").cpu_count(),
            "model_dtype": str(next(model.parameters()).dtype),
            "input_dtype": "torch.float32",
            "mkldnn_available": torch.backends.mkldnn.is_available(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "settings_changed_for_primary_replay": False,
        },
        "normalization_comparison": {
            "method": "diagnostic-only string comparison; never used by original decoding or metrics",
            "variants": [
                "raw",
                "NFC",
                "strip_only",
                "NFC_then_strip",
                "NFC_strip_and_collapse_whitespace",
            ],
            "equalities_across_all_14": normalization_counts,
        },
        "classification": {
            "confirmed_deterministic_behavior": (
                deterministic_count == len(cpu_replays)
            ),
            "confirmed_nondeterministic_behavior": False,
            "numerical_sensitivity_near_argmax_boundaries": (
                f"Current-run evidence: {len(mapped_low_margin_edits)}/{len(mapped_text_edits)} "
                "current emitted characters aligned to historical/current text edits fall within "
                "their sample's lowest 5% top-1/top-2 logit margins. This is descriptive evidence "
                "of low-margin decisions in the CPU path, not proof of the historical cause because "
                "historical logits/argmax paths are unavailable."
            ),
            "preprocessing_mismatch": (
                "No mismatch identified in available source: the CPU replay uses the same Phase 4B dataset/preprocessing/collation functions and original batch context; historical input tensors were not archived."
            ),
            "decoder_mismatch": (
                "No mismatch identified in available source: the same audited greedy decoder is used; historical per-timestep paths are unavailable for direct verification."
            ),
            "overall_cause": (
                "Unresolved between historical-vs-CPU model execution details. CPU inference is bitwise deterministic on repeat for the ten replayed samples, but that cannot identify why the historical decoded outputs differ."
            ),
        },
        "differing_samples": differences,
        "cpu_repeat_inference": [cpu_replays[name] for name in sorted(cpu_replays)],
        "source_integrity": {
            "archive_sha256": file_sha256(ARCHIVE_PATH),
            "phase4b_checkpoint_sha256": run_metadata["checkpoint"]["sha256"],
            "phase4b_prediction_csv_sha256": file_sha256(
                RUN_DIR / "predictions.csv"
            ),
            "phase4b_predictions_unchanged": True,
        },
    }
    JSON_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(build_markdown(result), encoding="utf-8")
    print(f"Report: {REPORT_PATH}")
    print(f"JSON: {JSON_PATH}")
    print(
        f"Differences: {len(differences)} total; {len(content_differences)} content; "
        f"{len(whitespace_differences)} whitespace-only"
    )
    print(
        f"Deterministic replays: {deterministic_count}/{len(cpu_replays)} "
        f"content-difference samples"
    )
    return 0


def build_markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    environment = result["numerical_environment"]
    lines = [
        "# Historical vs CPU baseline prediction discrepancy report",
        "",
        f"- **Created:** {result['created_at_utc']}",
        f"- **Baseline run:** `{result['baseline_run_id']}`",
        f"- **Diagnostic command:** `{result['run_command']}`",
        "",
        "## Summary",
        "",
        f"- Evaluated samples: {summary['evaluated_samples']}",
        f"- Prediction differences: {summary['historical_vs_current_prediction_differences']}",
        f"- Content-level: {summary['content_level_differences']}",
        f"- Whitespace-only: {summary['whitespace_only_differences']}",
        f"- Content differences by language: `{summary['content_difference_languages']}`",
        f"- Bitwise identical CPU replays: {summary['bitwise_identical_replays']}/{summary['replayed_content_samples']}",
        f"- CPU inference deterministic for all replayed cases: {summary['internally_deterministic_for_all_replayed_samples']}",
        f"- Text edits mapped to current emissions: {summary['content_edits_mapped_to_current_emissions']}; among per-sample lowest-5% margins: {summary['mapped_edits_within_sample_lowest_5_percent_margin']}; at a blank transition: {summary['mapped_edits_at_blank_transition']}",
        "",
        "## Numerical environment",
        "",
        f"- Python: `{environment['python_version']}`",
        f"- PyTorch: `{environment['torch_version']}`",
        f"- CPU: `{environment['cpu_processor']}`; logical CPU count {environment['cpu_count']}",
        f"- CPU backend: oneDNN/MKLDNN available={environment['mkldnn_available']}, enabled={environment['mkldnn_enabled']}",
        f"- Dtypes: model `{environment['model_dtype']}`, input `{environment['input_dtype']}`",
        f"- Deterministic algorithms enabled: {environment['deterministic_algorithms_enabled']}",
        f"- Threads: intra-op {environment['num_threads']}, inter-op {environment['num_interop_threads']}",
        f"- Float32 matmul precision: `{environment['float32_matmul_precision']}`",
        "- Backend and precision settings were observed, not changed.",
        "",
        "## Historical logits availability",
        "",
        "The recovered archive contains decoded strings and aggregate metrics, but no per-sample logits, timestep argmax sequences, top-two margins, batch tensors, or runtime/device library versions. Therefore historical-vs-CPU logit deltas and timestep-level argmax comparisons cannot be measured. The report records these as unavailable rather than deriving them from decoded strings.",
        "",
        "The ten content-level samples were replayed twice on CPU using their original batch membership and collated order from the Phase 4B run. For each, the JSON includes both full argmax sequences, per-sample logit max/mean absolute differences, differing timestep indices, top-two margin statistics, lowest-margin timesteps, blank transitions, and emitted character/timestep alignment. Each replay’s decoded string was checked against the saved Phase 4B prediction.",
        "",
        "## All differing predictions",
        "",
        "| Filename | Language | Type | Edit distance (historical→CPU) | Original size | Resized size |",
        "|---|---|---|---:|---:|---:|",
    ]
    for item in result["differing_samples"]:
        dimensions = item["dimensions"]
        lines.append(
            f"| `{item['filename']}` | {item['language']} | {item['difference_type']} | "
            f"{item['character_edit_distance_historical_to_current']} | "
            f"{dimensions['original_width']}×{dimensions['original_height']} | "
            f"{dimensions['resized_width']}×{dimensions['resized_height']} |"
        )
    for item in result["differing_samples"]:
        lines.extend(
            [
                "",
                f"### {item['filename']} ({item['language']}, {item['difference_type']})",
                "",
                f"- Historical (spaces rendered as ␠): `{visible_spaces(item['historical_prediction'])}`",
                f"- Current CPU (spaces rendered as ␠): `{visible_spaces(item['current_cpu_prediction'])}`",
                f"- Ground truth (spaces rendered as ␠): `{visible_spaces(item['ground_truth'])}`",
                f"- Whitespace-only: {item['whitespace_only']}",
                f"- Edit distance, historical→CPU: {item['character_edit_distance_historical_to_current']}",
                f"- Edit distance, ground truth→historical/current: "
                f"{item['character_edit_distance_ground_truth_to_historical']} / "
                f"{item['character_edit_distance_ground_truth_to_current']}",
                f"- Dimensions: original {item['dimensions']['original_width']}×"
                f"{item['dimensions']['original_height']}; resized "
                f"{item['dimensions']['resized_width']}×{item['dimensions']['resized_height']}",
                f"- Diagnostic normalization equality: `{item['normalization_comparison']['equal_after']}` "
                "(none of these transformations is applied to evaluation output).",
            ]
        )

    lines.extend(["", "## Repeat-inference and logit analysis", ""])
    for item in result["cpu_repeat_inference"]:
        replay = item["replay_comparison"]
        margin = item["current_logits_margin_analysis"]
        decode = item["current_decode_analysis"]
        lowest_margins = margin["lowest_margin_timesteps"][:5]
        compact_margins = [
            (
                entry["timestep"],
                entry["predicted_label"],
                entry["runner_up_label"],
                round(entry["margin"], 8),
                entry["blank_transition"],
            )
            for entry in lowest_margins
        ]
        lines.extend(
            [
                f"### {item['filename']}",
                "",
                f"- Original batch index/size/padded width: "
                f"{item['original_batch_context']['batch_index']} / "
                f"{item['original_batch_context']['batch_size']} / "
                f"{item['original_batch_context']['padded_batch_width']}",
                f"- Input content width: {item['original_batch_context']['content_width']}; "
                f"logit shape: `{item['logits_shape']}` ({item['logits_dtype']})",
                f"- CPU repeat logits bitwise equal: {replay['raw_logits_bitwise_equal']}",
                f"- Max/mean absolute logit difference: "
                f"{replay['maximum_absolute_logit_difference']:.9g} / "
                f"{replay['mean_absolute_logit_difference']:.9g}",
                f"- Repeat argmax equal: {replay['argmax_sequences_equal']}; "
                f"differing timesteps: `{replay['argmax_differing_timesteps']}`",
                f"- Current top-1/top-2 margin min/median/p05: "
                f"{margin['minimum_top1_top2_logit_margin']:.9g} / "
                f"{margin['median_top1_top2_logit_margin']:.9g} / "
                f"{margin['p05_top1_top2_logit_margin']:.9g}",
                f"- Blank/nonblank transitions: {decode['blank_transition_count']} "
                "(per-timestep locations are in the JSON report)",
                "- Five lowest top-1/top-2 margins (timestep, winner, runner-up, margin, blank transition): "
                f"`{compact_margins}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Normalization-only diagnostic",
            "",
            "Raw decoded output is unchanged for the baseline metrics. Diagnostic comparisons were made with raw strings, Unicode NFC, `strip()`, NFC followed by `strip()`, and NFC/strip/whitespace-collapse. Across all 14 differences, equality counts by variant are "
            f"`{summary['normalization_equalities_across_all_differences']}`. "
            "The four previously identified whitespace-only pairs become equal under strip; no normalization is used to replace predictions or recalculate metrics.",
            "",
            "## Evidence-based conclusion",
            "",
            "- **Confirmed deterministic behavior:** Yes for the ten replayed content-difference samples: repeated CPU inference in unchanged settings produced bitwise-identical logits, identical argmax sequences, and identical decoded text.",
            "- **Confirmed nondeterministic behavior:** No evidence in this replay.",
            f"- **Numerical sensitivity:** In the current CPU path, {summary['mapped_edits_within_sample_lowest_5_percent_margin']}/{summary['content_edits_mapped_to_current_emissions']} text edits that map to current emitted characters fall in their sample's lowest 5% top-1/top-2 margins. This supports low-margin sensitivity in the current path, but does not prove that historical hardware changed those argmax decisions.",
            "- **Preprocessing mismatch:** Not indicated by the available source audit and replay; historical preprocessed tensors are unavailable for direct comparison.",
            "- **Decoder mismatch:** Not indicated by source; historical timestep paths are unavailable for a direct equivalence test.",
            "- **Overall historical-vs-CPU cause:** Unresolved. The saved historical output differs from current CPU output for ten content cases, but archived evidence cannot prove whether hardware/backend arithmetic, historical runtime versions, or another unrecorded execution detail caused those changes.",
            "",
            "The notebook metadata identifies a GPU accelerator, but the historical evaluation’s actual device and numerical libraries were not recorded. That metadata alone does not prove which device produced the saved test predictions.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
