from __future__ import annotations

import csv
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RECOVERY_DIR = ROOT / "experiments" / "ocr_evaluation" / "recovery"
BASELINE_DIR = (
    ROOT
    / "experiments"
    / "ocr_evaluation"
    / "baseline_reproduction"
    / "run-20261106T091015.067700Z"
)
BASELINE_PREDICTIONS_PATH = BASELINE_DIR / "predictions.csv"
BASELINE_METRICS_PATH = BASELINE_DIR / "metrics.json"
REPORT_JSON_PATH = RECOVERY_DIR / "crnn_adapter_integration_report.json"
REPORT_MARKDOWN_PATH = RECOVERY_DIR / "crnn_adapter_integration_report.md"
EXPECTED_SAMPLES = 8094

from run_baseline_reproduction import (  # noqa: E402
    ARCHIVE_PATH,
    calculate_feasibility,
    load_configured_paths,
    preflight_dataset,
    sha256_file,
    verify_vocabulary,
)
from src.evaluation.cer_wer import calculate_cer_wer  # noqa: E402
from src.evaluation.runner import OCRExample, run_inference  # noqa: E402
from src.models import CRNNAdapter, load_model_adapter  # noqa: E402


def _read_prediction_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Phase 4B predictions are missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_SAMPLES:
        raise ValueError(f"Expected {EXPECTED_SAMPLES} baseline rows, found {len(rows)}")
    if len({row["filename"] for row in rows}) != len(rows):
        raise ValueError("Phase 4B predictions contain duplicate filenames")
    return rows


def _build_markdown(report: dict[str, Any]) -> str:
    comparison = report["equivalence"]
    lines = [
        "# CRNN adapter integration validation",
        "",
        f"- **Status:** {report['status']}",
        f"- **Command:** `{report['command']}`",
        f"- **Adapter:** `{report['architecture']}`",
        f"- **Checkpoint SHA-256:** `{report['checkpoint_sha256']}`",
        f"- **Vocabulary SHA-256:** `{report['vocabulary_sha256']}`",
        f"- **Device:** {report['device']}",
        f"- **Samples evaluated:** {report['samples_evaluated']}",
        f"- **CTC-infeasible test samples excluded:** "
        f"{report['infeasible_test_samples_excluded']}",
        f"- **Inference time:** {report['inference_seconds']:.3f} seconds",
        "",
        "## Phase 4B comparison",
        "",
        "| Check | Result |",
        "|---|---:|",
        f"| Same sample order | {comparison['same_sample_order']} |",
        f"| Same filenames, ground truth, source IDs, and languages | "
        f"{comparison['same_metadata']} |",
        f"| Exact prediction matches | {comparison['prediction_matches']}/"
        f"{comparison['prediction_count']} |",
        f"| CER | {comparison['adapter_cer']:.12f} "
        f"(baseline {comparison['baseline_cer']:.12f}) |",
        f"| WER | {comparison['adapter_wer']:.12f} "
        f"(baseline {comparison['baseline_wer']:.12f}) |",
        f"| CER exact match | {comparison['cer_matches']} |",
        f"| WER exact match | {comparison['wer_matches']} |",
        "",
        "## Verified inference contract",
        "",
        "- Safe CPU checkpoint deserialization uses `torch.load(..., map_location='cpu', weights_only=True)`.",
        "- The recovered notebook cell-94 CRNN architecture is loaded with `strict=True`.",
        "- The checkpoint and exact training-derived vocabulary are SHA-256 checked against the configured recovered ZIP members.",
        "- The adapter preserves grayscale conversion, height-32 aspect-preserving PIL bilinear resize, contrast factor 2.0, float32 `[0,1]` scaling, and normalization to `[-1,1]`.",
        "- Batch-local descending-width sorting, right padding to a multiple of four with normalized zero, `(content_width // 4) - 1` input lengths, blank index 0, and the original greedy CTC decoder are adapter-owned.",
        "- The shared evaluator handles only batches of image paths and transcription records; it has no CRNN tensor or CTC logic.",
        "- The common metric function uses JiWER 4.0.0 without text transformations.",
        "",
        "## Limits",
        "",
        "- This confirms exact parity with the Phase 4B CPU run for this checkpoint and dataset, not family-wide CRNN generalization.",
        "- Historical-versus-CPU discrepancies remain unresolved as documented in `prediction_discrepancy_report.md`.",
        "- No training, adaptive preprocessing, or Transformer/VLM adapter was performed.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("CRNN adapter equivalence validation requires CPU-only PyTorch")
    if not BASELINE_METRICS_PATH.is_file():
        raise FileNotFoundError(f"Phase 4B metrics are missing: {BASELINE_METRICS_PATH}")

    baseline_rows = _read_prediction_rows(BASELINE_PREDICTIONS_PATH)
    baseline_metrics = json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))
    image_root, metadata_path, dataset_config = load_configured_paths()
    all_metadata_rows, feasible_rows, sample_counts = preflight_dataset(
        image_root,
        metadata_path,
        dataset_config,
    )
    if sample_counts["feasible_test"] != EXPECTED_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLES} feasible test samples, "
            f"found {sample_counts['feasible_test']}"
        )

    rows_by_name = {row["filename"]: row for row in feasible_rows}
    if set(rows_by_name) != {row["filename"] for row in baseline_rows}:
        raise ValueError("Phase 4B prediction coverage differs from feasible test samples")
    examples: list[OCRExample] = []
    for baseline_row in baseline_rows:
        row = rows_by_name[baseline_row["filename"]]
        if (
            baseline_row["ground_truth"] != row["transcription"]
            or baseline_row["language"] != row["language"]
            or baseline_row["source_image_id"] != row["source_image_id"]
        ):
            raise ValueError(
                f"Phase 4B metadata differs from dataset annotation: {row['filename']}"
            )
        examples.append(
            OCRExample(
                image_path=image_root / row["filename"],
                filename=row["filename"],
                ground_truth=row["transcription"],
                language=row["language"],
                source_image_id=row["source_image_id"],
            )
        )

    model_configuration = yaml.safe_load(
        (ROOT / "configs" / "models.yaml").read_text(encoding="utf-8")
    )["models"]["crnn"]
    checkpoint_hash_before = sha256_file(ARCHIVE_PATH)
    adapter = load_model_adapter(
        "crnn",
        config_path=ROOT / "configs" / "models.yaml",
        project_root=ROOT,
    )
    if not isinstance(adapter, CRNNAdapter):
        raise TypeError("Configured CRNN model resolved to an unexpected adapter")
    verify_vocabulary(adapter.vocabulary, all_metadata_rows, feasible_rows)
    if model_configuration["batch_size"] != 64:
        raise ValueError("Configured CRNN batch size differs from Phase 4B")
    torch.set_num_threads(2)

    start = time.perf_counter()
    adapter_rows = run_inference(adapter, examples, batch_size=64)
    inference_seconds = time.perf_counter() - start

    same_order = [row["filename"] for row in adapter_rows] == [
        row["filename"] for row in baseline_rows
    ]
    same_metadata = all(
        adapter_row[key] == baseline_row[key]
        for adapter_row, baseline_row in zip(adapter_rows, baseline_rows, strict=True)
        for key in ("filename", "source_image_id", "language", "ground_truth")
    )
    prediction_matches = sum(
        adapter_row["prediction"] == baseline_row["prediction"]
        for adapter_row, baseline_row in zip(adapter_rows, baseline_rows, strict=True)
    )
    adapter_metrics = calculate_cer_wer(
        [row["ground_truth"] for row in adapter_rows],
        [row["prediction"] for row in adapter_rows],
    )
    test_feasibility_crosscheck = sum(
        calculate_feasibility(row)[2]
        for row in all_metadata_rows
        if row["split"].strip().lower() == "test"
    ) == EXPECTED_SAMPLES
    baseline_cer = float(baseline_metrics["cer"])
    baseline_wer = float(baseline_metrics["wer"])
    cer_matches = adapter_metrics["cer"] == baseline_cer
    wer_matches = adapter_metrics["wer"] == baseline_wer
    checkpoint_hash_after = sha256_file(ARCHIVE_PATH)
    unchanged_archive = checkpoint_hash_before == checkpoint_hash_after
    status = (
        "passed"
        if same_order
        and same_metadata
        and prediction_matches == EXPECTED_SAMPLES
        and cer_matches
        and wer_matches
        and test_feasibility_crosscheck
        and unchanged_archive
        else "failed"
    )
    report: dict[str, Any] = {
        "status": status,
        "command": (
            r".\.venv\Scripts\python.exe "
            r".\experiments\ocr_evaluation\recovery\run_crnn_adapter_validation.py"
        ),
        "architecture": adapter.configuration["architecture"],
        "checkpoint_sha256": adapter.checkpoint_sha256,
        "vocabulary_sha256": adapter.vocabulary_sha256,
        "device": str(next(adapter.model.parameters()).device),
        "model_training_mode": adapter.model.training,
        "parameter_count": sum(
            parameter.numel() for parameter in adapter.model.parameters()
        ),
        "vocabulary_entries": len(adapter.vocabulary),
        "samples_evaluated": len(adapter_rows),
        "infeasible_test_samples_excluded": sample_counts["infeasible_test"],
        "inference_seconds": inference_seconds,
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "checkpoint_archive_sha256_before": checkpoint_hash_before,
        "checkpoint_archive_sha256_after": checkpoint_hash_after,
        "checkpoint_archive_unchanged": unchanged_archive,
        "equivalence": {
            "same_sample_order": same_order,
            "same_metadata": same_metadata,
            "prediction_matches": prediction_matches,
            "prediction_count": len(adapter_rows),
            "adapter_cer": adapter_metrics["cer"],
            "baseline_cer": baseline_cer,
            "adapter_wer": adapter_metrics["wer"],
            "baseline_wer": baseline_wer,
            "cer_matches": cer_matches,
            "wer_matches": wer_matches,
            "exact_match_count": adapter_metrics["exact_match_count"],
            "exact_match_accuracy": adapter_metrics["exact_match_accuracy"],
            "jiwer_version": version("jiwer"),
        },
        "files": {
            "model_config": str(ROOT / "configs" / "models.yaml"),
            "dataset_metadata": str(metadata_path),
            "baseline_predictions": str(BASELINE_PREDICTIONS_PATH),
            "baseline_metrics": str(BASELINE_METRICS_PATH),
            "report_markdown": str(REPORT_MARKDOWN_PATH),
        },
        "ctc_feasibility_rule": (
            "actual_steps=(round(crop_width*32/crop_height)//4)-1; "
            "required_steps=len(text)+adjacent_repeated_characters"
        ),
        "metric_validation": {
            "references": len(adapter_rows),
            "calculation": "jiwer.cer and jiwer.wer with no explicit transformations",
            "test_split_feasibility_crosscheck": test_feasibility_crosscheck,
        },
    }
    REPORT_JSON_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    REPORT_MARKDOWN_PATH.write_text(_build_markdown(report), encoding="utf-8")
    print(f"Status: {status}")
    print(f"Samples: {len(adapter_rows)}")
    print(f"Exact prediction matches: {prediction_matches}/{len(adapter_rows)}")
    print(f"CER: {adapter_metrics['cer']:.12f} (baseline {baseline_cer:.12f})")
    print(f"WER: {adapter_metrics['wer']:.12f} (baseline {baseline_wer:.12f})")
    print(f"Report: {REPORT_MARKDOWN_PATH}")
    if status != "passed":
        raise RuntimeError("CRNN adapter did not exactly reproduce the Phase 4B CPU run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
