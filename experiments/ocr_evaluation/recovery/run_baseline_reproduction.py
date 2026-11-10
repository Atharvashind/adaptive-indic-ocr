from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageEnhance
from torch.nn import functional as torch_functional
from torch.utils.data import DataLoader, Dataset

from run_checkpoint_load_test import CRNN


ROOT = Path(__file__).resolve().parents[3]
RECOVERY_DIR = ROOT / "experiments" / "ocr_evaluation" / "recovery"
ARCHIVE_PATH = (
    ROOT
    / "experiments"
    / "recovered_baseline"
    / "Multilingual_CRNN_OCR-20261105T161607Z-1-001.zip"
)
CONFIG_PATH = ROOT / "configs" / "datasets.yaml"
CHECKPOINT_AUDIT_PATH = RECOVERY_DIR / "checkpoint_load_test.json"
NOTEBOOK_PATH = ROOT / "notebooks" / "crnn_tes (2).ipynb"
CHECKPOINT_MEMBER = "Multilingual_CRNN_OCR/Model_Weights/best_model_epoch4.pt"
VOCABULARY_MEMBER = "Multilingual_CRNN_OCR/Model_Weights/vocabulary.json"
HISTORICAL_REPORT_MEMBER = "Multilingual_CRNN_OCR/Evaluation_Results/test_evaluation_report.json"
HISTORICAL_PREDICTIONS_MEMBER = (
    "Multilingual_CRNN_OCR/Evaluation_Results/test_predictions_log.csv"
)
EXPECTED_FEASIBLE_TEST_COUNT = 8094
EXPECTED_TOTAL_TEST_COUNT = 8157
BATCH_SIZE = 64
INPUT_HEIGHT = 32
BLANK_INDEX = 0
OUTPUT_CLASSES = 280
RUN_COMMAND = (
    r".\.venv\Scripts\python.exe "
    r".\experiments\ocr_evaluation\recovery\run_baseline_reproduction.py"
)
JIWER_INSTALL_COMMAND = (
    r".\.venv\Scripts\python.exe -m pip install jiwer==4.0.0 --only-binary=:all:"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_csv(
    path: Path,
    encoding: str = "utf-8-sig",
    delimiter: str = ",",
) -> list[dict[str, str]]:
    with path.open("r", encoding=encoding, newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def parse_text_manifest(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue
            fields = line.split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(f"Malformed annotation at {path}:{line_number}")
            filename, transcription = fields
            if filename in records:
                raise ValueError(f"Duplicate annotation filename at {path}:{line_number}")
            records[filename] = transcription
    return records


def calculate_feasibility(row: dict[str, str]) -> tuple[int, int, bool]:
    transcription = row["transcription"]
    crop_width = float(row["crop_width"])
    crop_height = float(row["crop_height"])
    if not math.isfinite(crop_width) or not math.isfinite(crop_height):
        raise ValueError(f"Invalid crop dimensions for {row['filename']}")
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError(f"Non-positive crop dimensions for {row['filename']}")
    resized_width = int(round(crop_width * INPUT_HEIGHT / crop_height))
    repeated_adjacent = sum(
        left == right for left, right in zip(transcription, transcription[1:])
    )
    required_steps = len(transcription) + repeated_adjacent
    actual_steps = (resized_width // 4) - 1
    return resized_width, required_steps, actual_steps >= required_steps


def load_configured_paths() -> tuple[Path, Path, dict[str, Any]]:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Dataset configuration missing: {CONFIG_PATH}")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    dataset_config = config.get("datasets", {}).get("synth6k")
    require(isinstance(dataset_config, dict), "synth6k configuration is missing or invalid")
    require(dataset_config.get("enabled") is True, "synth6k is not enabled in datasets.yaml")
    annotation = dataset_config.get("annotation", {})
    require(annotation.get("format") == "csv", "synth6k annotation format is not verified CSV")
    image_root_value = dataset_config.get("image_root")
    metadata_value = annotation.get("path")
    require(bool(image_root_value), "synth6k image_root is not configured")
    require(bool(metadata_value), "synth6k annotation path is not configured")
    image_root = (ROOT / image_root_value).resolve()
    metadata_path = (ROOT / metadata_value).resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"synth6k image directory missing: {image_root}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"synth6k metadata missing: {metadata_path}")
    return image_root, metadata_path, dataset_config


def preflight_dataset(
    image_root: Path,
    metadata_path: Path,
    dataset_config: dict[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, int]]:
    annotation_config = dataset_config["annotation"]
    annotation_columns = annotation_config.get("columns", {})
    expected_mapping = {
        "image_path": "filename",
        "source_image_id": "source_image_id",
        "transcription": "transcription",
        "language": "language",
        "split": "split",
    }
    if any(annotation_columns.get(key) != value for key, value in expected_mapping.items()):
        raise ValueError("synth6k annotation column mapping differs from the verified format")
    rows = read_csv(
        metadata_path,
        encoding=str(annotation_config.get("encoding", "utf-8-sig")),
        delimiter=str(annotation_config.get("delimiter", ",")),
    )
    required_columns = {
        "filename",
        "source_image_id",
        "language",
        "transcription",
        "split",
        "crop_width",
        "crop_height",
    }
    columns = set(rows[0]) if rows else set()
    missing_columns = required_columns - columns
    if missing_columns:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing_columns)}")

    test_rows = [row for row in rows if row["split"].strip().lower() == "test"]
    if len(test_rows) != EXPECTED_TOTAL_TEST_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_TEST_COUNT} test annotations, found {len(test_rows)}"
        )
    filenames = [row["filename"] for row in test_rows]
    if len(set(filenames)) != len(filenames):
        raise ValueError("Duplicate filenames exist in the test annotations")

    reference_files = annotation_config.get("reference_files", [])
    test_reference = next(
        (value for value in reference_files if Path(value).stem.lower() == "test"),
        None,
    )
    if not test_reference:
        raise ValueError("The configured synth6k test annotation reference is missing")
    manifest_path = (ROOT / test_reference).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Test annotation reference missing: {manifest_path}")
    manifest_records = parse_text_manifest(manifest_path)
    metadata_records = {row["filename"]: row["transcription"] for row in test_rows}
    if manifest_records != metadata_records:
        raise ValueError(
            "The configured test annotation text file and metadata CSV disagree"
        )

    feasible_rows: list[dict[str, str]] = []
    counts = {"total_test": len(test_rows), "feasible_test": 0, "infeasible_test": 0}
    for row in test_rows:
        for field in ("filename", "source_image_id", "language", "transcription"):
            if not row[field]:
                raise ValueError(f"Missing {field} in test annotation for {row.get('filename')}")
        filename_path = Path(row["filename"])
        if filename_path.is_absolute() or len(filename_path.parts) != 1:
            raise ValueError(f"Unexpected non-filename path in metadata: {row['filename']}")
        image_path = image_root / row["filename"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Test image missing: {image_path}")
        try:
            with Image.open(image_path) as image:
                image_size = image.size
                image.load()
        except Exception as error:
            raise OSError(f"Test image is unreadable: {image_path}: {error}") from error
        annotated_size = (int(float(row["crop_width"])), int(float(row["crop_height"])))
        if image_size != annotated_size:
            raise ValueError(
                f"Annotation/image dimensions differ for {row['filename']}: "
                f"{annotated_size} != {image_size}"
            )
        _, _, feasible = calculate_feasibility(row)
        if feasible:
            feasible_rows.append(row)
            counts["feasible_test"] += 1
        else:
            counts["infeasible_test"] += 1

    if counts["feasible_test"] != EXPECTED_FEASIBLE_TEST_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEASIBLE_TEST_COUNT} feasible rows, "
            f"found {counts['feasible_test']}"
        )
    if counts["feasible_test"] + counts["infeasible_test"] != counts["total_test"]:
        raise RuntimeError("Test annotation accounting is incomplete")
    return rows, feasible_rows, counts


class HistoricalCRNNDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, str]],
        image_root: Path,
        char_to_index: dict[str, int],
    ) -> None:
        self.rows = rows
        self.image_root = image_root
        self.char_to_index = char_to_index

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = self.image_root / row["filename"]
        try:
            with Image.open(image_path) as source:
                grayscale = source.convert("L")
        except Exception as error:
            raise OSError(f"Unable to read test image {image_path}: {error}") from error

        original_width, original_height = grayscale.size
        scale_factor = INPUT_HEIGHT / original_height
        content_width = int(round(original_width * scale_factor))
        if content_width < 1:
            raise ValueError(f"Resized width is invalid for {row['filename']}")
        resized = grayscale.resize(
            (content_width, INPUT_HEIGHT),
            Image.Resampling.BILINEAR,
        )
        enhanced = ImageEnhance.Contrast(resized).enhance(2.0)
        image_tensor = torch.tensor(
            np.array(enhanced),
            dtype=torch.float32,
        ) / 255.0
        image_tensor = (image_tensor - 0.5) / 0.5
        image_tensor = image_tensor.unsqueeze(0)

        transcription = row["transcription"]
        missing_characters = [character for character in transcription if character not in self.char_to_index]
        if missing_characters:
            raise ValueError(
                f"Vocabulary has no entry for text in {row['filename']}: "
                f"{missing_characters!r}"
            )
        target = torch.tensor(
            [self.char_to_index[character] for character in transcription],
            dtype=torch.long,
        )
        return {
            "img": image_tensor,
            "target": target,
            "text": transcription,
            "filename": row["filename"],
            "source_image_id": row["source_image_id"],
            "language": row["language"],
            "new_w": content_width,
        }


def historical_collate(batch: list[dict[str, Any]]) -> tuple[Any, ...]:
    batch = sorted(batch, key=lambda item: item["new_w"], reverse=True)
    max_batch_width = batch[0]["new_w"]
    if max_batch_width % 4:
        max_batch_width = ((max_batch_width // 4) + 1) * 4

    padded_images = []
    targets = []
    target_lengths = []
    input_lengths = []
    raw_texts = []
    filenames = []
    source_image_ids = []
    languages = []
    for item in batch:
        image = item["img"]
        width = image.size(2)
        padded_images.append(
            torch_functional.pad(
                image,
                (0, max_batch_width - width),
                value=0.0,
            )
        )
        targets.append(item["target"])
        target_lengths.append(len(item["target"]))
        input_lengths.append((width // 4) - 1)
        raw_texts.append(item["text"])
        filenames.append(item["filename"])
        source_image_ids.append(item["source_image_id"])
        languages.append(item["language"])

    return (
        torch.stack(padded_images),
        torch.cat(targets),
        torch.tensor(target_lengths, dtype=torch.long),
        torch.tensor(input_lengths, dtype=torch.long),
        raw_texts,
        filenames,
        source_image_ids,
        languages,
        max_batch_width,
    )


def historical_greedy_decode(
    predictions: torch.Tensor,
    index_to_character: dict[int, str],
) -> list[str]:
    predicted_indices = predictions.permute(1, 0, 2).argmax(2).cpu().numpy()
    decoded_texts = []
    for sequence in predicted_indices:
        text = ""
        previous = -1
        for class_index in sequence:
            if class_index != BLANK_INDEX and class_index != previous:
                try:
                    text += index_to_character[int(class_index)]
                except KeyError as error:
                    raise ValueError(
                        f"Model emitted class absent from vocabulary: {class_index}"
                    ) from error
            previous = class_index
        decoded_texts.append(text)
    return decoded_texts


def load_historical_resources(
    archive_path: Path,
    checkpoint_audit: dict[str, Any],
) -> tuple[dict[str, int], dict[str, Any], list[dict[str, str]], str, str]:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Recovered checkpoint archive missing: {archive_path}")
    archive_hash_before = sha256_file(archive_path)
    expected_archive_hash = checkpoint_audit["archive"]["sha256_before"]
    if archive_hash_before != expected_archive_hash:
        raise ValueError("Recovered archive hash differs from the Phase 4A audit")

    expected_checkpoint = next(
        item
        for item in checkpoint_audit["checkpoints"]
        if item["name"] == "best_model_epoch4.pt"
    )
    with zipfile.ZipFile(archive_path, "r") as archive:
        vocabulary_bytes = archive.read(VOCABULARY_MEMBER)
        vocabulary = json.loads(vocabulary_bytes.decode("utf-8"))
        historical_report = json.loads(
            archive.read(HISTORICAL_REPORT_MEMBER).decode("utf-8")
        )
        with tempfile.TemporaryDirectory(prefix="crnn_baseline_reproduction_") as temp_dir:
            checkpoint_path = Path(temp_dir) / "best_model_epoch4.pt"
            checkpoint_path.write_bytes(archive.read(CHECKPOINT_MEMBER))
            checkpoint_hash = sha256_file(checkpoint_path)
            if checkpoint_hash != expected_checkpoint["sha256_before"]:
                raise ValueError("Best checkpoint hash differs from the Phase 4A audit")
            model = CRNN(num_classes=OUTPUT_CLASSES).to("cpu")
            state_dict = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(state_dict, dict):
                raise TypeError("Expected the archived checkpoint to be a raw state dictionary")
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            if parameter_count != expected_checkpoint["parameter_count"]:
                raise ValueError(
                    f"Model parameter count {parameter_count} differs from Phase 4A "
                    f"({expected_checkpoint['parameter_count']})"
                )
            require(not model.training, "Model failed to enter evaluation mode")
            del state_dict
            if sha256_file(checkpoint_path) != checkpoint_hash:
                raise RuntimeError("Checkpoint bytes changed during deserialization")

    archive_hash_after = sha256_file(archive_path)
    if archive_hash_after != archive_hash_before:
        raise RuntimeError("Recovered archive changed while loading model resources")
    with zipfile.ZipFile(archive_path, "r") as archive:
        historical_predictions = list(
            csv.DictReader(
                io.StringIO(
                    archive.read(HISTORICAL_PREDICTIONS_MEMBER).decode("utf-8-sig")
                )
            )
        )
    return (
        vocabulary,
        model,
        historical_predictions,
        archive_hash_before,
        checkpoint_hash,
    )


def verify_vocabulary(
    vocabulary: Any,
    all_rows: list[dict[str, str]],
    feasible_rows: list[dict[str, str]],
) -> tuple[dict[str, int], dict[int, str]]:
    if not isinstance(vocabulary, dict):
        raise TypeError("Archived vocabulary must be a character-to-index JSON object")
    if any(not isinstance(character, str) or not isinstance(index, int) for character, index in vocabulary.items()):
        raise TypeError("Vocabulary keys must be strings and values must be integer indices")
    train_characters = sorted(
        {
            character
            for row in all_rows
            if row["split"].strip().lower() == "train"
            for character in row["transcription"]
        }
    )
    expected = {character: index + 1 for index, character in enumerate(train_characters)}
    expected["<BLANK>"] = BLANK_INDEX
    if vocabulary != expected:
        raise ValueError("Archived vocabulary does not match sorted training-split characters")
    if len(vocabulary) != OUTPUT_CLASSES or vocabulary.get("<BLANK>") != BLANK_INDEX:
        raise ValueError("Vocabulary output dimension or CTC blank index is inconsistent")
    out_of_vocabulary = {
        character
        for row in feasible_rows
        for character in row["transcription"]
        if character not in vocabulary
    }
    if out_of_vocabulary:
        raise ValueError(
            f"Test text contains out-of-vocabulary characters: {sorted(out_of_vocabulary)!r}"
        )
    return vocabulary, {index: character for character, index in vocabulary.items()}


def evaluate(
    model: CRNN,
    dataset: HistoricalCRNNDataset,
    index_to_character: dict[int, str],
) -> tuple[list[dict[str, str]], float, int]:
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=historical_collate,
    )
    predictions_rows: list[dict[str, str]] = []
    inference_start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            (
                images,
                _targets,
                _target_lengths,
                _input_lengths,
                raw_texts,
                filenames,
                source_image_ids,
                languages,
                padded_width,
            ) = batch
            if not bool(torch.isfinite(images).all().item()):
                raise ValueError("Preprocessed model input contains non-finite values")
            predictions = model(images.to("cpu"))
            if predictions.ndim != 3:
                raise ValueError(f"Invalid model output rank: {predictions.ndim}")
            expected_time = (padded_width // 4) - 1
            if tuple(predictions.shape) != (expected_time, len(filenames), OUTPUT_CLASSES):
                raise ValueError(
                    f"Unexpected model output shape {tuple(predictions.shape)}; expected "
                    f"{(expected_time, len(filenames), OUTPUT_CLASSES)}"
                )
            if not bool(torch.isfinite(predictions).all().item()):
                raise ValueError("Model output contains NaN or infinite values")
            decoded = historical_greedy_decode(predictions, index_to_character)
            if len(decoded) != len(raw_texts):
                raise RuntimeError("Decoded prediction batch count differs from annotation count")
            for filename, source_id, language, truth, prediction in zip(
                filenames,
                source_image_ids,
                languages,
                raw_texts,
                decoded,
                strict=True,
            ):
                predictions_rows.append(
                    {
                        "filename": filename,
                        "source_image_id": source_id,
                        "language": language,
                        "ground_truth": truth,
                        "prediction": prediction,
                    }
                )
    inference_seconds = time.perf_counter() - inference_start
    if len(predictions_rows) != len(dataset):
        raise RuntimeError(
            f"Evaluated {len(predictions_rows)} rows for {len(dataset)} expected samples"
        )
    evaluated_names = [row["filename"] for row in predictions_rows]
    expected_names = [row["filename"] for row in dataset.rows]
    if len(set(evaluated_names)) != len(evaluated_names):
        raise RuntimeError("Duplicate filenames in generated predictions")
    if set(evaluated_names) != set(expected_names):
        raise RuntimeError("Prediction filenames do not exactly cover feasible test annotations")
    return predictions_rows, inference_seconds, len(predictions_rows)


def calculate_metrics(
    predictions_rows: list[dict[str, str]],
    historical_report: dict[str, Any],
    historical_predictions: list[dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    references = [row["ground_truth"] for row in predictions_rows]
    hypotheses = [row["prediction"] for row in predictions_rows]
    overall_cer = float(jiwer.cer(references, hypotheses))
    overall_wer = float(jiwer.wer(references, hypotheses))
    exact_matches = sum(reference == hypothesis for reference, hypothesis in zip(references, hypotheses))

    per_language: list[dict[str, Any]] = []
    for language in sorted({row["language"] for row in predictions_rows}):
        subset = [row for row in predictions_rows if row["language"] == language]
        language_refs = [row["ground_truth"] for row in subset]
        language_hypotheses = [row["prediction"] for row in subset]
        per_language.append(
            {
                "language": language,
                "sample_count": len(subset),
                "cer": float(jiwer.cer(language_refs, language_hypotheses)),
                "wer": float(jiwer.wer(language_refs, language_hypotheses)),
                "exact_match_count": sum(
                    reference == hypothesis
                    for reference, hypothesis in zip(language_refs, language_hypotheses)
                ),
                "exact_match_accuracy": sum(
                    reference == hypothesis
                    for reference, hypothesis in zip(language_refs, language_hypotheses)
                )
                / len(subset),
            }
        )

    historical_by_filename = {row["filename"]: row for row in historical_predictions}
    if len(historical_by_filename) != len(historical_predictions):
        raise ValueError("Historical prediction log contains duplicate filenames")
    generated_by_filename = {row["filename"]: row for row in predictions_rows}
    if set(historical_by_filename) != set(generated_by_filename):
        raise ValueError("Historical predictions do not cover the same feasible test filenames")
    historical_agreement = sum(
        historical_by_filename[name]["prediction"] == generated_by_filename[name]["prediction"]
        for name in generated_by_filename
    )
    historical_cer = float(historical_report["overall_test_cer"])
    historical_wer = float(historical_report["overall_test_wer"])
    metrics = {
        "evaluated_samples": len(predictions_rows),
        "skipped_infeasible_samples": EXPECTED_TOTAL_TEST_COUNT - len(predictions_rows),
        "total_test_samples": EXPECTED_TOTAL_TEST_COUNT,
        "cer": overall_cer,
        "wer": overall_wer,
        "exact_match_count": exact_matches,
        "exact_match_accuracy": exact_matches / len(predictions_rows),
        "historical_reference": {
            "evaluated_samples": sum(
                int(item["count"]) for item in historical_report["language_breakdown"].values()
            ),
            "cer": historical_cer,
            "wer": historical_wer,
        },
        "absolute_difference": {
            "cer": abs(overall_cer - historical_cer),
            "wer": abs(overall_wer - historical_wer),
        },
        "historical_prediction_agreement": {
            "matching_predictions": historical_agreement,
            "total_predictions": len(predictions_rows),
            "agreement_fraction": historical_agreement / len(predictions_rows),
        },
        "jiwer": {
            "version": version("jiwer"),
            "api": "jiwer.cer(references, hypotheses) and jiwer.wer(references, hypotheses)",
            "explicit_transformations": [],
            "historical_version_recorded": None,
        },
    }
    if metrics["historical_reference"]["evaluated_samples"] != len(predictions_rows):
        raise ValueError("Historical report's language counts differ from evaluated sample count")
    return metrics, per_language, historical_report


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(
    metrics: dict[str, Any],
    per_language: list[dict[str, Any]],
    run_metadata: dict[str, Any],
) -> str:
    lines = [
        "# Historical CRNN baseline reproduction",
        "",
        f"- **Run ID:** `{run_metadata['run_id']}`",
        f"- **Checkpoint:** `{run_metadata['checkpoint']['name']}`",
        f"- **Checkpoint SHA-256:** `{run_metadata['checkpoint']['sha256']}`",
        f"- **Device:** `{run_metadata['inference']['device']}`",
        f"- **PyTorch / JiWER:** `{run_metadata['environment']['torch_version']}` / `{metrics['jiwer']['version']}`",
        "",
        "## Overall results",
        "",
        "| Metric | Reproduced | Historical | Absolute difference |",
        "|---|---:|---:|---:|",
        f"| Samples | {metrics['evaluated_samples']} | {metrics['historical_reference']['evaluated_samples']} | 0 |",
        f"| CER | {metrics['cer']:.12f} | {metrics['historical_reference']['cer']:.12f} | {metrics['absolute_difference']['cer']:.12f} |",
        f"| WER | {metrics['wer']:.12f} | {metrics['historical_reference']['wer']:.12f} | {metrics['absolute_difference']['wer']:.12f} |",
        f"| Exact-match accuracy | {metrics['exact_match_accuracy']:.12f} | Not recorded | Not applicable |",
        "",
        f"- Feasible test samples evaluated: {metrics['evaluated_samples']}",
        f"- CTC-infeasible test samples excluded: {metrics['skipped_infeasible_samples']}",
        f"- Inference time: {run_metadata['inference']['total_seconds']:.6f} seconds "
        f"({run_metadata['inference']['milliseconds_per_sample']:.6f} ms/sample)",
        f"- Historical prediction agreement: "
        f"{metrics['historical_prediction_agreement']['matching_predictions']}/"
        f"{metrics['historical_prediction_agreement']['total_predictions']} "
        f"({metrics['historical_prediction_agreement']['agreement_fraction']:.6%})",
        "",
        "## Per-language results",
        "",
        "| Language | Samples | CER | WER | Exact-match accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in per_language:
        lines.append(
            f"| {row['language']} | {row['sample_count']} | {row['cer']:.12f} | "
            f"{row['wer']:.12f} | {row['exact_match_accuracy']:.6%} |"
        )
    lines.extend(
        [
            "",
            "## Reproduction notes",
            "",
            "- Preprocessing, batch padding, feasibility, and greedy decoding follow the original notebook's primary implementation (cells 57, 58/66, and 94).",
            "- CER/WER use the notebook's `jiwer.cer` / `jiwer.wer` calls with no explicit transformations. The historical JiWER version was not archived; this run pins the installed version in `run_metadata.json`.",
            "- Historical metrics are comparison targets loaded from the recovered archive, not used to compute the reproduced metrics.",
            "- No model training, adaptive preprocessing, test-time transformations, or OCR inference outside this baseline evaluation was performed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("Baseline reproduction is restricted to a CPU-only PyTorch runtime")
    if not CHECKPOINT_AUDIT_PATH.is_file():
        raise FileNotFoundError(f"Phase 4A checkpoint audit missing: {CHECKPOINT_AUDIT_PATH}")
    checkpoint_audit = json.loads(CHECKPOINT_AUDIT_PATH.read_text(encoding="utf-8"))
    if checkpoint_audit.get("status") != "passed":
        raise RuntimeError("Phase 4A checkpoint-load test did not pass")
    image_root, metadata_path, dataset_config = load_configured_paths()
    _all_test_rows, feasible_rows, sample_counts = preflight_dataset(
        image_root,
        metadata_path,
        dataset_config,
    )
    vocabulary, model, historical_predictions, archive_hash, checkpoint_hash = (
        load_historical_resources(ARCHIVE_PATH, checkpoint_audit)
    )
    vocabulary, index_to_character = verify_vocabulary(
        vocabulary,
        read_csv(metadata_path),
        feasible_rows,
    )
    require(model.training is False, "Loaded CRNN is not in evaluation mode")

    with zipfile.ZipFile(ARCHIVE_PATH, "r") as archive:
        historical_report = json.loads(
            archive.read(HISTORICAL_REPORT_MEMBER).decode("utf-8")
        )
        vocabulary_hash = sha256_bytes(archive.read(VOCABULARY_MEMBER))
    test_reference = next(
        value
        for value in dataset_config["annotation"]["reference_files"]
        if Path(value).stem.lower() == "test"
    )
    test_annotation_path = (ROOT / test_reference).resolve()

    dataset = HistoricalCRNNDataset(feasible_rows, image_root, vocabulary)
    predictions_rows, inference_seconds, evaluated_count = evaluate(
        model,
        dataset,
        index_to_character,
    )
    metrics, per_language, _ = calculate_metrics(
        predictions_rows,
        historical_report,
        historical_predictions,
    )
    if evaluated_count != sample_counts["feasible_test"]:
        raise RuntimeError("Inference count differs from feasibility preflight")

    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S.%fZ")
    output_dir = ROOT / "experiments" / "ocr_evaluation" / "baseline_reproduction" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    archive_hash_after = sha256_file(ARCHIVE_PATH)
    if archive_hash_after != archive_hash:
        raise RuntimeError("Recovered archive changed during baseline reproduction")

    run_metadata: dict[str, Any] = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_command": RUN_COMMAND,
        "jiwer_install_command": JIWER_INSTALL_COMMAND,
        "working_directory": str(ROOT),
        "dataset": {
            "name": "synth6k",
            "image_root": str(image_root),
            "metadata_path": str(metadata_path),
            "test_annotation_reference": str(test_annotation_path),
            **sample_counts,
            "evaluated_filenames_unique": True,
            "all_test_images_readable": True,
            "all_test_image_dimensions_match_annotations": True,
        },
        "inputs": {
            "dataset_config": {
                "path": str(CONFIG_PATH),
                "sha256": sha256_file(CONFIG_PATH),
            },
            "metadata_csv": {
                "path": str(metadata_path),
                "sha256": sha256_file(metadata_path),
            },
            "test_annotation_reference": {
                "path": str(test_annotation_path),
                "sha256": sha256_file(test_annotation_path),
            },
            "notebook": {
                "path": str(NOTEBOOK_PATH),
                "sha256": sha256_file(NOTEBOOK_PATH),
            },
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "archive": {
                "path": str(ARCHIVE_PATH),
                "sha256_before": archive_hash,
                "sha256_after": archive_hash_after,
                "unchanged": archive_hash_after == archive_hash,
            },
            "vocabulary": {
                "archive_member": VOCABULARY_MEMBER,
                "sha256": vocabulary_hash,
                "entries": len(vocabulary),
            },
        },
        "checkpoint": {
            "name": "best_model_epoch4.pt",
            "archive_member": CHECKPOINT_MEMBER,
            "sha256": checkpoint_hash,
            "size_bytes": next(
                item["file_size_bytes"]
                for item in checkpoint_audit["checkpoints"]
                if item["name"] == "best_model_epoch4.pt"
            ),
            "load": "torch.load(map_location='cpu', weights_only=True), load_state_dict(strict=True)",
            "device": str(next(model.parameters()).device),
            "evaluation_mode": not model.training,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "output_classes": OUTPUT_CLASSES,
            "blank_index": BLANK_INDEX,
        },
        "environment": {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "torch_num_threads": torch.get_num_threads(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "pillow_version": version("Pillow"),
            "jiwer_version": version("jiwer"),
            "jiwer_version_in_historical_artifacts": None,
        },
        "preprocessing": {
            "source_notebook_cell": 57,
            "color_mode": "grayscale (PIL L)",
            "height": INPUT_HEIGHT,
            "width": "int(round(original_width * 32.0 / original_height))",
            "aspect_ratio_preserved": True,
            "interpolation": "PIL Image.Resampling.BILINEAR",
            "contrast_enhancement_factor": 2.0,
            "array_dtype": "float32",
            "scaling": "divide by 255.0",
            "normalization": "(x - 0.5) / 0.5",
            "channels": 1,
        },
        "batching": {
            "source_notebook_cell": 57,
            "batch_size": BATCH_SIZE,
            "shuffle": False,
            "sort_within_batch": "descending resized content width",
            "padding": "right padding to max batch width rounded up to multiple of 4",
            "normalized_padding_value": 0.0,
            "ctc_input_length": "(content_width // 4) - 1",
            "decoder_uses_ctc_input_lengths": False,
            "data_loader_num_workers": 0,
        },
        "decoder": {
            "source_notebook_cell": 94,
            "type": "greedy argmax",
            "blank_index": BLANK_INDEX,
            "repeat_handling": "collapse consecutive repeated indices; blank resets previous index",
            "unicode_normalization": None,
            "whitespace_transformations": None,
        },
        "metrics": {
            "source_notebook_cells": [73, 75],
            "jiwer_functions": ["cer", "wer"],
            "explicit_text_transformations": [],
            "exact_match": "raw Python string equality",
        },
        "inference": {
            "device": "cpu",
            "total_seconds": inference_seconds,
            "milliseconds_per_sample": inference_seconds * 1000 / evaluated_count,
            "evaluated_samples": evaluated_count,
            "outputs_checked_for_finite_values": True,
        },
    }
    metrics["runtime"] = {
        "total_inference_seconds": inference_seconds,
        "milliseconds_per_sample": inference_seconds * 1000 / evaluated_count,
        "device": "cpu",
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(
        output_dir / "predictions.csv",
        ["filename", "source_image_id", "language", "ground_truth", "prediction"],
        predictions_rows,
    )
    write_csv(
        output_dir / "per_language_metrics.csv",
        [
            "language",
            "sample_count",
            "cer",
            "wer",
            "exact_match_count",
            "exact_match_accuracy",
        ],
        per_language,
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.md").write_text(
        build_markdown(metrics, per_language, run_metadata),
        encoding="utf-8",
    )

    print(f"Output directory: {output_dir}")
    print(f"Evaluated samples: {metrics['evaluated_samples']}")
    print(f"Infeasible samples excluded: {metrics['skipped_infeasible_samples']}")
    print(f"CER: {metrics['cer']:.12f} (historical {metrics['historical_reference']['cer']:.12f})")
    print(f"WER: {metrics['wer']:.12f} (historical {metrics['historical_reference']['wer']:.12f})")
    print(f"Exact-match accuracy: {metrics['exact_match_accuracy']:.12f}")
    print(
        "Historical prediction agreement: "
        f"{metrics['historical_prediction_agreement']['matching_predictions']}/"
        f"{metrics['historical_prediction_agreement']['total_predictions']}"
    )
    print(f"Inference time: {inference_seconds:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
