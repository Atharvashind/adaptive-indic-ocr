"""Tests for the recovered CRNN adapter and generic evaluation interface."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from PIL import Image

from src.evaluation.runner import OCRExample, run_inference
from src.models import CRNNAdapter, load_model_adapter
from src.models.base_adapter import BaseOCRAdapter

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "models.yaml"
BASELINE_PREDICTIONS = (
    ROOT
    / "experiments"
    / "ocr_evaluation"
    / "baseline_reproduction"
    / "run-20261106T091015.067700Z"
    / "predictions.csv"
)


@pytest.fixture(scope="module")
def adapter() -> CRNNAdapter:
    model_adapter = load_model_adapter("crnn", CONFIG_PATH, ROOT)
    assert isinstance(model_adapter, CRNNAdapter)
    return model_adapter


def _read_model_configuration() -> dict[str, Any]:
    loaded = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return loaded["models"]["crnn"]


def test_strict_checkpoint_loading_and_vocabulary_validation(adapter: CRNNAdapter) -> None:
    assert adapter.checkpoint_sha256 == (
        "9a63c6ce29b37fdbc1b9181e5fb494d501351ebc916a1ed3189f0819c63fea10"
    )
    assert adapter.vocabulary_sha256 == (
        "5220a159757d92bccb69e447dd1f2e3ecb30d6bdcca55538c7fe429f68822d5f"
    )
    assert len(adapter.vocabulary) == 280
    assert adapter.vocabulary["<BLANK>"] == 0
    assert sum(parameter.numel() for parameter in adapter.model.parameters()) == 6_095_128
    assert not adapter.model.training
    assert {parameter.device.type for parameter in adapter.model.parameters()} == {"cpu"}


def test_vocabulary_validation_rejects_duplicate_or_noncontiguous_indices() -> None:
    with pytest.raises(ValueError, match="unique and contiguous"):
        CRNNAdapter._validate_vocabulary({"<BLANK>": 0, "a": 2})


def test_preprocessing_dimensions_padding_and_ctc_lengths(
    adapter: CRNNAdapter,
    tmp_path: Path,
) -> None:
    short_path = tmp_path / "short.png"
    long_path = tmp_path / "long.png"
    Image.new("RGB", (10, 8), color=(128, 128, 128)).save(short_path)
    Image.new("RGB", (20, 8), color=(128, 128, 128)).save(long_path)

    short_tensor, short_width = adapter.preprocess_image(short_path)
    assert short_tensor.shape == (1, 32, 40)
    assert short_width == 40

    prepared = adapter.prepare_batch([short_path, long_path])
    assert prepared.images.shape == (2, 1, 32, 80)
    assert prepared.padded_width == 80
    assert prepared.content_widths == (80, 40)
    assert prepared.input_lengths == (19, 9)
    assert prepared.original_indices == (1, 0)
    assert torch.count_nonzero(prepared.images[1, :, :, 40:]).item() == 0


def test_greedy_ctc_decoding_collapses_repeats_and_resets_on_blank(
    adapter: CRNNAdapter,
) -> None:
    indices = [0, 1, 1, 0, 1, 2, 2, 0]
    logits = torch.full((len(indices), 1, 280), -1.0)
    for timestep, index in enumerate(indices):
        logits[timestep, 0, index] = 1.0
    expected = adapter.index_to_character[1] * 2 + adapter.index_to_character[2]
    assert adapter.decode(logits) == [expected]


def test_adapter_implements_common_prediction_interface(adapter: CRNNAdapter) -> None:
    assert isinstance(adapter, BaseOCRAdapter)
    assert adapter.predict([]) == []


def test_missing_checkpoint_and_vocabulary_fail_closed(tmp_path: Path) -> None:
    checkpoint_config = _read_model_configuration()
    checkpoint_config["checkpoint_path"] = str(tmp_path / "missing.zip")
    with pytest.raises(FileNotFoundError, match="artifact does not exist"):
        CRNNAdapter(checkpoint_config, ROOT)

    vocabulary_config = _read_model_configuration()
    vocabulary_config["vocabulary_path"] = str(tmp_path / "missing-vocabulary.zip")
    with pytest.raises(FileNotFoundError, match="artifact does not exist"):
        CRNNAdapter(vocabulary_config, ROOT)


def test_predictions_match_phase_4b_baseline_batch(adapter: CRNNAdapter) -> None:
    assert BASELINE_PREDICTIONS.is_file()
    with BASELINE_PREDICTIONS.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))[:64]
    examples = [
        OCRExample(
            image_path=ROOT / "synth6k" / "images" / row["filename"],
            filename=row["filename"],
            source_image_id=row["source_image_id"],
            language=row["language"],
            ground_truth=row["ground_truth"],
        )
        for row in rows
    ]

    actual = run_inference(adapter, examples, batch_size=64)

    assert [row["filename"] for row in actual] == [row["filename"] for row in rows]
    assert [row["ground_truth"] for row in actual] == [
        row["ground_truth"] for row in rows
    ]
    assert [row["language"] for row in actual] == [row["language"] for row in rows]
    assert [row["prediction"] for row in actual] == [
        row["prediction"] for row in rows
    ]
