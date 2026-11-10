"""Tests for validation sampling and contrast-only CRNN diagnostic behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from experiments.ocr_evaluation.phase5b_ablation.diagnostic_adapter import (
    ContrastAblationCRNNAdapter,
)
from experiments.ocr_evaluation.phase5b_ablation.run_ablation import (
    _sample_validation_rows,
)
from src.models import CRNNAdapter

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model_configuration() -> dict:
    config = yaml.safe_load(
        (ROOT / "configs" / "models.yaml").read_text(encoding="utf-8")
    )
    return config["models"]["crnn"]


def _validation_rows() -> list[dict[str, str]]:
    rows = []
    for language, count in (("bn", 120), ("mr", 80)):
        for board in range(count):
            source_id = f"{language}-{board:03d}"
            for line in range(2):
                rows.append(
                    {
                        "filename": f"{source_id}-{line}.png",
                        "source_image_id": source_id,
                        "language": language,
                        "split": "val",
                    }
                )
    return rows


def test_validation_sampler_is_deterministic_stratified_and_source_unique() -> None:
    rows = _validation_rows()
    first = _sample_validation_rows(rows, sample_size=40, seed=7)
    second = _sample_validation_rows(rows, sample_size=40, seed=7)

    assert [row["filename"] for row in first] == [row["filename"] for row in second]
    assert len(first) == 40
    assert len({row["source_image_id"] for row in first}) == 40
    assert {row["split"] for row in first} == {"val"}
    assert {row["language"] for row in first} == {"bn", "mr"}


def test_validation_sampler_rejects_source_with_multiple_languages() -> None:
    rows = [
        {
            "filename": "line-bn.png",
            "source_image_id": "shared",
            "language": "bn",
            "split": "val",
        },
        {
            "filename": "line-mr.png",
            "source_image_id": "shared",
            "language": "mr",
            "split": "val",
        },
    ]
    with pytest.raises(ValueError, match="multiple languages"):
        _sample_validation_rows(rows, sample_size=1, seed=7)


def test_default_diagnostic_preprocessing_matches_frozen_adapter(
    model_configuration: dict,
    tmp_path: Path,
) -> None:
    path = tmp_path / "contrast-test.png"
    array = np.tile(np.arange(64, dtype=np.uint8), (24, 1))
    Image.fromarray(array).save(path)

    frozen = CRNNAdapter(model_configuration, ROOT)
    diagnostic = ContrastAblationCRNNAdapter(
        model_configuration,
        ROOT,
        native_contrast_enabled=True,
    )

    frozen_tensor, frozen_width = frozen.preprocess_image(path)
    diagnostic_tensor, diagnostic_width = diagnostic.preprocess_image(path)

    assert diagnostic.native_contrast_enabled is True
    assert diagnostic.contrast_factor == 2.0
    assert diagnostic_width == frozen_width
    assert torch.equal(diagnostic_tensor, frozen_tensor)


def test_disabling_native_contrast_skips_only_pil_contrast(
    model_configuration: dict,
    tmp_path: Path,
) -> None:
    path = tmp_path / "contrast-test.png"
    array = np.tile(np.arange(64, dtype=np.uint8), (24, 1))
    Image.fromarray(array).save(path)
    diagnostic = ContrastAblationCRNNAdapter(
        model_configuration,
        ROOT,
        native_contrast_enabled=False,
    )

    actual, width = diagnostic.preprocess_image(path)
    source = Image.open(path).convert("L")
    expected_width = int(round(64 * 32 / 24))
    resized = source.resize((expected_width, 32), Image.Resampling.BILINEAR)
    expected = torch.tensor(np.array(resized), dtype=torch.float32) / 255.0
    expected = ((expected - 0.5) / 0.5).unsqueeze(0)

    assert diagnostic.native_contrast_enabled is False
    assert width == expected_width
    assert actual.shape == (1, 32, expected_width)
    assert torch.equal(actual, expected)


def test_configuration_keeps_native_contrast_default_at_two() -> None:
    model_config = yaml.safe_load(
        (ROOT / "configs" / "models.yaml").read_text(encoding="utf-8")
    )["models"]["crnn"]
    ablation_config = yaml.safe_load(
        (ROOT / "configs" / "ocr_evaluation_phase5b.yaml").read_text(
            encoding="utf-8"
        )
    )["phase5b_ablation"]
    assert model_config["contrast_factor"] == 2.0
    assert ablation_config["native_contrast_factor"] == 2.0
    assert ablation_config["split"] == "val"
