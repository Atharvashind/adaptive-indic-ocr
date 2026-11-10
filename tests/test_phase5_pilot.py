"""Reproducibility checks for Phase 5A pilot sampling and model contexts."""

from __future__ import annotations

from collections import Counter

from experiments.ocr_evaluation.run_phase5_pilot import (
    _baseline_groups,
    _stratified_unique_source_sample,
)


def _rows() -> list[dict[str, str]]:
    rows = []
    for language, count in (("bn", 120), ("en", 80)):
        for source_index in range(count):
            source_id = f"{language}-board-{source_index:03d}"
            for line_index in range(2):
                rows.append(
                    {
                        "filename": f"{source_id}-line-{line_index}.png",
                        "source_image_id": source_id,
                        "language": language,
                        "split": "test",
                        "crop_width": str(80 + line_index),
                        "crop_height": "32",
                    }
                )
    return rows


def test_pilot_sampler_is_seeded_stratified_and_unique_by_source() -> None:
    rows = _rows()
    first = _stratified_unique_source_sample(rows, sample_size=40, seed=7)
    second = _stratified_unique_source_sample(rows, sample_size=40, seed=7)

    assert [row["filename"] for row in first] == [row["filename"] for row in second]
    assert len(first) == 40
    assert len({row["source_image_id"] for row in first}) == 40
    assert Counter(row["language"] for row in first) == {"bn": 24, "en": 16}


def test_pilot_sampler_fails_when_unique_source_count_is_too_small() -> None:
    rows = _rows()[:20]
    for row in rows:
        row["source_image_id"] = "same-board"

    try:
        _stratified_unique_source_sample(rows, sample_size=2, seed=7)
    except ValueError as error:
        assert "unique feasible source boards" in str(error)
    else:
        raise AssertionError("Expected unique-source population validation to fail")


def test_baseline_contexts_retain_sequential_batch_padding_width() -> None:
    rows = [
        {
            "filename": f"{index}.png",
            "crop_width": str(width),
            "crop_height": "32",
        }
        for index, width in enumerate((33, 40, 50, 64, 81))
    ]

    contexts = _baseline_groups(rows, input_height=32, batch_size=3)

    assert contexts == {
        "0.png": (0, 52),
        "1.png": (0, 52),
        "2.png": (0, 52),
        "3.png": (1, 84),
        "4.png": (1, 84),
    }
