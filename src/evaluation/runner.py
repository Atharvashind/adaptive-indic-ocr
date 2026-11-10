"""Model-independent batching and prediction collection for OCR evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.models.base_adapter import BaseOCRAdapter


@dataclass(frozen=True)
class OCRExample:
    image_path: Path
    filename: str
    ground_truth: str
    language: str
    source_image_id: str = ""


def run_inference(
    adapter: BaseOCRAdapter,
    examples: list[OCRExample],
    batch_size: int,
) -> list[dict[str, str]]:
    """Run model predictions in input order without model-specific tensor logic."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    results: list[dict[str, str]] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        predictions = adapter.predict([example.image_path for example in batch])
        if len(predictions) != len(batch):
            raise RuntimeError(
                f"Adapter returned {len(predictions)} predictions for {len(batch)} examples"
            )
        for example, prediction in zip(batch, predictions, strict=True):
            if not isinstance(prediction, str):
                raise TypeError("OCR adapters must return string predictions")
            results.append(
                {
                    "filename": example.filename,
                    "source_image_id": example.source_image_id,
                    "language": example.language,
                    "ground_truth": example.ground_truth,
                    "prediction": prediction,
                }
            )
    return results
