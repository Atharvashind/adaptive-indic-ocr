"""Character and word error rate calculations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jiwer


def calculate_cer_wer(
    references: Sequence[str],
    hypotheses: Sequence[str],
) -> dict[str, Any]:
    """Calculate unnormalized JiWER metrics and raw-string exact-match accuracy."""
    if len(references) != len(hypotheses):
        raise ValueError("Reference and hypothesis counts must match")
    if not references:
        raise ValueError("Cannot calculate OCR metrics for an empty evaluation")
    exact_matches = sum(
        reference == hypothesis
        for reference, hypothesis in zip(references, hypotheses, strict=True)
    )
    return {
        "sample_count": len(references),
        "cer": float(jiwer.cer(list(references), list(hypotheses))),
        "wer": float(jiwer.wer(list(references), list(hypotheses))),
        "exact_match_count": exact_matches,
        "exact_match_accuracy": exact_matches / len(references),
    }
