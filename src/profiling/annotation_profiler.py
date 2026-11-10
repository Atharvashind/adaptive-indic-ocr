"""Summarize annotation coverage, granularity, and transcription evidence."""

from __future__ import annotations

from collections import Counter
from statistics import mean, median
from typing import Any

from src.ingestion.base_dataset import DatasetManifest


def profile_annotations(manifest: DatasetManifest) -> dict[str, Any]:
    """Return annotation and transcription statistics without inferring labels."""
    annotations = manifest.annotations
    granularity_counts = Counter(
        annotation.granularity or "unspecified" for annotation in annotations
    )
    language_counts = Counter(
        annotation.language for annotation in annotations if annotation.language
    )
    script_counts = Counter(
        annotation.script for annotation in annotations if annotation.script
    )
    transcription_lengths = [
        len(annotation.transcription)
        for annotation in annotations
        if annotation.transcription is not None
    ]
    images_with_annotations = sum(bool(image.annotations) for image in manifest.images)
    if manifest.annotation_status in {"loaded", "empty"}:
        missing_record_count: int | None = len(manifest.images) - images_with_annotations
        coverage_status = "assessed"
    else:
        missing_record_count = None
        coverage_status = "unavailable"
    return {
        "annotation_status": manifest.annotation_status,
        "annotation_count": len(annotations),
        "images_with_annotations": images_with_annotations,
        "images_without_annotations": missing_record_count,
        "annotation_coverage_status": coverage_status,
        "granularity_distribution": dict(sorted(granularity_counts.items())),
        "language_distribution": dict(sorted(language_counts.items())),
        "script_distribution": dict(sorted(script_counts.items())),
        "transcription_count": len(transcription_lengths),
        "transcription_length_characters": {
            "minimum": min(transcription_lengths) if transcription_lengths else None,
            "median": median(transcription_lengths) if transcription_lengths else None,
            "mean": mean(transcription_lengths) if transcription_lengths else None,
            "maximum": max(transcription_lengths) if transcription_lengths else None,
        },
        "missing_and_inconsistent_records": {
            "annotation_file_missing": manifest.annotation_status == "missing",
            "annotation_file_not_configured": (
                manifest.annotation_status == "not_configured"
            ),
            "unlinked_or_malformed_records": len(manifest.annotation_issues),
            "issues": list(manifest.annotation_issues),
        },
    }
