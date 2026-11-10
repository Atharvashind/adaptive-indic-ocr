"""Separate domain metadata from preliminary automatic evidence."""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.ingestion.base_dataset import DatasetManifest

SUPPORTED_DOMAINS = {
    "handwritten",
    "printed_document",
    "scene_text",
    "mixed",
    "unknown",
}


def profile_domains(
    manifest: DatasetManifest,
    configured_labels: list[str] | None = None,
    user_domain: str | None = None,
) -> dict[str, Any]:
    """Report metadata labels separately from unavailable image classification."""
    metadata_evidence: list[dict[str, str]] = []
    dataset_metadata = getattr(manifest, "dataset_metadata", {}) or {}
    source_domain = dataset_metadata.get("source_domain")
    source_content = dataset_metadata.get("source_content")
    image_granularity = dataset_metadata.get("image_granularity")
    for label in configured_labels or []:
        metadata_evidence.append({"label": str(label), "source": "dataset_configuration"})
    if user_domain:
        metadata_evidence.append({"label": user_domain, "source": "user_specified"})

    annotation_labels: Counter[str] = Counter()
    for annotation in manifest.annotations:
        for key in ("domain", "domain_label"):
            label = annotation.metadata.get(key)
            if label:
                annotation_labels[str(label)] += 1
    metadata_evidence.extend(
        {"label": label, "source": "annotation_metadata"}
        for label in sorted(annotation_labels)
    )

    recognized = {
        evidence["label"]
        for evidence in metadata_evidence
        if evidence["label"] in SUPPORTED_DOMAINS
    }
    return {
        "source_domain": str(source_domain) if source_domain is not None else None,
        "source_content": str(source_content) if source_content is not None else None,
        "image_granularity": (
            str(image_granularity) if image_granularity is not None else None
        ),
        "metadata_domain_labels": metadata_evidence,
        "metadata_label_counts": dict(sorted(annotation_labels.items())),
        "multi_label_candidates_from_metadata": sorted(recognized),
        "automatic_assessment": {
            "status": "preliminary_metadata_only",
            "image_based_predictions": {
                "status": "unavailable",
                "labels": [],
                "reason": (
                    "No handwriting, print, or scene-text classifier is implemented; "
                    "image statistics are not reliable domain labels."
                ),
            },
            "available_annotation_evidence": {
                "granularities": sorted(
                    {
                        annotation.granularity
                        for annotation in manifest.annotations
                        if annotation.granularity
                    }
                ),
                "transcription_records": sum(
                    annotation.transcription is not None
                    for annotation in manifest.annotations
                ),
                "metadata_domain_label_counts": dict(sorted(annotation_labels.items())),
            },
        },
    }
