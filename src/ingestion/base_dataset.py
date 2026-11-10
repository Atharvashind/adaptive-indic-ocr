"""Shared data structures for dataset ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_GRANULARITIES = {
    "image",
    "page",
    "region",
    "line",
    "line_crop",
    "word",
    "character",
}


@dataclass
class AnnotationRecord:
    """One normalized annotation while retaining its source identifiers."""

    source_image_id: str | None = None
    annotation_id: str | None = None
    parent_annotation_id: str | None = None
    image_path: str | None = None
    granularity: str | None = None
    transcription: str | None = None
    language: str | None = None
    script: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    polygon: list[tuple[float, float]] | None = None
    polygons: list[list[tuple[float, float]]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnnotationParseResult:
    """Annotations and source-image metadata returned by an adapter."""

    annotations: list[AnnotationRecord] = field(default_factory=list)
    image_ids_by_path: dict[str, str] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    reference_annotations: list[AnnotationRecord] = field(default_factory=list)
    reference_issues: list[str] = field(default_factory=list)


@dataclass
class DatasetImage:
    """A discovered image and its linked annotation records."""

    path: Path
    relative_path: str
    source_image_id: str | None = None
    annotations: list[AnnotationRecord] = field(default_factory=list)


@dataclass
class DatasetManifest:
    """An in-memory, read-only view of a configured dataset."""

    name: str
    image_root: Path
    images: list[DatasetImage]
    annotations: list[AnnotationRecord]
    annotation_status: str
    annotation_issues: list[str] = field(default_factory=list)
    discovered_image_count: int | None = None
    sample_limit: int | None = None
    sample_seed: int | None = None
    reference_annotations: list[AnnotationRecord] = field(default_factory=list)
    dataset_metadata: dict[str, Any] = field(default_factory=dict)
