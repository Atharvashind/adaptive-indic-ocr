"""Dataset loading and annotation parsing."""

from src.ingestion.annotation_parser import (
    UnsupportedAnnotationFormatError,
    parse_annotations,
)
from src.ingestion.base_dataset import AnnotationRecord, DatasetImage, DatasetManifest
from src.ingestion.dataset_loader import (
    DatasetConfigurationError,
    discover_images,
    load_dataset,
)

__all__ = [
    "AnnotationRecord",
    "DatasetConfigurationError",
    "DatasetImage",
    "DatasetManifest",
    "UnsupportedAnnotationFormatError",
    "discover_images",
    "load_dataset",
    "parse_annotations",
]
