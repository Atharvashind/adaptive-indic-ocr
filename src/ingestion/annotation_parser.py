"""Adapters for explicitly configured annotation formats."""

from __future__ import annotations

import ast
import csv
import json
import logging
import re
from pathlib import Path
from typing import Any

from src.ingestion.base_dataset import (
    AnnotationParseResult,
    AnnotationRecord,
    SUPPORTED_GRANULARITIES,
)

logger = logging.getLogger(__name__)


class UnsupportedAnnotationFormatError(ValueError):
    """Raised when a configured annotation format has no implemented adapter."""


def _normalized_path(path: Path) -> str:
    return str(path.resolve()).casefold()


def _string_value(row: dict[str, Any], field_name: str | None) -> str | None:
    if not field_name:
        return None
    value = row.get(field_name)
    if value is None or str(value).strip() == "":
        return None
    return str(value).strip()


def _parse_bbox_value(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    raw_value = str(value).strip()
    if raw_value == "":
        return None
    try:
        parsed = ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        found = [float(part) for part in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw_value)]
        if len(found) != 4:
            raise ValueError(f"invalid bounding box values: {value!r}")
        parsed = found
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 4:
        raise ValueError(f"bounding box must contain four numeric values: {value!r}")
    try:
        return tuple(float(item) for item in parsed)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid bounding box values: {value!r}") from error


def _parse_bbox(
    row: dict[str, Any], columns: list[str] | None, bbox_field: str | None = None
) -> tuple[float, float, float, float] | None:
    if bbox_field:
        return _parse_bbox_value(row.get(bbox_field))
    if not columns:
        return None
    if len(columns) != 4:
        raise ValueError("bbox_columns must contain exactly four column names")
    values = [row.get(column) for column in columns]
    if any(value is None or str(value).strip() == "" for value in values):
        return None
    try:
        return (
            float(values[0]),
            float(values[1]),
            float(values[2]),
            float(values[3]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid bounding box values: {values}") from error


def _parse_polygon(
    row: dict[str, Any], field_name: str | None
) -> list[tuple[float, float]] | None:
    if not field_name:
        return None
    value = row.get(field_name)
    if value is None or str(value).strip() == "":
        return None
    try:
        points = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ValueError(f"polygon column {field_name!r} must contain JSON") from error
    if not isinstance(points, list):
        raise ValueError("polygon JSON must be a list of [x, y] points")
    if points and all(isinstance(point, (int, float)) for point in points):
        if len(points) % 2:
            raise ValueError("flat polygon coordinate lists must have an even length")
        points = list(zip(points[::2], points[1::2]))
    try:
        parsed = [(float(point[0]), float(point[1])) for point in points]
    except (TypeError, ValueError, IndexError) as error:
        raise ValueError("polygon JSON must contain [x, y] points") from error
    return parsed


def _csv_adapter(
    path: Path, image_root: Path, config: dict[str, Any]
) -> AnnotationParseResult:
    columns = config.get("columns", {})
    if not isinstance(columns, dict):
        raise ValueError("CSV annotation config 'columns' must be a mapping")
    if not columns.get("source_image_id") and not columns.get("image_path"):
        raise ValueError(
            "CSV annotation config must map at least 'source_image_id' or 'image_path'"
        )

    result = AnnotationParseResult()
    delimiter = str(config.get("delimiter", ","))
    encoding = str(config.get("encoding", "utf-8-sig"))
    with path.open("r", encoding=encoding, newline="") as annotation_file:
        reader = csv.DictReader(annotation_file, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"annotation CSV has no header: {path}")
        mapped_fields = {field for field in columns.values() if field}
        bbox_columns = config.get("bbox_columns")
        bbox_field = config.get("bbox_field")
        if bbox_columns:
            if not isinstance(bbox_columns, list) or len(bbox_columns) != 4:
                raise ValueError("bbox_columns must contain exactly four column names")
            mapped_fields.update(bbox_columns)
        if bbox_field:
            mapped_fields.add(str(bbox_field))
        polygon_field = config.get("polygon_field")
        if polygon_field:
            mapped_fields.add(polygon_field)
        missing_fields = mapped_fields - set(reader.fieldnames)
        if missing_fields:
            raise ValueError(
                f"annotation CSV is missing configured columns: {sorted(missing_fields)}"
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                source_image_id = _string_value(row, columns.get("source_image_id"))
                raw_image_path = _string_value(row, columns.get("image_path"))
                image_path: str | None = None
                if raw_image_path:
                    candidate = Path(raw_image_path)
                    if not candidate.is_absolute():
                        candidate = image_root / candidate
                    image_path = str(candidate.resolve())
                    if source_image_id:
                        result.image_ids_by_path[_normalized_path(candidate)] = (
                            source_image_id
                        )

                granularity = (
                    _string_value(row, columns.get("granularity"))
                    or _string_value({"value": config.get("granularity")}, "value")
                )
                if granularity and granularity not in SUPPORTED_GRANULARITIES:
                    result.issues.append(
                        f"CSV line {line_number}: unsupported granularity "
                        f"{granularity!r}; preserved as source metadata"
                    )

                metadata = {
                    key: value
                    for key, value in row.items()
                    if value not in (None, "")
                }
                for field_name in (
                    "domain",
                    "domain_label",
                    "board_type",
                    "split",
                    "crop_width",
                    "crop_height",
                ):
                    source_column = columns.get(field_name)
                    if source_column:
                        value = _string_value(row, source_column)
                        if value is not None:
                            metadata[field_name] = value
                for key, value in row.items():
                    if key not in mapped_fields and value not in (None, ""):
                        metadata.setdefault(key, value)
                record = AnnotationRecord(
                    source_image_id=source_image_id,
                    annotation_id=_string_value(row, columns.get("annotation_id")),
                    parent_annotation_id=_string_value(
                        row, columns.get("parent_annotation_id")
                    ),
                    image_path=image_path,
                    granularity=granularity,
                    transcription=_string_value(row, columns.get("transcription")),
                    language=_string_value(row, columns.get("language")),
                    script=_string_value(row, columns.get("script")),
                    bbox=_parse_bbox(row, bbox_columns, bbox_field),
                    polygon=_parse_polygon(row, polygon_field),
                    metadata=metadata,
                )
                if not record.image_path and not record.source_image_id:
                    result.issues.append(
                        f"CSV line {line_number}: no configured image path or image ID"
                    )
                    continue
                result.annotations.append(record)
            except (TypeError, ValueError) as error:
                result.issues.append(f"CSV line {line_number}: {error}")
                logger.warning("Skipping malformed annotation row: %s", error)
    return result


def _coco_instances_adapter(
    path: Path, image_root: Path, config: dict[str, Any]
) -> AnnotationParseResult:
    """Read the documented COCO instances structure; text fields are opt-in."""
    with path.open("r", encoding=str(config.get("encoding", "utf-8"))) as source:
        document = json.load(source)
    if not isinstance(document, dict):
        raise ValueError("COCO annotation JSON must contain a top-level object")
    images = document.get("images")
    annotations = document.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("COCO instances JSON requires 'images' and 'annotations' lists")

    result = AnnotationParseResult()
    image_metadata: dict[str, tuple[str, str]] = {}
    for index, image in enumerate(images):
        if not isinstance(image, dict) or "id" not in image or "file_name" not in image:
            result.issues.append(f"COCO image entry {index} lacks id or file_name")
            continue
        image_id = str(image["id"])
        image_path = (image_root / str(image["file_name"])).resolve()
        key = _normalized_path(image_path)
        image_metadata[image_id] = (str(image_path), image_id)
        result.image_ids_by_path[key] = image_id

    transcription_field = config.get("transcription_field")
    language_field = config.get("language_field")
    script_field = config.get("script_field")
    granularity_field = config.get("granularity_field")
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            result.issues.append(f"COCO annotation entry {index} is not an object")
            continue
        image_id = str(annotation.get("image_id", ""))
        matched_image = image_metadata.get(image_id)
        if matched_image is None:
            result.issues.append(
                f"COCO annotation entry {index} references unknown image_id {image_id!r}"
            )
        bbox_values = annotation.get("bbox")
        bbox = None
        if bbox_values is not None:
            try:
                if len(bbox_values) != 4:
                    raise ValueError("COCO bbox must contain four values")
                bbox = tuple(float(value) for value in bbox_values)
            except (TypeError, ValueError) as error:
                result.issues.append(f"COCO annotation entry {index}: {error}")
        segmentation = annotation.get("segmentation")
        polygon: list[tuple[float, float]] | None = None
        polygons: list[list[tuple[float, float]]] = []
        if isinstance(segmentation, list):
            for polygon_index, coordinates in enumerate(segmentation):
                if not isinstance(coordinates, list) or len(coordinates) % 2:
                    result.issues.append(
                        f"COCO annotation entry {index}: polygon {polygon_index} "
                        "must be a flat list of coordinate pairs"
                    )
                    continue
                try:
                    polygons.append(list(
                        zip(
                            (float(value) for value in coordinates[::2]),
                            (float(value) for value in coordinates[1::2]),
                        )
                    ))
                except (TypeError, ValueError) as error:
                    result.issues.append(
                        f"COCO annotation entry {index}, polygon {polygon_index}: "
                        f"invalid segmentation: {error}"
                    )
        elif isinstance(segmentation, dict):
            result.issues.append(
                f"COCO annotation entry {index}: compressed/uncompressed RLE "
                "segmentation is not supported by this adapter"
            )
        if polygons:
            polygon = polygons[0]
        granularity = (
            str(annotation[granularity_field])
            if granularity_field and annotation.get(granularity_field) is not None
            else None
        )
        result.annotations.append(
            AnnotationRecord(
                source_image_id=image_id or None,
                annotation_id=(
                    str(annotation["id"]) if annotation.get("id") is not None else None
                ),
                image_path=matched_image[0] if matched_image else None,
                granularity=granularity,
                transcription=(
                    str(annotation[transcription_field])
                    if transcription_field and annotation.get(transcription_field) is not None
                    else None
                ),
                language=(
                    str(annotation[language_field])
                    if language_field and annotation.get(language_field) is not None
                    else None
                ),
                script=(
                    str(annotation[script_field])
                    if script_field and annotation.get(script_field) is not None
                    else None
                ),
                bbox=bbox,
                polygon=polygon,
                polygons=polygons or None,
                metadata={
                    key: value
                    for key, value in annotation.items()
                    if key not in {
                        "id",
                        "image_id",
                        "bbox",
                        "segmentation",
                        transcription_field,
                        language_field,
                        script_field,
                        granularity_field,
                    }
                },
            )
        )
    return result


def parse_text_reference_annotations(
    path: Path,
    image_root: Path,
    config: dict[str, Any] | None = None,
) -> AnnotationParseResult:
    """Parse a text reference file whose first token is an image filename."""
    if config is None:
        config = {}
    result = AnnotationParseResult()
    if not path.is_file():
        result.issues.append(f"Text annotation file not found: {path}")
        return result
    encoding = str(config.get("encoding", "utf-8-sig"))
    granularity = str(config.get("granularity", "line")).strip().lower() or "line"
    split_name = str(config.get("split_name", path.stem)).strip().lower()
    with path.open("r", encoding=encoding, newline="") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue
            match = re.match(r"^(\S+)(?:\s+(.*))?$", line.strip())
            if match is None:
                result.issues.append(
                    f"Text annotation line {line_number} in {path.name} is missing an image filename"
                )
                continue
            filename, transcription = match.groups()
            candidate = Path(filename)
            if not candidate.is_absolute():
                candidate = image_root / candidate
            resolved = candidate.resolve()
            reference_record = AnnotationRecord(
                source_image_id=None,
                annotation_id=f"{path.name}:{line_number}",
                image_path=str(resolved),
                granularity=granularity if granularity in SUPPORTED_GRANULARITIES else None,
                transcription=transcription if transcription is not None else None,
                metadata={
                    "split": split_name,
                    "source_annotation_file": path.name,
                    "source_annotation_format": "text_reference",
                },
            )
            result.annotations.append(reference_record)
            if not resolved.is_file():
                result.issues.append(
                    f"Text annotation line {line_number} references missing image {resolved}"
                )
    return result


def parse_annotations(
    annotation_path: Path | None,
    image_root: Path,
    config: dict[str, Any] | None,
) -> AnnotationParseResult:
    """Parse CSV with an explicit column map, text reference files, or COCO JSON."""
    if config is not None and not isinstance(config, dict):
        raise ValueError("Annotation configuration must be a YAML mapping")
    if not config or not annotation_path:
        return AnnotationParseResult(issues=["No annotation file configured"])
    if not annotation_path.is_file():
        return AnnotationParseResult(
            issues=[f"Annotation file not found: {annotation_path}"]
        )

    annotation_format = str(config.get("format", "")).strip().lower()
    if annotation_format == "csv":
        return _csv_adapter(annotation_path, image_root, config)
    if annotation_format in {"coco_instances", "coco-instances"}:
        return _coco_instances_adapter(annotation_path, image_root, config)
    if annotation_format in {"text", "text_reference"}:
        return parse_text_reference_annotations(annotation_path, image_root, config)
    if annotation_format in {"", "unknown", "none", "null"}:
        raise UnsupportedAnnotationFormatError(
            f"No verified annotation adapter is configured for {annotation_path}. "
            "Set annotation.format to 'csv' with an explicit column map, "
            "'text_reference' for `<filename> <transcription>` files, or "
            "'coco_instances' for COCO instances JSON."
        )
    raise UnsupportedAnnotationFormatError(
        f"Unsupported annotation format {annotation_format!r} for {annotation_path}. "
        "Supported formats are 'csv', 'text_reference', and 'coco_instances'."
    )
