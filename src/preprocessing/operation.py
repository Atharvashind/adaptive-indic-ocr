"""Shared result contract and base class for image preprocessing operations."""

from __future__ import annotations

import copy
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


ImageArray = np.ndarray
Dimensions = tuple[int, int]


@dataclass
class OperationResult:
    """Outcome of one independent, non-destructive image operation."""

    processed_image: ImageArray | None
    operation_name: str
    parameters: dict[str, Any]
    input_dimensions: Dimensions | None
    output_dimensions: Dimensions | None
    execution_time_ms: float
    success: bool
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        """Return a simple stable status suitable for reports."""
        return "success" if self.success else "failure"

    def to_dict(self, include_metadata: bool = True) -> dict[str, Any]:
        """Return JSON-compatible transformation details, excluding pixels."""
        result: dict[str, Any] = {
            "operation_name": self.operation_name,
            "parameters": self.parameters,
            "input_dimensions": _dimensions_dict(self.input_dimensions),
            "output_dimensions": _dimensions_dict(self.output_dimensions),
            "execution_time_ms": self.execution_time_ms,
            "success": self.success,
            "status": self.status,
            "warnings": list(self.warnings),
            "details": self.details,
        }
        if include_metadata:
            result["metadata"] = self.metadata
        return result


def _dimensions_dict(dimensions: Dimensions | None) -> dict[str, int] | None:
    if dimensions is None:
        return None
    width, height = dimensions
    return {"width": width, "height": height}


def image_dimensions(image: ImageArray) -> Dimensions:
    """Return image dimensions as (width, height)."""
    height, width = image.shape[:2]
    return int(width), int(height)


def _transform_annotation_geometry(
    metadata: dict[str, Any], matrix: list[list[float]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Transform annotation geometry explicitly declared in input-image space."""
    annotations = metadata.get("annotations")
    if not isinstance(annotations, list):
        return [], []

    affine = np.asarray(matrix, dtype=np.float64)
    if affine.shape != (2, 3):
        return [], ["Annotation geometry was not transformed: invalid affine matrix."]

    transformed: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            continue
        bbox = annotation.get("bbox")
        polygon = annotation.get("polygon")
        if bbox is None and polygon is None:
            continue

        space = annotation.get(
            "bbox_coordinate_space", metadata.get("bbox_coordinate_space")
        )
        entry: dict[str, Any] = {
            "annotation_index": index,
            "source_image_id": annotation.get("source_image_id"),
            "input_coordinate_space": space,
            "output_coordinate_space": (
                "output_image"
                if space in {"input_image", "input"}
                else space
            ),
            "transformation_status": "retained_in_source_coordinates",
        }
        if space not in {"input_image", "input"}:
            entry["reason"] = (
                "Coordinates are not declared to be relative to the input image."
            )
            transformed.append(entry)
            if space is None:
                warnings.append(
                    f"Annotation {index} geometry was not transformed because its "
                    "coordinate space is unspecified."
                )
            continue

        try:
            if bbox is not None:
                if annotation.get("bbox_format", metadata.get("bbox_format")) != "xyxy":
                    raise ValueError("bbox format must be explicitly declared as xyxy")
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    raise ValueError("bbox must contain four coordinates")
                x1, y1, x2, y2 = (float(value) for value in bbox)
                corners = np.asarray(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    dtype=np.float64,
                )
                points = np.column_stack((corners, np.ones(4))) @ affine.T
                entry["input_bbox"] = list(bbox)
                entry["output_bbox"] = [
                    float(points[:, 0].min()),
                    float(points[:, 1].min()),
                    float(points[:, 0].max()),
                    float(points[:, 1].max()),
                ]
            if polygon is not None:
                points = np.asarray(polygon, dtype=np.float64)
                if points.ndim != 2 or points.shape[1] != 2:
                    raise ValueError("polygon must contain [x, y] points")
                transformed_points = np.column_stack(
                    (points, np.ones(len(points)))
                ) @ affine.T
                entry["input_polygon"] = polygon
                entry["output_polygon"] = transformed_points.tolist()
            entry["transformation_status"] = "transformed"
        except (TypeError, ValueError) as error:
            entry["transformation_status"] = "not_transformed"
            entry["reason"] = str(error)
            warnings.append(
                f"Annotation {index} geometry was not transformed: {error}."
            )
        transformed.append(entry)
    return transformed, warnings


class PreprocessingOperation(ABC):
    """Base class validating inputs and enforcing the operation result contract."""

    name: str
    defaults: dict[str, Any] = {}
    allowed_parameters: set[str] = set()

    def execute(
        self,
        image: ImageArray,
        parameters: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OperationResult:
        """Run this operation once without allowing it to mutate the input."""
        started = time.perf_counter()
        input_dimensions: Dimensions | None = None
        used_parameters = dict(self.defaults)
        warnings: list[str] = []
        source_metadata = copy.deepcopy(metadata or {})
        try:
            input_dimensions = self._validate_image(image)
            supplied = {} if parameters is None else parameters
            if not isinstance(supplied, dict):
                raise ValueError("parameters must be a mapping")
            unknown = set(supplied) - self.allowed_parameters
            if unknown:
                raise ValueError(f"unknown parameters: {sorted(unknown)}")
            used_parameters.update(supplied)
            self.validate_parameters(used_parameters)
            # Operations may use OpenCV routines that support in-place mutation;
            # the copy establishes a hard immutability boundary for the caller.
            safe_input = np.array(image, copy=True)
            output, warnings, details = self._apply(safe_input, used_parameters)
            transform = details.get("coordinate_transform_matrix")
            if transform is not None:
                annotation_geometry, geometry_warnings = (
                    _transform_annotation_geometry(source_metadata, transform)
                )
                if annotation_geometry:
                    details["annotation_geometry"] = annotation_geometry
                warnings.extend(geometry_warnings)
            self._validate_image(output)
            if np.shares_memory(output, image):
                output = np.array(output, copy=True)
            return OperationResult(
                processed_image=output,
                operation_name=self.name,
                parameters=used_parameters,
                input_dimensions=input_dimensions,
                output_dimensions=image_dimensions(output),
                execution_time_ms=(time.perf_counter() - started) * 1000,
                success=True,
                warnings=warnings,
                metadata=source_metadata,
                details=details,
            )
        except (TypeError, ValueError, cv2.error) as error:
            warnings.append(str(error))
            return OperationResult(
                processed_image=None,
                operation_name=self.name,
                parameters=used_parameters,
                input_dimensions=input_dimensions,
                output_dimensions=None,
                execution_time_ms=(time.perf_counter() - started) * 1000,
                success=False,
                warnings=warnings,
                metadata=source_metadata,
            )

    @staticmethod
    def _validate_image(image: ImageArray) -> Dimensions:
        if not isinstance(image, np.ndarray):
            raise ValueError("input image must be a NumPy array")
        if image.size == 0 or image.ndim not in {2, 3}:
            raise ValueError("input image must be non-empty grayscale or colour data")
        if image.ndim == 3 and image.shape[2] not in {1, 3, 4}:
            raise ValueError("colour image must have 1, 3, or 4 channels")
        if image.dtype != np.uint8:
            raise ValueError("input image dtype must be uint8")
        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError("input image dimensions must be positive")
        return int(width), int(height)

    def validate_parameters(self, parameters: dict[str, Any]) -> None:
        """Validate operation-specific parameters."""

    @abstractmethod
    def _apply(
        self,
        image: ImageArray,
        parameters: dict[str, Any],
    ) -> tuple[ImageArray, list[str], dict[str, Any]]:
        """Return output pixels, warnings, and operation-specific details."""
