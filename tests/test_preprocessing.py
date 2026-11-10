"""Tests for independently callable Phase 2 preprocessing operations."""

from __future__ import annotations

import csv
import json

import cv2
import numpy as np
import pytest

from src.preprocessing import create_default_registry
from src.preprocessing.operation import OperationResult
from src.preprocessing.transformation_log import TransformationLog


@pytest.fixture
def registry():
    return create_default_registry()


@pytest.mark.parametrize(
    "operation_name",
    [
        "clahe",
        "histogram_equalization",
        "gaussian_filter",
        "median_filter",
        "bilateral_filter",
        "otsu_threshold",
        "adaptive_gaussian_threshold",
        "adaptive_mean_threshold",
        "unsharp_mask",
        "deskew",
        "aspect_ratio_padding",
    ],
)
@pytest.mark.parametrize(
    "shape",
    [(32, 48), (32, 48, 3)],
    ids=["grayscale", "rgb"],
)
def test_operations_support_grayscale_and_rgb_without_mutation(
    registry, operation_name: str, shape: tuple[int, ...]
) -> None:
    source = np.random.default_rng(42).integers(0, 256, size=shape, dtype=np.uint8)
    original = source.copy()
    metadata = {
        "filename": "line_000001.png",
        "source_image_id": "board_1",
        "transcription": "भारत निर्माण",
    }

    result = registry.get(operation_name).execute(source, metadata=metadata)

    assert result.success is True
    assert result.status == "success"
    assert result.processed_image is not None
    assert result.processed_image.shape[:2] == result.output_dimensions[::-1]
    assert result.input_dimensions == (48, 32)
    assert result.processed_image is not source
    assert np.array_equal(source, original)
    assert result.metadata == metadata
    assert result.metadata["transcription"] == "भारत निर्माण"


@pytest.mark.parametrize(
    ("operation_name", "parameters"),
    [
        ("clahe", {"clip_limit": 0}),
        ("gaussian_filter", {"kernel_size": 4}),
        ("median_filter", {"kernel_size": 2}),
        ("bilateral_filter", {"sigma_color": 0}),
        ("otsu_threshold", {"threshold_type": "local"}),
        ("adaptive_gaussian_threshold", {"block_size": 4}),
        ("adaptive_mean_threshold", {"constant": 999}),
        ("unsharp_mask", {"amount": 6}),
        ("deskew", {"maximum_angle_degrees": 90}),
        ("aspect_ratio_padding", {"multiple": 1}),
    ],
)
def test_invalid_parameters_return_explicit_failure(
    registry, operation_name: str, parameters: dict
) -> None:
    source = np.zeros((20, 20), dtype=np.uint8)

    result = registry.get(operation_name).execute(source, parameters)

    assert result.success is False
    assert result.status == "failure"
    assert result.processed_image is None
    assert result.warnings


def test_non_mapping_parameters_are_rejected(registry) -> None:
    result = registry.get("clahe").execute(
        np.zeros((8, 8), dtype=np.uint8), parameters=[]
    )

    assert result.success is False
    assert any("parameters must be a mapping" in warning for warning in result.warnings)


def test_small_images_are_safe_and_adaptive_threshold_reports_failure(
    registry,
) -> None:
    small = np.arange(6, dtype=np.uint8).reshape(2, 3)

    median_result = registry.get("median_filter").execute(small)
    deskew_result = registry.get("deskew").execute(small)
    adaptive_result = registry.get("adaptive_mean_threshold").execute(small)

    assert median_result.success is True
    assert median_result.details["effective_kernel_size"] == 1
    assert deskew_result.success is True
    assert deskew_result.details["quality"] == "unavailable"
    assert adaptive_result.success is False
    assert any("at least 3" in warning for warning in adaptive_result.warnings)


@pytest.mark.parametrize(
    "source",
    [
        np.empty((0, 0), dtype=np.uint8),
        np.zeros((10, 10), dtype=np.float32),
        np.zeros((10, 10, 2), dtype=np.uint8),
    ],
)
def test_invalid_images_return_failure(registry, source: np.ndarray) -> None:
    result = registry.get("clahe").execute(source)

    assert result.success is False
    assert result.processed_image is None
    assert result.input_dimensions is None
    assert result.warnings


def test_registry_has_unique_names_and_rejects_duplicate(registry) -> None:
    assert len(registry.names()) == 11
    with pytest.raises(ValueError, match="already registered"):
        registry.register(registry.get("clahe"))
    with pytest.raises(KeyError, match="Unknown preprocessing operation"):
        registry.get("not_registered")


def test_operation_metadata_and_dimensions_are_recorded(registry) -> None:
    source = np.zeros((12, 20, 3), dtype=np.uint8)
    result: OperationResult = registry.get("aspect_ratio_padding").execute(
        source,
        {"multiple": 8},
        {"filename": "crop.png", "source_image_id": "board_9"},
    )

    record = result.to_dict()
    assert record["operation_name"] == "aspect_ratio_padding"
    assert record["input_dimensions"] == {"width": 20, "height": 12}
    assert record["output_dimensions"] == {"width": 24, "height": 16}
    assert record["success"] is True
    assert record["status"] == "success"
    assert record["execution_time_ms"] >= 0
    assert record["metadata"]["source_image_id"] == "board_9"


def test_operations_are_deterministic_for_same_inputs_and_parameters(registry) -> None:
    source = np.random.default_rng(7).integers(0, 256, (40, 60), dtype=np.uint8)

    for operation_name in registry.names():
        first = registry.get(operation_name).execute(source)
        second = registry.get(operation_name).execute(source)
        assert first.success == second.success
        if first.success:
            assert np.array_equal(first.processed_image, second.processed_image)


def test_padding_transforms_only_declared_input_image_boxes(registry) -> None:
    source = np.zeros((12, 20), dtype=np.uint8)
    metadata = {
        "annotations": [
            {
                "source_image_id": "board-1",
                "bbox": (1, 2, 5, 6),
                "bbox_format": "xyxy",
                "bbox_coordinate_space": "input_image",
            },
            {
                "source_image_id": "board-2",
                "bbox": (700, 120, 840, 160),
                "bbox_format": "xyxy",
                "bbox_coordinate_space": "source_image",
            },
        ]
    }

    result = registry.get("aspect_ratio_padding").execute(
        source, {"multiple": 8}, metadata
    )

    assert result.success
    assert result.details["padding"] == {
        "top": 2,
        "bottom": 2,
        "left": 2,
        "right": 2,
    }
    transformed, retained = result.details["annotation_geometry"]
    assert transformed["transformation_status"] == "transformed"
    assert transformed["input_bbox"] == [1, 2, 5, 6]
    assert transformed["output_bbox"] == [3.0, 4.0, 7.0, 8.0]
    assert retained["transformation_status"] == "retained_in_source_coordinates"
    assert retained["output_coordinate_space"] == "source_image"
    assert metadata["annotations"][0]["bbox"] == (1, 2, 5, 6)


def test_deskew_applies_the_estimated_correction_angle(registry) -> None:
    horizontal_lines = np.full((120, 400), 255, dtype=np.uint8)
    for y in (25, 50, 75, 100):
        cv2.line(horizontal_lines, (30, y), (370, y), 0, 2)
    input_rotation = cv2.getRotationMatrix2D((200, 60), 5, 1.0)
    source = cv2.warpAffine(
        horizontal_lines, input_rotation, (400, 120), borderValue=255
    )

    result = registry.get("deskew").execute(source)

    assert result.success
    estimate = result.details["estimated_angle_degrees"]
    assert estimate < -3
    transform = np.asarray(
        result.details["coordinate_transform_matrix"], dtype=np.float64
    )
    applied_angle = np.degrees(np.arctan2(transform[0, 1], transform[0, 0]))
    assert estimate is not None
    assert applied_angle == pytest.approx(estimate)
    assert result.output_dimensions != result.input_dimensions


def test_unsharp_mask_preserves_rgba_alpha_channel(registry) -> None:
    source = np.zeros((24, 24, 4), dtype=np.uint8)
    source[:, :, :3] = 127
    source[8:16, 8:16, 3] = 200
    original_alpha = source[:, :, 3].copy()

    result = registry.get("unsharp_mask").execute(source)

    assert result.success
    assert np.array_equal(result.processed_image[:, :, 3], original_alpha)
    assert np.array_equal(source[:, :, 3], original_alpha)


def test_transformation_log_records_saved_paths_and_failures(
    registry, tmp_path
) -> None:
    log = TransformationLog()
    successful = registry.get("clahe").execute(np.zeros((8, 8), dtype=np.uint8))
    failed = registry.get("adaptive_mean_threshold").execute(
        np.zeros((2, 2), dtype=np.uint8)
    )
    log.add(
        successful,
        sample_index=1,
        source_image_id="board-1",
        filename="line.png",
        image_path="images/line.png",
        processed_image="sample-01/clahe/processed.png",
        comparison_image="sample-01/clahe/comparison.png",
    )
    log.add(
        failed,
        sample_index=2,
        source_image_id="board-2",
        filename="line-2.png",
        image_path="images/line-2.png",
    )

    json_path = tmp_path / "transformations.json"
    csv_path = tmp_path / "transformations.csv"
    log.write_json(json_path)
    log.write_csv(csv_path)

    records = json.loads(json_path.read_text(encoding="utf-8"))
    assert records[0]["processed_image"] == "sample-01/clahe/processed.png"
    assert records[0]["comparison_image"] == "sample-01/clahe/comparison.png"
    assert records[1]["status"] == "failure"
    assert records[1]["processed_image"] is None
    with csv_path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    assert [row["status"] for row in rows] == ["success", "failure"]
    assert rows[1]["processed_image"] == ""


@pytest.mark.parametrize(
    ("shape", "pixel"),
    [
        ((12, 20), 37),
        ((12, 20, 3), (17, 83, 149)),
        ((12, 20, 3), (149, 83, 17)),
    ],
    ids=["grayscale", "rgb", "bgr"],
)
def test_padding_uses_neutral_white_and_preserves_input(
    registry, shape: tuple[int, ...], pixel: int | tuple[int, int, int]
) -> None:
    source = np.empty(shape, dtype=np.uint8)
    source[...] = pixel
    original = source.copy()

    result = registry.get("aspect_ratio_padding").execute(
        source, {"multiple": 8, "fill_value": 255}
    )

    assert result.success
    assert result.output_dimensions == (24, 16)
    assert np.array_equal(source, original)
    padding = result.details["padding"]
    output = result.processed_image
    assert np.all(output[: padding["top"]] == 255)
    assert np.all(output[-padding["bottom"] :] == 255) if padding["bottom"] else True
    assert np.all(output[:, : padding["left"]] == 255) if padding["left"] else True
    assert np.all(output[:, -padding["right"] :] == 255) if padding["right"] else True
    assert np.array_equal(
        output[
            padding["top"] : padding["top"] + source.shape[0],
            padding["left"] : padding["left"] + source.shape[1],
        ],
        original,
    )


def test_rgba_padding_uses_opaque_white_and_preserves_source_alpha(registry) -> None:
    source = np.zeros((12, 20, 4), dtype=np.uint8)
    source[:, :, :3] = (13, 42, 91)
    source[:, :, 3] = 123
    original = source.copy()

    result = registry.get("aspect_ratio_padding").execute(
        source, {"multiple": 8, "fill_value": 255}
    )

    assert result.success
    assert np.all(result.processed_image[0, 0] == [255, 255, 255, 255])
    assert np.array_equal(source, original)
    assert np.array_equal(result.processed_image[2:14, 2:22], original)


@pytest.mark.parametrize("channels", [1, 3, 4], ids=["grayscale", "rgb-bgr", "rgba"])
def test_deskew_uses_neutral_white_border_without_cropping(
    registry, channels: int
) -> None:
    horizontal_lines = np.full((120, 400), 255, dtype=np.uint8)
    for y in (25, 50, 75, 100):
        cv2.line(horizontal_lines, (30, y), (370, y), 0, 2)
    input_rotation = cv2.getRotationMatrix2D((200, 60), 5, 1.0)
    skewed = cv2.warpAffine(
        horizontal_lines, input_rotation, (400, 120), borderValue=255
    )
    if channels == 1:
        source = skewed
    else:
        source = np.repeat(skewed[:, :, None], channels, axis=2)
        if channels == 4:
            source[:, :, 3] = 255
    original = source.copy()

    result = registry.get("deskew").execute(source)

    assert result.success
    assert result.output_dimensions[0] > source.shape[1]
    assert result.output_dimensions[1] > source.shape[0]
    assert np.array_equal(source, original)
    if channels > 1:
        assert np.all(result.processed_image[0] == 255)
        assert np.all(result.processed_image[-1] == 255)
        assert np.all(result.processed_image[:, 0] == 255)
        assert np.all(result.processed_image[:, -1] == 255)
    else:
        assert np.all(result.processed_image[0] == 255)
        assert np.all(result.processed_image[-1] == 255)
        assert np.all(result.processed_image[:, 0] == 255)
        assert np.all(result.processed_image[:, -1] == 255)
    assert np.count_nonzero(result.processed_image < 200) >= int(
        np.count_nonzero(source < 200) * 0.75
    )
