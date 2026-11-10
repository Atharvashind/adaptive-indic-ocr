"""Independent geometry operations; neither changes the source array."""

from __future__ import annotations

import math

import cv2
import numpy as np

from src.preprocessing.operation import ImageArray, PreprocessingOperation
from src.preprocessing.operations._image_utils import grayscale


def _border_fill(image: ImageArray, value: int) -> int | tuple[int, ...]:
    if image.ndim == 2:
        return value
    return tuple(value for _ in range(image.shape[2]))


class DeskewOperation(PreprocessingOperation):
    """Estimate dominant near-horizontal line angle and rotate without cropping."""

    name = "deskew"
    defaults = {
        "minimum_line_length_ratio": 0.1,
        "minimum_lines": 3,
        "maximum_angle_degrees": 20.0,
        "maximum_angle_spread": 2.0,
        "border_value": 255,
    }
    allowed_parameters = {
        "minimum_line_length_ratio",
        "minimum_lines",
        "maximum_angle_degrees",
        "maximum_angle_spread",
        "border_value",
    }

    def validate_parameters(self, parameters: dict) -> None:
        ratio = float(parameters["minimum_line_length_ratio"])
        maximum_angle = float(parameters["maximum_angle_degrees"])
        spread = float(parameters["maximum_angle_spread"])
        lines = parameters["minimum_lines"]
        border = parameters["border_value"]
        if not 0 < ratio <= 1:
            raise ValueError("minimum_line_length_ratio must be in the range (0, 1]")
        if not 0 < maximum_angle <= 45:
            raise ValueError("maximum_angle_degrees must be in the range (0, 45]")
        if not 0 < spread <= 15:
            raise ValueError("maximum_angle_spread must be in the range (0, 15]")
        if not isinstance(lines, int) or not 1 <= lines <= 1000:
            raise ValueError("minimum_lines must be an integer from 1 to 1000")
        if not isinstance(border, int) or not 0 <= border <= 255:
            raise ValueError("border_value must be an integer from 0 to 255")
        parameters.update(
            minimum_line_length_ratio=ratio,
            maximum_angle_degrees=maximum_angle,
            maximum_angle_spread=spread,
        )

    def _apply(
        self, image: ImageArray, parameters: dict
    ) -> tuple[ImageArray, list[str], dict]:
        height, width = image.shape[:2]
        warning = (
            "Skew is an image-only estimate; non-text edges and scene layouts "
            "can dominate."
        )
        if min(height, width) < 16:
            return image.copy(), [
                warning,
                "Image is too small for a stable skew estimate; no rotation applied.",
            ], {
                "estimated_angle_degrees": None,
                "confidence": 0.0,
                "quality": "unavailable",
                "coordinate_transform_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            }

        edges = cv2.Canny(grayscale(image), 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(10, int(min(height, width) * 0.05)),
            minLineLength=max(
                10, int(width * float(parameters["minimum_line_length_ratio"]))
            ),
            maxLineGap=max(2, int(width * 0.01)),
        )
        angles: list[float] = []
        if lines is not None:
            candidates = lines[:, 0, :] if lines.ndim == 3 else lines
            for line in candidates:
                x1, y1, x2, y2 = (int(value) for value in line[:4])
                angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if abs(angle) <= float(parameters["maximum_angle_degrees"]):
                    angles.append(angle)

        if len(angles) < int(parameters["minimum_lines"]):
            return image.copy(), [
                warning,
                "Insufficient near-horizontal line evidence; no rotation applied.",
            ], {
                "estimated_angle_degrees": None,
                "confidence": 0.0,
                "quality": "insufficient_evidence",
                "candidate_line_count": len(angles),
                "coordinate_transform_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            }

        estimate = float(np.median(angles))
        dispersion = float(np.median(np.abs(np.asarray(angles) - estimate)))
        if dispersion > float(parameters["maximum_angle_spread"]):
            return image.copy(), [
                warning,
                "Candidate line angles disagree; no rotation applied.",
            ], {
                "estimated_angle_degrees": estimate,
                "confidence": 0.0,
                "quality": "inconsistent_evidence",
                "candidate_line_count": len(angles),
                "median_absolute_deviation_degrees": dispersion,
                "coordinate_transform_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            }

        confidence = min(0.95, 0.45 + 0.5 * len(angles) / (len(angles) + 10))
        if abs(estimate) < 0.05:
            output = image.copy()
            matrix = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        else:
            radians = math.radians(estimate)
            new_width = int(
                math.ceil(abs(height * math.sin(radians)) + abs(width * math.cos(radians)))
            )
            new_height = int(
                math.ceil(abs(height * math.cos(radians)) + abs(width * math.sin(radians)))
            )
            matrix = cv2.getRotationMatrix2D(
                (width / 2, height / 2), estimate, 1.0
            )
            matrix[0, 2] += new_width / 2 - width / 2
            matrix[1, 2] += new_height / 2 - height / 2
            output = cv2.warpAffine(
                image,
                matrix,
                (new_width, new_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=_border_fill(image, int(parameters["border_value"])),
            )
        return output, [warning], {
            "estimated_angle_degrees": estimate,
            "confidence": confidence,
            "quality": "consistent_near_horizontal_lines",
            "candidate_line_count": len(angles),
            "median_absolute_deviation_degrees": dispersion,
            "coordinate_transform_matrix": matrix.tolist(),
        }


class AspectRatioPaddingOperation(PreprocessingOperation):
    """Pad to the next multiple without resizing or distorting source pixels."""

    name = "aspect_ratio_padding"
    defaults = {"multiple": 32, "fill_value": 255}
    allowed_parameters = {"multiple", "fill_value"}

    def validate_parameters(self, parameters: dict) -> None:
        multiple = parameters["multiple"]
        fill = parameters["fill_value"]
        if not isinstance(multiple, int) or not 2 <= multiple <= 512:
            raise ValueError("multiple must be an integer from 2 to 512")
        if not isinstance(fill, int) or not 0 <= fill <= 255:
            raise ValueError("fill_value must be an integer from 0 to 255")

    def _apply(
        self, image: ImageArray, parameters: dict
    ) -> tuple[ImageArray, list[str], dict]:
        height, width = image.shape[:2]
        multiple = int(parameters["multiple"])
        target_width = int(math.ceil(width / multiple) * multiple)
        target_height = int(math.ceil(height / multiple) * multiple)
        horizontal = target_width - width
        vertical = target_height - height
        left, right = horizontal // 2, horizontal - horizontal // 2
        top, bottom = vertical // 2, vertical - vertical // 2
        output = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=_border_fill(image, int(parameters["fill_value"])),
        )
        warnings = [
            "Original pixels were not resized; padding changes the canvas dimensions."
        ]
        return output, warnings, {
            "padding": {"top": top, "bottom": bottom, "left": left, "right": right},
            "source_aspect_ratio": width / height,
            "source_pixels_resized": False,
            "coordinate_transform_matrix": [
                [1.0, 0.0, float(left)],
                [0.0, 1.0, float(top)],
            ],
        }
