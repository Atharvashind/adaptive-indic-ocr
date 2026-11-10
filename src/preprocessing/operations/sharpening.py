"""Independent sharpening operations."""

from __future__ import annotations

import cv2
import numpy as np

from src.preprocessing.operation import ImageArray, PreprocessingOperation
from src.preprocessing.operations._image_utils import restore_alpha, split_alpha


class UnsharpMaskOperation(PreprocessingOperation):
    """Apply unsharp masking to a grayscale or colour input."""

    name = "unsharp_mask"
    defaults = {"amount": 1.0, "sigma": 1.0, "threshold": 0}
    allowed_parameters = {"amount", "sigma", "threshold"}

    def validate_parameters(self, parameters: dict) -> None:
        amount = float(parameters["amount"])
        sigma = float(parameters["sigma"])
        threshold = parameters["threshold"]
        if not 0 <= amount <= 5:
            raise ValueError("amount must be in the range [0, 5]")
        if not 0 < sigma <= 50:
            raise ValueError("sigma must be in the range (0, 50]")
        if not isinstance(threshold, int) or not 0 <= threshold <= 255:
            raise ValueError("threshold must be an integer from 0 to 255")
        parameters["amount"] = amount
        parameters["sigma"] = sigma

    def _apply(
        self, image: ImageArray, parameters: dict
    ) -> tuple[ImageArray, list[str], dict]:
        amount = float(parameters["amount"])
        color, alpha = split_alpha(image)
        blurred = cv2.GaussianBlur(
            color,
            (0, 0),
            sigmaX=float(parameters["sigma"]),
            sigmaY=float(parameters["sigma"]),
        )
        sharpened = cv2.addWeighted(color, 1 + amount, blurred, -amount, 0)
        threshold = int(parameters["threshold"])
        if threshold:
            difference = cv2.absdiff(color, blurred)
            if difference.ndim == 3:
                difference = np.max(difference[:, :, :3], axis=2)
            mask = difference >= threshold
            if sharpened.ndim == 3:
                mask = mask[:, :, None]
            output = np.where(mask, sharpened, color).astype(np.uint8)
        else:
            output = sharpened
        return restore_alpha(output, alpha), [], {}
