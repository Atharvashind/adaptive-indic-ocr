"""Colour-space helpers shared by independent preprocessing operations."""

from __future__ import annotations

import cv2
import numpy as np

from src.preprocessing.operation import ImageArray


def split_alpha(image: ImageArray) -> tuple[ImageArray, ImageArray | None]:
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3], image[:, :, 3].copy()
    return image, None


def grayscale(image: ImageArray) -> ImageArray:
    if image.ndim == 2:
        return image
    color, _ = split_alpha(image)
    if color.ndim == 2 or color.shape[2] == 1:
        return color.reshape(color.shape[:2])
    return cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)


def restore_alpha(
    image: ImageArray, alpha: ImageArray | None
) -> ImageArray:
    if alpha is None:
        return image
    return np.dstack((image, alpha))


def apply_to_color_planes(
    image: ImageArray,
    transform,
) -> ImageArray:
    """Apply a single-plane transform while preserving grayscale/RGB/RGBA."""
    color, alpha = split_alpha(image)
    if color.ndim == 2 or color.shape[2] == 1:
        plane = color.reshape(color.shape[:2])
        output = transform(plane)
        if color.ndim == 3:
            output = output[:, :, None]
        return restore_alpha(output, alpha)
    output = transform(color)
    return restore_alpha(output, alpha)
