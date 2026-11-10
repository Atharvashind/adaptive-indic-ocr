"""Diagnostic-only CRNN adapter with native contrast enhancement ablation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.models.base_adapter import ImageInput
from src.models.crnn_adapter import CRNNAdapter


class ContrastAblationCRNNAdapter(CRNNAdapter):
    """Reuse the verified CRNN model path while optionally skipping contrast."""

    def __init__(
        self,
        configuration: dict[str, Any],
        project_root: Path,
        *,
        native_contrast_enabled: bool,
    ) -> None:
        super().__init__(configuration, project_root)
        self.native_contrast_enabled = native_contrast_enabled
        self.native_preprocessing_seconds = 0.0

    def preprocess_image(self, image: ImageInput) -> tuple[torch.Tensor, int]:
        started = time.perf_counter()
        try:
            if self.native_contrast_enabled:
                return super().preprocess_image(image)

            image_path = Path(image)
            try:
                with Image.open(image_path) as source:
                    grayscale = source.convert("L")
            except (OSError, ValueError) as error:
                raise OSError(
                    f"Unable to read CRNN input image {image_path}: {error}"
                ) from error

            original_width, original_height = grayscale.size
            if original_width <= 0 or original_height <= 0:
                raise ValueError(f"Invalid image dimensions for {image_path}")
            content_width = int(
                round(original_width * self.input_height / original_height)
            )
            if content_width < 1:
                raise ValueError(f"Resized width is invalid for {image_path}")
            resized = grayscale.resize(
                (content_width, self.input_height),
                Image.Resampling.BILINEAR,
            )
            image_tensor = torch.tensor(
                np.array(resized),
                dtype=torch.float32,
                device=self.device,
            ) / 255.0
            image_tensor = (image_tensor - 0.5) / 0.5
            return image_tensor.unsqueeze(0), content_width
        finally:
            self.native_preprocessing_seconds += time.perf_counter() - started

    def reset_native_preprocessing_timer(self) -> None:
        self.native_preprocessing_seconds = 0.0
