"""Shared interface for OCR model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeAlias

ImageInput: TypeAlias = str | Path


class BaseOCRAdapter(ABC):
    """Model-independent prediction contract used by evaluation code."""

    @abstractmethod
    def predict(self, images: list[ImageInput]) -> list[str]:
        """Return one decoded transcription per input image, in input order."""
