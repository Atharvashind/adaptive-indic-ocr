"""Registry for independently executable preprocessing operations."""

from __future__ import annotations

from src.preprocessing.operation import PreprocessingOperation
from src.preprocessing.operations.contrast import (
    ClaheOperation,
    HistogramEqualizationOperation,
)
from src.preprocessing.operations.denoising import (
    BilateralFilterOperation,
    GaussianFilterOperation,
    MedianFilterOperation,
)
from src.preprocessing.operations.geometry import (
    AspectRatioPaddingOperation,
    DeskewOperation,
)
from src.preprocessing.operations.sharpening import UnsharpMaskOperation
from src.preprocessing.operations.thresholding import (
    AdaptiveGaussianThresholdOperation,
    AdaptiveMeanThresholdOperation,
    OtsuThresholdOperation,
)


class PreprocessingRegistry:
    """Store operation instances under unique, stable names."""

    def __init__(self) -> None:
        self._operations: dict[str, PreprocessingOperation] = {}

    def register(self, operation: PreprocessingOperation) -> None:
        """Register an operation, rejecting duplicate or empty names."""
        name = str(getattr(operation, "name", "")).strip()
        if not name:
            raise ValueError("preprocessing operation must have a non-empty name")
        if name in self._operations:
            raise ValueError(f"preprocessing operation {name!r} is already registered")
        self._operations[name] = operation

    def get(self, name: str) -> PreprocessingOperation:
        """Get an operation by its exact registered name."""
        try:
            return self._operations[name]
        except KeyError as error:
            raise KeyError(
                f"Unknown preprocessing operation {name!r}; available: "
                f"{', '.join(self.names())}"
            ) from error

    def names(self) -> tuple[str, ...]:
        """Return registered operation names in insertion order."""
        return tuple(self._operations)


def create_default_registry() -> PreprocessingRegistry:
    """Create a registry containing all Phase 2 standalone operations."""
    registry = PreprocessingRegistry()
    for operation in (
        ClaheOperation(),
        HistogramEqualizationOperation(),
        GaussianFilterOperation(),
        MedianFilterOperation(),
        BilateralFilterOperation(),
        OtsuThresholdOperation(),
        AdaptiveGaussianThresholdOperation(),
        AdaptiveMeanThresholdOperation(),
        UnsharpMaskOperation(),
        DeskewOperation(),
        AspectRatioPaddingOperation(),
    ):
        registry.register(operation)
    return registry
