"""Configurable, independently callable image preprocessing operations."""

from src.preprocessing.adaptive_selector import AdaptivePreprocessorSelector
from src.preprocessing.decision_models import SelectionDecision
from src.preprocessing.operation import OperationResult, PreprocessingOperation
from src.preprocessing.registry import (
    PreprocessingRegistry,
    create_default_registry,
)

__all__ = [
    "OperationResult",
    "PreprocessingOperation",
    "PreprocessingRegistry",
    "AdaptivePreprocessorSelector",
    "SelectionDecision",
    "create_default_registry",
]
