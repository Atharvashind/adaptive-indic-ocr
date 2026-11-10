"""Serializable decision records for rule-based preprocessing selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SelectionDecision:
    """One reproducible recommendation; no image transformation is executed."""

    image_identifier: str
    operating_mode: str
    domain_metadata: dict[str, Any]
    measured_quality_values: dict[str, Any]
    selected_operation: str
    operation_parameters: dict[str, Any]
    rejected_candidates: list[dict[str, Any]]
    rule_evaluations: list[dict[str, Any]]
    reasons: list[str]
    confidence: float
    uncertainty: list[str]
    rule_version: str
    configuration_version: str
    evaluation_time_ms: float
    output_path: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible decision record."""
        return asdict(self)
