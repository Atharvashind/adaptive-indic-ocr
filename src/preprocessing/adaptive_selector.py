"""Single-operation rule-based adaptive preprocessing recommendations."""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np

from src.profiling.domain_profiler import SUPPORTED_DOMAINS
from src.preprocessing.decision_models import SelectionDecision
from src.preprocessing.rule_engine import PreprocessingRuleEngine
from src.quality.quality_analyzer import QualityAnalyzer


class AdaptivePreprocessorSelector:
    """Recommend one operation or keep the source image unchanged."""

    RULE_VERSION = "1.0"

    def __init__(
        self,
        configuration: dict[str, Any],
        quality_analyzer: QualityAnalyzer | None = None,
        operation_configuration: dict[str, Any] | None = None,
    ) -> None:
        self.configuration = configuration
        self.quality_analyzer = quality_analyzer or QualityAnalyzer()
        self.operation_configuration = operation_configuration or {}

    def evaluate(
        self,
        image: np.ndarray,
        *,
        image_identifier: str,
        mode: str = "automatic",
        domain: str | None = None,
        metadata: dict[str, Any] | None = None,
        quality_profile: dict[str, Any] | None = None,
        color_order: str = "RGB",
    ) -> SelectionDecision:
        """Evaluate quality rules and return a recommendation without applying it."""
        started = time.perf_counter()
        if mode not in {"automatic", "user_guided"}:
            raise ValueError("mode must be 'automatic' or 'user_guided'")
        if mode == "automatic" and domain is not None:
            raise ValueError(
                "automatic mode reads domain metadata; use user_guided mode "
                "to supply a domain directly"
            )
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a mapping")
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.size == 0
            or image.ndim not in {2, 3}
            or (image.ndim == 3 and image.shape[2] not in {1, 3, 4})
        ):
            raise ValueError("image must be non-empty uint8 grayscale or colour data")
        if mode == "user_guided":
            if domain not in SUPPORTED_DOMAINS:
                raise ValueError(
                    "user_guided mode requires a supported domain: "
                    + ", ".join(sorted(SUPPORTED_DOMAINS))
                )
            selected_domain = domain
        else:
            selected_domain = self._metadata_domain(metadata)

        if quality_profile is None:
            quality_profile = self.quality_analyzer.analyze_array(
                image,
                image_identifier=image_identifier,
                color_order=color_order,
            )
        else:
            quality_profile = copy.deepcopy(quality_profile)
        if not isinstance(quality_profile, dict):
            raise ValueError("quality_profile must be a mapping")

        selection_config = self.configuration.get(
            "adaptive_selection", self.configuration
        )
        operations = self.operation_configuration.get(
            "preprocessing", self.operation_configuration
        )
        engine = PreprocessingRuleEngine(selection_config, operations)
        if quality_profile.get("valid") is False:
            candidates, rejected, rule_evaluations = [], [], []
            uncertainty = ["The supplied quality profile marks the image invalid."]
        else:
            candidates, rejected, rule_evaluations, uncertainty = engine.evaluate(
                quality_profile
            )
        selected, lower_priority = engine.select(candidates, selected_domain)
        rejected.extend(lower_priority)
        selected_operation = selected.operation_name if selected else "keep_original"
        reasons = (
            [selected.reason]
            if selected
            else ["No configured candidate rule justified changing the source image."]
        )
        if selected is None and not quality_profile.get("measurements"):
            uncertainty.append(
                "Safe fallback: keep_original because no measurements are available."
            )
        confidence = (
            selected.confidence
            if selected
            else 0.75
            if quality_profile.get("measurements") and not uncertainty
            else 0.25
        )
        warnings = list(quality_profile.get("warnings", []))
        if quality_profile.get("valid") is False:
            confidence = 0.0
            selected_operation = "keep_original"
            selected = None
            reasons = ["Invalid image quality profile; the source was not changed."]
        if selected:
            reasons.extend(
                evaluation
                for evaluation in rule_evaluations
                if evaluation.startswith(selected.rule_name + "=")
            )
        return SelectionDecision(
            image_identifier=image_identifier,
            operating_mode=mode,
            domain_metadata={
                "selected_domain": selected_domain,
                "user_specified_domain": domain if mode == "user_guided" else None,
                "source_domain": (metadata or {}).get("source_domain"),
                "metadata_domain": (metadata or {}).get("domain"),
                "domain_labels": self._metadata_labels(metadata),
                "domain_source": (
                    "user"
                    if mode == "user_guided"
                    else "metadata"
                    if selected_domain != "unknown"
                    else "unavailable"
                ),
            },
            measured_quality_values=quality_profile,
            selected_operation=selected_operation,
            operation_parameters=(
                engine.parameters_for(selected_operation)
                if selected is not None
                else {}
            ),
            rejected_candidates=rejected,
            rule_evaluations=[
                {"rule": item.split("=", 1)[0], "result": item}
                for item in rule_evaluations
            ],
            reasons=reasons,
            confidence=confidence,
            uncertainty=sorted(set(uncertainty)),
            rule_version=self.RULE_VERSION,
            configuration_version=str(
                selection_config.get("version", "unversioned")
            ),
            evaluation_time_ms=(time.perf_counter() - started) * 1000,
            output_path=None,
            warnings=warnings,
        )

    @staticmethod
    def _metadata_domain(metadata: dict[str, Any] | None) -> str:
        if not isinstance(metadata, dict):
            return "unknown"
        value = metadata.get("domain") or metadata.get("source_domain")
        if isinstance(value, str) and value in SUPPORTED_DOMAINS:
            return str(value)
        labels = metadata.get("domain_labels", [])
        if isinstance(labels, (list, tuple)):
            for label in labels:
                if isinstance(label, str) and label in SUPPORTED_DOMAINS:
                    return str(label)
        return "unknown"

    @staticmethod
    def _metadata_labels(metadata: dict[str, Any] | None) -> list[str]:
        labels = (metadata or {}).get("domain_labels", [])
        if isinstance(labels, str):
            return [labels]
        if isinstance(labels, (list, tuple)):
            return [label for label in labels if isinstance(label, str)]
        return []
