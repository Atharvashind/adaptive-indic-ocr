"""Explainable, configurable rules for single-operation recommendations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Candidate:
    operation_name: str
    confidence: float
    reason: str
    rule_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation_name,
            "confidence": self.confidence,
            "reason": self.reason,
            "rule": self.rule_name,
        }


class PreprocessingRuleEngine:
    """Evaluate measurable quality signals without chaining operations."""

    def __init__(
        self,
        configuration: dict[str, Any],
        operation_parameters: dict[str, Any],
    ) -> None:
        self.configuration = configuration
        self.operation_parameters = operation_parameters
        self.thresholds = configuration.get("thresholds", {})
        self.domain_preferences = configuration.get("domain_preferences", {})
        self.default_priority = configuration.get(
            "default_candidate_priority",
            ["clahe", "median_filter", "bilateral_filter", "deskew", "unsharp_mask"],
        )

    def evaluate(
        self, quality: dict[str, Any]
    ) -> tuple[list[Candidate], list[dict[str, Any]], list[str], list[str]]:
        measurements = quality.get("measurements", {})
        candidates: list[Candidate] = []
        rejected: list[dict[str, Any]] = []
        evaluations: list[str] = []
        uncertainty: list[str] = []

        contrast = measurements.get("grayscale_standard_deviation", {})
        contrast_value = contrast.get("value")
        if self._finite_number(contrast_value):
            low_contrast = float(contrast_value) < float(
                self.thresholds.get("low_contrast_below", 30.0)
            )
            evaluations.append(
                f"low_contrast={'triggered' if low_contrast else 'not triggered'} "
                f"(grayscale standard deviation={float(contrast_value):.3f})"
            )
            if low_contrast:
                self._add_candidate(
                    candidates,
                    "clahe",
                    contrast.get("confidence"),
                    "Global grayscale standard deviation is below the configured "
                    "low-contrast threshold.",
                    "low_contrast",
                )
        else:
            evaluations.append("low_contrast=unavailable")
            uncertainty.append("Contrast measurement is unavailable.")

        noise = measurements.get("noise", {})
        noise_value = noise.get("value")
        noise_confidence = noise.get("confidence")
        if self._finite_number(noise_value):
            threshold = float(self.thresholds.get("noise_above", 10.0))
            minimum_confidence = float(
                self.thresholds.get("minimum_noise_confidence", 0.2)
            )
            triggered = (
                float(noise_value) >= threshold
                and self._finite_number(noise_confidence)
                and float(noise_confidence) >= minimum_confidence
            )
            evaluations.append(
                f"high_frequency_residual={'triggered' if triggered else 'not triggered'} "
                f"(value={float(noise_value):.3f}, confidence="
                f"{noise_confidence})"
            )
            if triggered:
                for operation in ("median_filter", "bilateral_filter"):
                    self._add_candidate(
                        candidates,
                        operation,
                        noise_confidence,
                        "The existing high-frequency residual exceeds its configured "
                        "threshold; this estimator can also respond to text strokes, "
                        "texture, and compression artifacts.",
                        "high_frequency_residual",
                    )
                uncertainty.append(
                    "Noise-related candidates are low-confidence because the residual "
                    "estimator also responds to strokes, texture, and compression."
                )
        else:
            evaluations.append("high_frequency_residual=unavailable")
            uncertainty.append("Noise measurement is unavailable.")

        blur = measurements.get("blur", {})
        blur_value = blur.get("value")
        if self._finite_number(blur_value):
            low_blur = float(blur_value) < float(
                self.thresholds.get("blur_below", 100.0)
            )
            evaluations.append(
                f"possible_blur={'triggered' if low_blur else 'not triggered'} "
                f"(variance of Laplacian={float(blur_value):.3f})"
            )
            if low_blur:
                rejected.append(
                    {
                        "operation": "gaussian_filter",
                        "reason": "Not selected: additional Gaussian smoothing is not "
                        "a justified response to a possible blur signal.",
                        "rule": "possible_blur",
                    }
                )
                if self.configuration.get("enable_unsharp_for_blur", False):
                    self._add_candidate(
                        candidates,
                        "unsharp_mask",
                        blur.get("confidence"),
                        "Optional sharpening candidate enabled by configuration; "
                        "possible blur alone does not establish suitability.",
                        "possible_blur",
                    )
                else:
                    rejected.append(
                        {
                            "operation": "unsharp_mask",
                            "reason": "Possible blur was detected, but sharpening is "
                            "disabled by the provisional rule configuration.",
                            "rule": "possible_blur",
                        }
                    )
        else:
            evaluations.append("possible_blur=unavailable")
            uncertainty.append("Blur measurement is unavailable.")

        illumination = measurements.get("illumination", {})
        illumination_value = illumination.get("value")
        if self._finite_number(illumination_value):
            uneven = float(illumination_value) >= float(
                self.thresholds.get("uneven_illumination_above", 18.0)
            )
            evaluations.append(
                f"uneven_illumination={'triggered' if uneven else 'not triggered'} "
                f"(spatial variation={float(illumination_value):.3f})"
            )
            if uneven:
                rejected.append(
                    {
                        "operation": "illumination_correction",
                        "reason": "Possible uneven illumination was measured, but no "
                        "validated illumination-correction operation is implemented.",
                        "rule": "uneven_illumination",
                    }
                )
        else:
            evaluations.append("uneven_illumination=unavailable")
            uncertainty.append("Illumination measurement is unavailable.")

        brightness = measurements.get("brightness", {})
        brightness_value = brightness.get("value")
        if self._finite_number(brightness_value):
            evaluations.append(
                f"brightness=measured ({float(brightness_value):.3f}); no "
                "brightness-correction operation is enabled"
            )
        else:
            evaluations.append("brightness=unavailable")
            uncertainty.append("Brightness measurement is unavailable.")

        aspect_ratio = quality.get("aspect_ratio")
        if self._finite_number(aspect_ratio):
            evaluations.append(
                f"aspect_ratio=measured ({float(aspect_ratio):.6f}); no "
                "aspect-ratio selection rule is configured"
            )
        else:
            evaluations.append("aspect_ratio=unavailable")
            uncertainty.append("Aspect-ratio measurement is unavailable.")

        skew = measurements.get("skew_angle_degrees", {})
        skew_value = skew.get("value")
        skew_confidence = skew.get("confidence")
        if (
            self._finite_number(skew_value)
            and self._finite_number(skew_confidence)
        ):
            minimum_angle = float(
                self.thresholds.get("skew_min_angle_degrees", 1.0)
            )
            minimum_confidence = float(
                self.thresholds.get("skew_min_confidence", 0.7)
            )
            angle_detected = abs(float(skew_value)) >= minimum_angle
            confidence_sufficient = float(skew_confidence) >= minimum_confidence
            triggered = angle_detected and confidence_sufficient
            evaluations.append(
                f"deskew={'triggered' if triggered else 'not triggered'} "
                f"(angle={float(skew_value):.3f}, confidence="
                f"{float(skew_confidence):.3f})"
            )
            if triggered:
                self._add_candidate(
                    candidates,
                    "deskew",
                    skew_confidence,
                    "A skew estimate exceeds the configured angle and confidence "
                    "thresholds.",
                    "deskew",
                )
            else:
                reason = (
                    "Skew angle is below the configured minimum."
                    if not angle_detected
                    else "Skew confidence is below the configured minimum."
                )
                rejected.append(
                    {"operation": "deskew", "reason": reason, "rule": "reliable_skew"}
                )
        else:
            evaluations.append("deskew=unavailable")
            uncertainty.append("Skew angle or confidence is unavailable.")
            rejected.append(
                {
                    "operation": "deskew",
                    "reason": "No sufficiently reliable skew measurement is available.",
                    "rule": "reliable_skew",
                }
            )

        for operation in (
            "otsu_threshold",
            "adaptive_gaussian_threshold",
            "adaptive_mean_threshold",
        ):
            rejected.append(
                {
                    "operation": operation,
                    "reason": "Thresholding is not automatically selected in this "
                    "initial version because visual validation showed speckling and "
                    "possible foreground merging.",
                    "rule": "thresholding_safety",
                }
            )

        if not measurements:
            uncertainty.append("No quality measurements were supplied.")
        return candidates, rejected, evaluations, uncertainty

    def select(
        self, candidates: list[Candidate], domain: str
    ) -> tuple[Candidate | None, list[dict[str, Any]]]:
        """Choose at most one candidate using configured domain priorities."""
        priorities = self.domain_preferences.get(domain, self.default_priority)
        rank = {name: index for index, name in enumerate(priorities)}
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                rank.get(candidate.operation_name, len(rank)),
                -candidate.confidence,
                candidate.operation_name,
            ),
        )
        if not ordered:
            return None, []
        selected = ordered[0]
        rejected = [
            {
                **candidate.to_dict(),
                "reason": "A different candidate had higher configured priority; "
                "only one primary operation is recommended.",
            }
            for candidate in ordered[1:]
        ]
        return selected, rejected

    def parameters_for(self, operation_name: str) -> dict[str, Any]:
        value = self.operation_parameters.get(operation_name, {})
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _finite_number(value: Any) -> bool:
        try:
            return value is not None and math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _add_candidate(
        candidates: list[Candidate],
        operation_name: str,
        confidence: Any,
        reason: str,
        rule_name: str,
    ) -> None:
        try:
            score = float(confidence)
        except (TypeError, ValueError):
            score = 0.0
        candidates.append(
            Candidate(
                operation_name=operation_name,
                confidence=max(0.0, min(1.0, score)),
                reason=reason,
                rule_name=rule_name,
            )
        )
