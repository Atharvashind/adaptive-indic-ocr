"""Tests for Phase 3 single-operation adaptive recommendations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from src.preprocessing.adaptive_selector import AdaptivePreprocessorSelector
from src.run_adaptive_selection import main
from src.quality.quality_analyzer import QualityAnalyzer


@pytest.fixture
def selector() -> AdaptivePreprocessorSelector:
    config_path = Path("configs/preprocessing.yaml")
    configuration = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return AdaptivePreprocessorSelector(
        configuration,
        operation_configuration=configuration,
    )


def _profile(
    *,
    contrast: float = 50.0,
    blur: float = 200.0,
    noise: float = 0.0,
    noise_confidence: float = 0.25,
    skew: float | None = None,
    skew_confidence: float = 0.0,
) -> dict:
    return {
        "valid": True,
        "aspect_ratio": 4.0,
        "measurements": {
            "grayscale_standard_deviation": {
                "value": contrast,
                "confidence": 0.7,
            },
            "brightness": {"value": 128.0, "confidence": 1.0},
            "blur": {"value": blur, "confidence": 0.55},
            "noise": {"value": noise, "confidence": noise_confidence},
            "illumination": {"value": 0.0, "confidence": 0.5},
            "skew_angle_degrees": {
                "value": skew,
                "confidence": skew_confidence,
            },
        },
        "warnings": [],
    }


def _image() -> np.ndarray:
    return np.tile(np.arange(128, dtype=np.uint8), (32, 1))


def test_good_quality_image_keeps_original(selector) -> None:
    image = _image()
    decision = selector.evaluate(image, image_identifier="good.png")

    assert decision.selected_operation == "keep_original"
    assert decision.operation_parameters == {}
    assert decision.confidence > 0


def test_low_contrast_considers_clahe(selector) -> None:
    decision = selector.evaluate(
        _image(),
        image_identifier="low-contrast.png",
        quality_profile=_profile(contrast=12.0),
    )

    assert decision.selected_operation == "clahe"
    assert decision.operation_parameters == {"clip_limit": 2.0, "tile_grid_size": [8, 8]}
    assert any("low_contrast" in reason for reason in decision.reasons)


def test_noisy_image_considers_denoising(selector) -> None:
    decision = selector.evaluate(
        _image(),
        image_identifier="noisy.png",
        quality_profile=_profile(noise=22.0),
    )

    assert decision.selected_operation == "median_filter"
    assert any(
        item["operation"] == "bilateral_filter"
        for item in decision.rejected_candidates
    )
    assert decision.uncertainty


def test_blurry_image_does_not_select_gaussian_smoothing(selector) -> None:
    decision = selector.evaluate(
        _image(),
        image_identifier="blurred.png",
        quality_profile=_profile(blur=12.0),
    )

    assert decision.selected_operation == "keep_original"
    assert any(
        item["operation"] == "gaussian_filter"
        for item in decision.rejected_candidates
    )


def test_high_confidence_skew_can_select_deskew(selector) -> None:
    decision = selector.evaluate(
        _image(),
        image_identifier="skewed.png",
        quality_profile=_profile(skew=4.2, skew_confidence=0.9),
    )

    assert decision.selected_operation == "deskew"


def test_low_confidence_skew_does_not_select_deskew(selector) -> None:
    decision = selector.evaluate(
        _image(),
        image_identifier="uncertain-skew.png",
        quality_profile=_profile(skew=4.2, skew_confidence=0.2),
    )

    assert decision.selected_operation == "keep_original"
    assert any(item["operation"] == "deskew" for item in decision.rejected_candidates)


def test_thresholding_is_never_selected_automatically(selector) -> None:
    decision = selector.evaluate(
        _image(),
        image_identifier="threshold-risk.png",
        quality_profile=_profile(contrast=10.0, noise=25.0),
    )

    assert decision.selected_operation not in {
        "otsu_threshold",
        "adaptive_gaussian_threshold",
        "adaptive_mean_threshold",
    }
    threshold_rejections = {
        item["operation"]
        for item in decision.rejected_candidates
        if item["rule"] == "thresholding_safety"
    }
    assert threshold_rejections == {
        "otsu_threshold",
        "adaptive_gaussian_threshold",
        "adaptive_mean_threshold",
    }


def test_user_guided_mode_uses_domain_context_but_measures_quality(selector) -> None:
    image = _image()
    profile = _profile(contrast=12.0, noise=22.0)
    handwritten = selector.evaluate(
        image,
        image_identifier="sample.png",
        mode="user_guided",
        domain="handwritten",
        metadata={"source_domain": "scene_text"},
        quality_profile=profile,
    )
    scene_text = selector.evaluate(
        image,
        image_identifier="sample.png",
        mode="user_guided",
        domain="scene_text",
        quality_profile=profile,
    )

    assert handwritten.selected_operation == "bilateral_filter"
    assert scene_text.selected_operation == "clahe"
    assert handwritten.domain_metadata["selected_domain"] == "handwritten"
    assert handwritten.domain_metadata["source_domain"] == "scene_text"
    assert handwritten.measured_quality_values == scene_text.measured_quality_values


def test_missing_quality_measurements_produce_uncertain_safe_decision(selector) -> None:
    decision = selector.evaluate(
        _image(),
        image_identifier="missing-profile.png",
        quality_profile={"valid": True, "measurements": {}},
    )

    assert decision.selected_operation == "keep_original"
    assert decision.confidence < 0.5
    assert decision.uncertainty


def test_invalid_quality_profile_never_recommends_an_operation(selector) -> None:
    decision = selector.evaluate(
        _image(),
        image_identifier="invalid.png",
        quality_profile={
            "valid": False,
            "measurements": {
                "grayscale_standard_deviation": {"value": 0.0, "confidence": 1.0}
            },
            "warnings": ["decode failed"],
        },
    )

    assert decision.selected_operation == "keep_original"
    assert decision.confidence == 0.0
    assert decision.uncertainty


def test_selection_does_not_mutate_original_and_decision_is_reproducible(selector) -> None:
    image = _image()
    original = image.copy()
    first = selector.evaluate(image, image_identifier="stable.png")
    second = selector.evaluate(image, image_identifier="stable.png")

    assert np.array_equal(image, original)
    assert first.to_dict() == second.to_dict() or (
        first.selected_operation == second.selected_operation
        and first.operation_parameters == second.operation_parameters
        and first.rule_evaluations == second.rule_evaluations
    )
    assert first.evaluation_time_ms >= 0
    assert first.output_path is None


def test_quality_analyzer_reuses_shared_metrics_for_rgb_and_bgr_arrays() -> None:
    grayscale = np.tile(np.arange(32, dtype=np.uint8), (24, 1))
    rgb = np.dstack((grayscale, grayscale // 2, np.flipud(grayscale)))
    bgr = rgb[:, :, ::-1].copy()
    original_rgb = rgb.copy()

    analyzer = QualityAnalyzer()
    rgb_profile = analyzer.analyze_array(
        rgb, image_identifier="color.png", color_order="RGB"
    )
    bgr_profile = analyzer.analyze_array(
        bgr, image_identifier="color.png", color_order="BGR"
    )

    assert rgb_profile["width"] == 32
    assert rgb_profile["height"] == 24
    assert rgb_profile["source_mode"] == "RGB"
    assert bgr_profile["source_mode"] == "BGR"
    assert rgb_profile["measurements"] == bgr_profile["measurements"]
    assert np.array_equal(rgb, original_rgb)


def test_automatic_mode_reads_only_metadata_domain(selector) -> None:
    decision = selector.evaluate(
        _image(),
        image_identifier="domain.png",
        metadata={"source_domain": "scene_text"},
        quality_profile=_profile(contrast=12.0),
    )

    assert decision.domain_metadata["selected_domain"] == "scene_text"
    assert decision.domain_metadata["domain_source"] == "metadata"
    assert decision.selected_operation == "clahe"


def test_user_guided_mode_requires_a_domain(selector) -> None:
    with pytest.raises(ValueError, match="requires a supported domain"):
        selector.evaluate(_image(), image_identifier="missing-domain.png", mode="user_guided")


def test_cli_emits_json_without_overwriting_image(tmp_path, capsys) -> None:
    image_path = tmp_path / "input.png"
    output_path = tmp_path / "decision.json"
    Image.fromarray(_image()).save(image_path)
    original_bytes = image_path.read_bytes()

    status = main(
        [
            "--image",
            str(image_path),
            "--mode",
            "user_guided",
            "--domain",
            "scene_text",
            "--output-json",
            str(output_path),
        ]
    )

    assert status == 0
    stdout_decision = json.loads(capsys.readouterr().out)
    file_decision = json.loads(output_path.read_text(encoding="utf-8"))
    assert stdout_decision == file_decision
    assert file_decision["operating_mode"] == "user_guided"
    assert file_decision["domain_metadata"]["selected_domain"] == "scene_text"
    assert image_path.read_bytes() == original_bytes


def test_cli_refuses_to_overwrite_an_existing_decision_file(tmp_path) -> None:
    image_path = tmp_path / "input.png"
    output_path = tmp_path / "decision.json"
    Image.fromarray(_image()).save(image_path)
    output_path.write_text("keep me", encoding="utf-8")

    status = main(
        [
            "--image",
            str(image_path),
            "--output-json",
            str(output_path),
        ]
    )

    assert status == 2
    assert output_path.read_text(encoding="utf-8") == "keep me"
