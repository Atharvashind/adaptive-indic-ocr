"""Run reproducible batch validation of adaptive-selector recommendations."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from src.ingestion.dataset_loader import DatasetConfigurationError, load_dataset
from src.preprocessing.adaptive_selector import AdaptivePreprocessorSelector
from src.preprocessing.batch_validation import (
    BatchAdaptiveValidation,
    create_run_directory,
    sample_manifest_by_source,
)
from src.quality.quality_analyzer import QualityAnalyzer
from src.run_profiling import _dataset_mapping, _resolve_from_project

PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        document = yaml.safe_load(source)
    if not isinstance(document, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate adaptive selector recommendations on a reproducible sample; "
            "does not execute preprocessing operations."
        )
    )
    parser.add_argument("--dataset", default="synth6k", help="Dataset key")
    parser.add_argument("--sample-limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--user-domain", default="scene_text")
    parser.add_argument("--dataset-config", default="configs/datasets.yaml")
    parser.add_argument("--preprocessing-config", default="configs/preprocessing.yaml")
    parser.add_argument(
        "--output-root",
        default="experiments/adaptive_selection",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        dataset_config_path = _resolve_from_project(
            args.dataset_config, PROJECT_ROOT
        )
        preprocessing_config_path = _resolve_from_project(
            args.preprocessing_config, PROJECT_ROOT
        )
        dataset_document = _read_yaml(dataset_config_path)
        preprocessing_document = _read_yaml(preprocessing_config_path)
        datasets = _dataset_mapping(dataset_document)
        if args.dataset not in datasets:
            raise DatasetConfigurationError(
                f"Dataset {args.dataset!r} is not configured in {dataset_config_path}"
            )
        config = datasets[args.dataset]
        if not config.get("enabled", False):
            raise DatasetConfigurationError(
                f"Dataset {args.dataset!r} is disabled in {dataset_config_path}"
            )
        manifest = load_dataset(args.dataset, config, PROJECT_ROOT)
        population_count = len(manifest.images)
        sampled_manifest = sample_manifest_by_source(
            manifest, args.sample_limit, args.seed
        )
        sampled_manifest.discovered_image_count = population_count
        if not sampled_manifest.images:
            raise DatasetConfigurationError(
                f"No supported images found under {manifest.image_root}"
            )

        quality_config = dataset_document.get("quality", {})
        if not isinstance(quality_config, dict):
            raise ValueError("dataset configuration 'quality' section must be a mapping")
        selector = AdaptivePreprocessorSelector(
            preprocessing_document,
            quality_analyzer=QualityAnalyzer(quality_config),
            operation_configuration=preprocessing_document,
        )
        validation = BatchAdaptiveValidation(
            selector,
            QualityAnalyzer(quality_config),
            user_domain=args.user_domain,
        )
        output_root = _resolve_from_project(args.output_root, PROJECT_ROOT)
        run_directory = create_run_directory(output_root, args.dataset)
        summary = validation.run(sampled_manifest, run_directory)
        logger.info(
            "Batch selector validation complete: %s (%d images, %d quality failures)",
            run_directory,
            summary["sample_size"],
            summary["quality_failures"],
        )
        print(run_directory)
        return 0
    except (
        DatasetConfigurationError,
        OSError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        logger.error("%s", error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
