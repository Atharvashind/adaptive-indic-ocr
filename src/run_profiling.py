"""Command-line entry point for dataset profiling."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from src.ingestion.annotation_parser import UnsupportedAnnotationFormatError
from src.ingestion.base_dataset import DatasetManifest
from src.ingestion.dataset_loader import DatasetConfigurationError, load_dataset
from src.profiling.dataset_profiler import profile_dataset
from src.quality.quality_analyzer import QualityAnalyzer
from src.reporting.dataset_report import generate_reports

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_DOMAINS = (
    "handwritten",
    "printed_document",
    "scene_text",
    "mixed",
    "unknown",
)


def _resolve_from_project(path_value: str, project_root: Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else project_root / path


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML configuration document and validate its top-level mapping."""
    try:
        with path.open("r", encoding="utf-8") as source:
            config = yaml.safe_load(source)
    except OSError as error:
        raise DatasetConfigurationError(f"Unable to read config {path}: {error}") from error
    except yaml.YAMLError as error:
        raise DatasetConfigurationError(f"Invalid YAML in {path}: {error}") from error
    if not isinstance(config, dict):
        raise DatasetConfigurationError(f"Configuration must be a YAML mapping: {path}")
    return config


def _dataset_mapping(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = config.get("datasets")
    if isinstance(entries, dict):
        return {
            str(name): value
            for name, value in entries.items()
            if isinstance(value, dict)
        }
    if isinstance(entries, list):
        return {
            str(value["name"]): value
            for value in entries
            if isinstance(value, dict) and value.get("name")
        }
    raise DatasetConfigurationError("'datasets' must be a mapping or list in the YAML config")


def _image_split(image: Any) -> str:
    for annotation in getattr(image, "annotations", []):
        split = (annotation.metadata or {}).get("split")
        if split:
            return str(split).strip().lower()
    return "unknown"


def _image_language(image: Any) -> str:
    values = {
        str((annotation.language or "")).strip().lower()
        for annotation in getattr(image, "annotations", [])
        if annotation.language
    }
    return next(iter(sorted(values))) if values else "unknown"


def _allocate_sample_counts(
    total: int, groups: dict[str, list[Any]], ensure_each: bool
) -> dict[str, int]:
    allocations = {name: 0 for name in groups}
    if total <= 0 or not groups:
        return allocations
    names = sorted(groups)
    if ensure_each and total >= len(names):
        for name in names:
            allocations[name] = 1
    remaining = min(total, sum(len(group) for group in groups.values())) - sum(
        allocations.values()
    )
    while remaining > 0:
        available = {
            name: len(groups[name]) - allocations[name]
            for name in names
            if len(groups[name]) > allocations[name]
        }
        if not available:
            break
        weight_total = sum(available.values())
        quotas = {
            name: remaining * count / weight_total
            for name, count in available.items()
        }
        additions = {
            name: min(available[name], int(quotas[name]))
            for name in available
        }
        allocated = sum(additions.values())
        for name, count in additions.items():
            allocations[name] += count
        remaining -= allocated
        if remaining <= 0:
            break
        ranked = sorted(
            (
                name
                for name in available
                if allocations[name] < len(groups[name])
            ),
            key=lambda name: (-(quotas[name] - int(quotas[name])), name),
        )
        for name in ranked:
            if remaining == 0:
                break
            allocations[name] += 1
            remaining -= 1
    return allocations


def _apply_sample_limit(manifest: DatasetManifest, limit: int, seed: int) -> None:
    if limit <= 0:
        raise DatasetConfigurationError("--sample-limit must be greater than zero")
    manifest.sample_limit = limit
    manifest.sample_seed = seed
    if len(manifest.images) <= limit:
        return

    rng = random.Random(seed)
    split_groups: dict[str, list[Any]] = {}
    for image in manifest.images:
        split_groups.setdefault(_image_split(image), []).append(image)

    split_counts = _allocate_sample_counts(
        limit, split_groups, ensure_each=limit >= len(split_groups)
    )
    selected: list[Any] = []
    for split_name in sorted(split_groups):
        bucket = split_groups[split_name]
        split_sample_count = split_counts[split_name]
        language_groups: dict[str, list[Any]] = {}
        for image in bucket:
            language_groups.setdefault(_image_language(image), []).append(image)
        language_counts = _allocate_sample_counts(
            split_sample_count,
            language_groups,
            ensure_each=split_sample_count >= len(language_groups),
        )
        for language in sorted(language_groups):
            language_bucket = language_groups[language]
            count = language_counts[language]
            selected.extend(rng.sample(language_bucket, count))

    manifest.images = sorted(selected, key=lambda image: image.relative_path.casefold())
    manifest.annotations = [
        annotation
        for image in manifest.images
        for annotation in image.annotations
    ]


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Profile dataset composition, annotations, and image quality."
    )
    parser.add_argument("--dataset", required=True, help="Dataset key in datasets.yaml")
    parser.add_argument(
        "--dataset-config",
        default="configs/datasets.yaml",
        help="Dataset YAML path (default: configs/datasets.yaml)",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        help="Profile at most this many images using a reproducible random sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Sampling seed when --sample-limit is used (default: 0)",
    )
    parser.add_argument(
        "--domain",
        choices=SUPPORTED_DOMAINS,
        help="User-specified domain label; image quality is still assessed independently",
    )
    parser.add_argument(
        "--output-root",
        default="experiments/profiling",
        help="Report root directory (default: experiments/profiling)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load a configured dataset, profile it, and save a new report run."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    try:
        config_path = _resolve_from_project(args.dataset_config, PROJECT_ROOT)
        config = load_config(config_path)
        datasets = _dataset_mapping(config)
        if args.dataset not in datasets:
            raise DatasetConfigurationError(
                f"Dataset {args.dataset!r} is not configured in {config_path}. "
                f"Available dataset keys: {', '.join(sorted(datasets)) or '<none>'}"
            )
        dataset_config = datasets[args.dataset]
        if not dataset_config.get("enabled", False):
            raise DatasetConfigurationError(
                f"Dataset {args.dataset!r} is disabled. Verify its image path and "
                "annotation format in configs/datasets.yaml before enabling it."
            )

        manifest = load_dataset(args.dataset, dataset_config, PROJECT_ROOT)
        if args.sample_limit is not None:
            _apply_sample_limit(manifest, args.sample_limit, args.seed)
        if not manifest.images:
            logging.warning("No supported images found under %s", manifest.image_root)

        analyzer = QualityAnalyzer(config.get("quality", {}))
        quality_results: dict[str, dict[str, Any]] = {}
        for image in tqdm(manifest.images, desc="Assessing images", unit="image"):
            result = analyzer.analyze_safely(image.path)
            quality_results[image.relative_path] = result

        profile = profile_dataset(
            manifest,
            quality_results,
            configured_labels=dataset_config.get("domain_labels", []),
            user_domain=args.domain,
        )
        output_root = _resolve_from_project(args.output_root, PROJECT_ROOT)
        report_directory = generate_reports(
            profile, manifest, quality_results, output_root
        )
        logging.info("Profile complete: %s", report_directory)
        return 0
    except (
        DatasetConfigurationError,
        UnsupportedAnnotationFormatError,
        OSError,
        ValueError,
    ) as error:
        logging.error("%s", error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
