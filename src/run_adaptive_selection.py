"""Evaluate one image and emit a rule-based preprocessing recommendation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from src.preprocessing.adaptive_selector import AdaptivePreprocessorSelector
from src.profiling.domain_profiler import SUPPORTED_DOMAINS
from src.quality.quality_analyzer import QualityAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _load_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode in {"1", "L", "I", "F"}:
            return np.asarray(image.convert("L"), dtype=np.uint8).copy()
        if image.mode == "RGBA":
            return np.asarray(image, dtype=np.uint8).copy()
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        document = yaml.safe_load(source)
    if not isinstance(document, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one image and recommend at most one independent "
            "preprocessing operation. Does not apply operations or modify the image."
        )
    )
    parser.add_argument("--image", required=True, help="Image file to assess")
    parser.add_argument(
        "--mode",
        choices=("automatic", "user_guided"),
        default="automatic",
    )
    parser.add_argument(
        "--domain",
        choices=sorted(SUPPORTED_DOMAINS),
        help="Required in user_guided mode; automatic mode reads domain metadata only",
    )
    parser.add_argument(
        "--metadata-json",
        help="Optional JSON object containing domain/source metadata for automatic mode",
    )
    parser.add_argument("--image-id", help="Identifier recorded in the JSON decision")
    parser.add_argument(
        "--preprocessing-config",
        default="configs/preprocessing.yaml",
    )
    parser.add_argument(
        "--dataset-config",
        default="configs/datasets.yaml",
        help="Provides the configured Phase 1 quality-measurement thresholds",
    )
    parser.add_argument(
        "--output-json",
        help="Optional JSON destination separate from the input image",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        image_path = _resolve_path(args.image, PROJECT_ROOT)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        if args.mode == "user_guided" and not args.domain:
            raise ValueError("--domain is required when --mode user_guided is selected")
        if args.mode == "automatic" and args.domain:
            raise ValueError(
                "Use --mode user_guided to specify --domain; automatic mode reads "
                "domain information from --metadata-json."
            )

        preprocessing_document = _read_yaml(
            _resolve_path(args.preprocessing_config, PROJECT_ROOT)
        )
        dataset_document = _read_yaml(
            _resolve_path(args.dataset_config, PROJECT_ROOT)
        )
        quality_config = dataset_document.get("quality", {})
        if not isinstance(quality_config, dict):
            raise ValueError("dataset config 'quality' section must be a mapping")

        image = _load_array(image_path)
        metadata: dict[str, Any] = {"filename": image_path.name}
        if args.metadata_json:
            metadata_document = json.loads(
                _resolve_path(args.metadata_json, PROJECT_ROOT).read_text(
                    encoding="utf-8"
                )
            )
            if not isinstance(metadata_document, dict):
                raise ValueError("--metadata-json must contain a JSON object")
            metadata.update(metadata_document)
        selector = AdaptivePreprocessorSelector(
            preprocessing_document,
            quality_analyzer=QualityAnalyzer(quality_config),
            operation_configuration=preprocessing_document,
        )
        decision = selector.evaluate(
            image,
            image_identifier=(
                args.image_id
                or metadata.get("source_image_id")
                or image_path.name
            ),
            mode=args.mode,
            domain=args.domain,
            metadata=metadata,
        )
        serialized = json.dumps(
            decision.to_dict(),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        if args.output_json:
            output_path = _resolve_path(args.output_json, PROJECT_ROOT)
            if output_path.resolve() == image_path.resolve():
                raise ValueError("Decision JSON output must not overwrite the input image")
            if output_path.exists():
                raise FileExistsError(
                    f"Decision JSON output already exists; choose a new path: {output_path}"
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(serialized + "\n", encoding="utf-8")
        print(serialized)
        return 0
    except (OSError, ValueError, yaml.YAMLError) as error:
        logger.error("%s", error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
