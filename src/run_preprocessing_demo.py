"""Run independent preprocessing operations on a deterministic dataset sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

from src.ingestion.dataset_loader import DatasetConfigurationError, load_dataset
from src.preprocessing import create_default_registry
from src.preprocessing.transformation_log import TransformationLog
from src.run_profiling import _apply_sample_limit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _new_run_directory(root: Path, dataset: str) -> Path:
    dataset_root = root / dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = dataset_root / f"run-{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = dataset_root / f"run-{timestamp}-{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_array(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        if source.mode in {"1", "L", "I", "F"}:
            return np.asarray(source.convert("L"), dtype=np.uint8).copy()
        if source.mode == "RGBA":
            return np.asarray(source, dtype=np.uint8).copy()
        return np.asarray(source.convert("RGB"), dtype=np.uint8).copy()


def _as_rgb(array: np.ndarray) -> Image.Image:
    if array.ndim == 2:
        return Image.fromarray(array).convert("RGB")
    if array.shape[2] == 1:
        return Image.fromarray(array[:, :, 0]).convert("RGB")
    if array.shape[2] == 4:
        rgba = Image.fromarray(array, mode="RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return Image.fromarray(array, mode="RGB")


def _save_comparison(
    original: np.ndarray,
    processed: np.ndarray,
    output_path: Path,
    operation_name: str,
) -> None:
    before = _as_rgb(original)
    after = _as_rgb(processed)
    label_height = 28
    gap = 12
    height = max(before.height, after.height) + label_height
    canvas = Image.new("RGB", (before.width + after.width + gap, height), "white")
    canvas.paste(before, (0, label_height))
    canvas.paste(after, (before.width + gap, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 6), "Original", fill="black")
    draw.text((before.width + gap + 4, 6), operation_name, fill="black")
    canvas.save(output_path)


def _annotation_metadata(
    image: Any, annotation_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    annotation_config = annotation_config or {}
    records = []
    for annotation in image.annotations:
        records.append(
            {
                "source_image_id": annotation.source_image_id,
                "filename": image.path.name,
                "granularity": annotation.granularity,
                "transcription": annotation.transcription,
                "language": annotation.language,
                "bbox": annotation.bbox,
                "polygon": annotation.polygon,
                "bbox_coordinate_space": annotation_config.get(
                    "bbox_coordinate_space"
                ),
                "bbox_format": annotation_config.get("bbox_format"),
                "metadata": annotation.metadata,
            }
        )
    return {
        "source_image_id": image.source_image_id,
        "filename": image.path.name,
        "image_path": image.relative_path,
        "annotations": records,
    }


def _serialize_result(result: Any) -> dict[str, Any]:
    return result.to_dict(include_metadata=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply each configured preprocessing operation independently to a "
            "deterministic dataset sample."
        )
    )
    parser.add_argument("--dataset", required=True, help="Dataset key in datasets.yaml")
    parser.add_argument("--sample-limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--dataset-config",
        default="configs/datasets.yaml",
        help="Dataset configuration YAML path",
    )
    parser.add_argument(
        "--preprocessing-config",
        default="configs/preprocessing.yaml",
        help="Preprocessing configuration YAML path",
    )
    parser.add_argument(
        "--output-root",
        default="experiments/preprocessing",
        help="Experiment output root",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        dataset_config_path = _resolve_path(args.dataset_config, PROJECT_ROOT)
        preprocessing_config_path = _resolve_path(args.preprocessing_config, PROJECT_ROOT)
        with dataset_config_path.open("r", encoding="utf-8") as source:
            dataset_document = yaml.safe_load(source)
        with preprocessing_config_path.open("r", encoding="utf-8") as source:
            preprocessing_document = yaml.safe_load(source)
        datasets = dataset_document.get("datasets", {})
        if args.dataset not in datasets:
            raise DatasetConfigurationError(
                f"Dataset {args.dataset!r} is not configured in {dataset_config_path}"
            )
        dataset_config = datasets[args.dataset]
        if not dataset_config.get("enabled", False):
            raise DatasetConfigurationError(f"Dataset {args.dataset!r} is disabled")
        operations_config = preprocessing_document.get("preprocessing")
        if not isinstance(operations_config, dict):
            raise ValueError("preprocessing config must contain a mapping named 'preprocessing'")

        manifest = load_dataset(args.dataset, dataset_config, PROJECT_ROOT)
        if args.sample_limit <= 0:
            raise ValueError("--sample-limit must be greater than zero")
        _apply_sample_limit(manifest, args.sample_limit, args.seed)
        if not manifest.images:
            raise ValueError(f"No supported images found under {manifest.image_root}")

        output_root = _resolve_path(args.output_root, PROJECT_ROOT)
        run_directory = _new_run_directory(output_root, args.dataset)
        registry = create_default_registry()
        unknown = set(operations_config) - set(registry.names())
        if unknown:
            raise ValueError(f"Unknown configured operations: {sorted(unknown)}")

        input_hashes = {image.relative_path: _sha256(image.path) for image in manifest.images}
        transformation_log = TransformationLog()
        per_image_records: list[dict[str, Any]] = []

        for sample_index, image in enumerate(manifest.images, start=1):
            source = _load_array(image.path)
            metadata = _annotation_metadata(
                image, dataset_config.get("annotation", {})
            )
            operation_records = []
            sample_directory = run_directory / f"sample-{sample_index:02d}"
            sample_directory.mkdir()
            for operation_name, operation_parameters in operations_config.items():
                if operation_parameters is None:
                    operation_parameters = {}
                if not isinstance(operation_parameters, dict):
                    raise ValueError(
                        f"Configuration for {operation_name!r} must be a mapping"
                    )
                result = registry.get(operation_name).execute(
                    source,
                    parameters=operation_parameters,
                    metadata=metadata,
                )
                record = _serialize_result(result)
                output_path: Path | None = None
                comparison_path: Path | None = None
                if result.success and result.processed_image is not None:
                    operation_directory = sample_directory / operation_name
                    operation_directory.mkdir()
                    output_path = operation_directory / "processed.png"
                    comparison_path = operation_directory / "comparison.png"
                    Image.fromarray(result.processed_image).save(output_path)
                    _save_comparison(
                        source,
                        result.processed_image,
                        comparison_path,
                        operation_name,
                    )
                transformation_log.add(
                    result,
                    sample_index=sample_index,
                    source_image_id=image.source_image_id,
                    filename=image.path.name,
                    image_path=image.relative_path,
                    processed_image=(
                        str(output_path.relative_to(run_directory))
                        if output_path
                        else None
                    ),
                    comparison_image=(
                        str(comparison_path.relative_to(run_directory))
                        if comparison_path
                        else None
                    ),
                )
                record.update(
                    sample_index=sample_index,
                    source_image_id=image.source_image_id,
                    filename=image.path.name,
                    image_path=image.relative_path,
                    output_file=(
                        str(output_path.relative_to(run_directory))
                        if output_path
                        else None
                    ),
                    comparison_file=(
                        str(comparison_path.relative_to(run_directory))
                        if comparison_path
                        else None
                    ),
                )
                operation_records.append(record)
            per_image_records.append(
                {
                    "sample_index": sample_index,
                    "source_image_id": image.source_image_id,
                    "filename": image.path.name,
                    "image_path": image.relative_path,
                    "annotations": metadata["annotations"],
                    "operations": operation_records,
                }
            )
            (sample_directory / "operation_metadata.json").write_text(
                json.dumps(
                    per_image_records[-1],
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )

        output_hashes = {
            image.relative_path: _sha256(image.path) for image in manifest.images
        }
        unchanged = input_hashes == output_hashes
        (run_directory / "source_hash_verification.json").write_text(
            json.dumps(
                {
                    "algorithm": "sha256",
                    "unchanged": unchanged,
                    "images": [
                        {
                            "image_path": name,
                            "sha256_before": input_hashes[name],
                            "sha256_after": output_hashes[name],
                            "unchanged": input_hashes[name] == output_hashes[name],
                        }
                        for name in input_hashes
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if not unchanged:
            raise RuntimeError("Source image hashes changed during preprocessing demo")

        (run_directory / "operation_metadata.json").write_text(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "sample_limit": args.sample_limit,
                    "seed": args.seed,
                    "sampling_method": "stratified_by_split_and_language",
                    "operations_run_independently": list(operations_config),
                    "images": per_image_records,
                },
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        transformation_log.write_csv(run_directory / "transformations.csv")
        transformation_log.write_json(run_directory / "transformations.json")
        logger.info("Preprocessing demonstration complete: %s", run_directory)
        logger.info(
            "Verified unchanged SHA-256 hashes for %d source images",
            len(input_hashes),
        )
        return 0
    except (
        DatasetConfigurationError,
        OSError,
        ValueError,
        yaml.YAMLError,
        RuntimeError,
    ) as error:
        logger.error("%s", error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
