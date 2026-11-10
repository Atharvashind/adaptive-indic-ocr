"""Collect and export records for independent preprocessing operations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.preprocessing.operation import OperationResult


class TransformationLog:
    """Keep JSON-compatible operation records and write a reproducible CSV."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    @property
    def records(self) -> list[dict[str, Any]]:
        return [dict(record) for record in self._records]

    def add(
        self,
        result: OperationResult,
        *,
        sample_index: int,
        source_image_id: str | None,
        filename: str,
        image_path: str,
        processed_image: str | None = None,
        comparison_image: str | None = None,
    ) -> None:
        """Append one completed operation and its source-image relationship."""
        self._records.append(
            {
                "sample_index": sample_index,
                "source_image_id": source_image_id,
                "filename": filename,
                "image_path": image_path,
                "operation_name": result.operation_name,
                "status": result.status,
                "success": result.success,
                "parameters": result.parameters,
                "input_dimensions": result.to_dict()["input_dimensions"],
                "output_dimensions": result.to_dict()["output_dimensions"],
                "execution_time_ms": result.execution_time_ms,
                "warnings": result.warnings,
                "details": result.details,
                "metadata": result.metadata,
                "processed_image": processed_image,
                "comparison_image": comparison_image,
            }
        )

    def write_json(self, path: Path) -> None:
        """Write all detailed records as UTF-8 JSON without losing Unicode."""
        path.write_text(
            json.dumps(self._records, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )

    def write_csv(self, path: Path) -> None:
        """Write a flat CSV with nested fields encoded as JSON strings."""
        if not self._records:
            raise ValueError("cannot write a transformation CSV with no records")
        fieldnames = list(self._records[0])
        with path.open("w", newline="", encoding="utf-8-sig") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for record in self._records:
                writer.writerow(
                    {
                        key: (
                            json.dumps(value, ensure_ascii=False, allow_nan=False)
                            if isinstance(value, (dict, list))
                            else value
                        )
                        for key, value in record.items()
                    }
                )
