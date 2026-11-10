from __future__ import annotations

from experiments.ocr_evaluation import run_phase5_final as phase5c


def test_cluster_bootstrap_ci_is_reproducible_and_contains_mean() -> None:
    grouped = {
        "board-a": [0.2, 0.0],
        "board-b": [-0.1],
        "board-c": [0.3],
    }

    first = phase5c._cluster_bootstrap_ci(grouped, seed=7, replicates=100)
    second = phase5c._cluster_bootstrap_ci(grouped, seed=7, replicates=100)

    assert first == second
    mean_delta = sum(sum(values) for values in grouped.values()) / sum(
        len(values) for values in grouped.values()
    )
    assert first["lower_95_percent"] <= mean_delta <= first["upper_95_percent"]
    assert first["cluster_count"] == 3


def test_metrics_report_paired_deltas_and_operation_conditioning() -> None:
    rows = [
        {
            "filename": "line-a.png",
            "source_image_id": "board-a",
            "language": "hi",
            "transcription": "राम",
        },
        {
            "filename": "line-b.png",
            "source_image_id": "board-b",
            "language": "ta",
            "transcription": "தமிழ்",
        },
    ]
    predictions = {
        arm: {row["filename"]: row["transcription"] for row in rows}
        for arm in phase5c.ARM_ORDER
    }
    predictions["fixed"]["line-a.png"] = "राज"
    records = [
        {
            "arm": arm,
            "filename": row["filename"],
            "language": row["language"],
            "operation_selected": operation,
            "transformation_status": "success",
            "total_preprocessing_time_ms": 1.0,
        }
        for arm, operation in (
            ("original", "keep_original"),
            ("fixed", "histogram_equalization"),
            ("scene_text", "clahe"),
            ("automatic", "keep_original"),
            ("user_guided", "keep_original"),
        )
        for row in rows
    ]

    metrics, per_language, prediction_rows = phase5c._build_metrics(
        rows,
        predictions,
        records,
    )
    operation_rows = phase5c._operation_selection_rows(records, prediction_rows)

    assert metrics["arms"]["fixed"]["paired_sample_cer_difference"]["degraded_count"] == 1
    assert metrics["arms"]["fixed"]["paired_sample_cer_difference"]["unchanged_count"] == 1
    assert metrics["arms"]["original"]["aggregate_delta_cer_vs_original"] == 0.0
    assert len(per_language) == len(phase5c.ARM_ORDER) * 2
    assert any(
        row["arm"] == "fixed"
        and row["operation_selected"] == "histogram_equalization"
        for row in operation_rows
    )
    assert len(prediction_rows) == len(phase5c.ARM_ORDER) * len(rows)
