from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def build_gold(
    silver_path: str | Path,
    gold_root: str | Path,
    dataset_version: str,
    batch_id: str,
    feature_schema_version: str,
    parent_features_path: str | Path | None = None,
    bootstrap_development: pd.DataFrame | None = None,
    feature_config: dict[str, Any] | None = None,
    quality_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    new_batch = pd.read_parquet(silver_path)
    if parent_features_path is not None:
        previous = pd.read_parquet(parent_features_path)
    elif bootstrap_development is not None:
        previous = bootstrap_development
    else:
        previous = pd.DataFrame(columns=new_batch.columns)
    previous_ids = set(previous.get("patient_id", pd.Series(dtype=str)).astype(str))
    incoming_ids = set(new_batch["patient_id"].astype(str))
    overlap = previous_ids & incoming_ids
    if overlap:
        previous = previous[~previous["patient_id"].astype(str).isin(overlap)].copy()
    frame = pd.concat([previous, new_batch], ignore_index=True)
    frame = frame.sort_values(["patient_id", "ICULOS"], kind="stable").reset_index(drop=True)
    destination = Path(gold_root) / dataset_version
    destination.mkdir(parents=True, exist_ok=False)
    feature_path = destination / "features.parquet"
    frame.to_parquet(feature_path, index=False)

    manifest = (
        frame.groupby("patient_id", sort=False)
        .agg(source=("source", "first"), has_sepsis=("SepsisLabel", "max"), n_hours=("ICULOS", "size"))
        .reset_index()
    )
    manifest["batch_id"] = batch_id
    manifest["dataset_split"] = "development"
    manifest_path = destination / "patient_manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    metadata = {
        "dataset_version": dataset_version,
        "batch_id": batch_id,
        "parent_features_path": str(parent_features_path) if parent_features_path else None,
        "feature_schema_version": feature_schema_version,
        "n_patients": int(manifest["patient_id"].nunique()),
        "n_rows": int(len(frame)),
        "n_new_patients": len(incoming_ids - previous_ids),
        "n_replaced_patients": len(overlap),
        "features_path": str(feature_path),
        "manifest_path": str(manifest_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (destination / "feature_config.json").write_text(
        json.dumps(feature_config or {}, indent=2, sort_keys=True), encoding="utf-8"
    )
    (destination / "feature_schema.json").write_text(
        json.dumps(
            {
                "canonical_columns": frame.columns.tolist(),
                "feature_schema_version": feature_schema_version,
                "prepared_features_are_fold_specific": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (destination / "quality_report.json").write_text(
        json.dumps(quality_report or {}, indent=2, sort_keys=True), encoding="utf-8"
    )
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
