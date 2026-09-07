from __future__ import annotations

import json
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import pandas as pd
from airflow.sdk import dag, task
from airflow.sdk.exceptions import AirflowSkipException
from sklearn.model_selection import train_test_split

from src.pipelines.config import ensure_runtime_directories, load_config, load_schema, resolve_project_path
from src.pipelines.ingestion.batch_detector import detect_batch, save_batch_metadata
from src.pipelines.ingestion.bronze_pipeline import create_bronze
from src.pipelines.observability.retraining_logger import write_retraining_log
from src.pipelines.processing.gold_pipeline import build_gold
from src.pipelines.processing.silver_pipeline import (
    build_silver,
    load_raw_patients,
)
from src.pipelines.registry.dataset_registry import latest_dataset_metadata, next_dataset_version, register_dataset
from src.pipelines.registry.model_registry import promote_model, register_model
from src.pipelines.training.cv_search import run_training_workflow
from src.pipelines.training.evaluator import evaluate_model
from src.pipelines.training.trainer import train_final_model
from src.pipelines.validation.quality_gate import require_quality, validate_silver


CONFIG = load_config()
SCHEMA = load_schema()


def _path(value: str) -> str:
    return str(resolve_project_path(value))


def _write_failure_log(context) -> None:
    task_instance = context.get("task_instance")
    exception = context.get("exception")
    write_retraining_log(
        {
            "dag_run_id": context.get("run_id"),
            "status": "failed",
            "failed_task": getattr(task_instance, "task_id", None),
            "error_message": str(exception) if exception else None,
        },
        _path(CONFIG["logging"]["history_file"]),
    )


@dag(
    dag_id="sepsis_retraining",
    schedule=CONFIG["pipeline"]["schedule"],
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=int(CONFIG["pipeline"]["max_active_runs"]),
    default_args={
        "retries": int(CONFIG["pipeline"]["retries"]),
        "retry_delay": timedelta(minutes=5),
        "on_failure_callback": _write_failure_log,
    },
    tags=["sepsis", "retraining"],
)
def sepsis_retraining():
    @task
    def validate_bootstrap_assets() -> dict:
        ensure_runtime_directories(CONFIG)
        missing = []
        raw_root = resolve_project_path(CONFIG["data"]["raw_dir"])
        if not any(raw_root.rglob("*.psv")):
            missing.append(f"raw PSV data: {raw_root}")
        if missing:
            raise FileNotFoundError("Bootstrap assets are missing:\n" + "\n".join(missing))
        return {"status": "ready"}

    @task
    def detect_new_batch(_: dict) -> dict:
        batch = detect_batch(_path(CONFIG["data"]["incoming_dir"]), _path(CONFIG["data"]["registry_dir"]))
        if batch["status"] in {"no_data", "skipped_duplicate"}:
            raise AirflowSkipException(batch["status"])
        return batch

    @task
    def bronze(batch: dict) -> dict:
        return create_bronze(batch, _path(CONFIG["data"]["bronze_dir"]))

    @task
    def silver(batch: dict) -> dict:
        report = build_silver(
            batch["bronze_path"],
            _path(CONFIG["data"]["silver_dir"]),
            batch["batch_id"],
            SCHEMA,
        )
        return {**batch, **report}

    @task
    def quality(batch: dict) -> dict:
        report = validate_silver(batch["silver_path"], SCHEMA)
        require_quality(report)
        return {**batch, "quality_report": report}

    @task
    def gold(batch: dict) -> dict:
        registry_dir = _path(CONFIG["data"]["registry_dir"])
        version = next_dataset_version(registry_dir)
        parent = latest_dataset_metadata(registry_dir)
        bootstrap_development = None
        parent_features_path = parent["features_path"] if parent else None
        if parent is None:
            bootstrap_development = load_raw_patients(_path(CONFIG["data"]["raw_dir"]), SCHEMA)
        metadata = build_gold(
            batch["silver_path"],
            _path(CONFIG["data"]["gold_dir"]),
            version,
            batch["batch_id"],
            CONFIG["features"]["feature_set"],
            parent_features_path=parent_features_path,
            bootstrap_development=bootstrap_development,
            feature_config=CONFIG["features"],
            quality_report=batch["quality_report"],
        )
        register_dataset(metadata, registry_dir)
        save_batch_metadata({**batch, "status": "dataset_registered", "dataset_version": version}, registry_dir)
        return metadata

    @task
    def cross_validate(dataset: dict) -> dict:
        run_id = f"run_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
        run_dir = resolve_project_path(CONFIG["model_registry"]["runs_dir"]) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        frame = pd.read_parquet(dataset["features_path"])
        split_config = CONFIG["train_test_split"]
        manifest = frame.groupby("patient_id", as_index=False)[SCHEMA["label_column"]].max()
        seed = secrets.randbelow(2**32)
        train_ids, test_ids = train_test_split(
            manifest["patient_id"],
            test_size=float(split_config["test_size"]),
            random_state=seed,
            stratify=manifest[SCHEMA["label_column"]],
        )
        train_id_set = set(train_ids.astype(str))
        test_id_set = set(test_ids.astype(str))
        train_frame = frame[frame["patient_id"].astype(str).isin(train_id_set)].copy()
        test_frame = frame[frame["patient_id"].astype(str).isin(test_id_set)].copy()
        train_path = run_dir / "train.parquet"
        test_path = run_dir / "test.parquet"
        train_frame.to_parquet(train_path, index=False)
        test_frame.to_parquet(test_path, index=False)
        split_metadata = {
            "strategy": "patient-level stratified random split per retrain",
            "seed": seed,
            "test_size": float(split_config["test_size"]),
            "n_train_patients": len(train_id_set),
            "n_test_patients": len(test_id_set),
            "train_patient_ids": sorted(train_id_set),
            "test_patient_ids": sorted(test_id_set),
        }
        split_metadata_path = run_dir / "split_metadata.json"
        split_metadata_path.write_text(json.dumps(split_metadata, indent=2), encoding="utf-8")
        result = run_training_workflow(train_frame, CONFIG, SCHEMA)
        result_path = run_dir / "cv_result.joblib"
        joblib.dump(result, result_path)
        return {
            **dataset,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "cv_result_path": str(result_path),
            "train_path": str(train_path),
            "test_path": str(test_path),
            "split_metadata_path": str(split_metadata_path),
        }

    @task
    def train(run: dict) -> dict:
        frame = pd.read_parquet(run["train_path"])
        cv_result = joblib.load(run["cv_result_path"])
        candidate_dir = Path(run["run_dir"]) / "candidate"
        trained = train_final_model(
            frame,
            CONFIG,
            SCHEMA,
            cv_result,
            candidate_dir,
            run["dataset_version"],
        )
        return {**run, **trained}

    @task
    def evaluate(run: dict) -> dict:
        test_frame = pd.read_parquet(run["test_path"])
        candidate_metrics = evaluate_model(run["candidate_dir"], test_frame, CONFIG, SCHEMA)
        return {**run, "candidate_metrics": candidate_metrics}

    @task
    def register_and_promote(run: dict) -> dict:
        model_version = "model_" + run["run_id"].removeprefix("run_")
        registered = register_model(run["candidate_dir"], _path(CONFIG["model_registry"]["root_dir"]), model_version)
        pointer = promote_model(
            registered,
            _path(CONFIG["model_registry"]["current_model_file"]),
            run["dataset_version"],
            run["threshold"],
        )
        record = {
            "run_id": run["run_id"],
            "batch_id": run["batch_id"],
            "dataset_version": run["dataset_version"],
            "candidate_model": model_version,
            "current_model_after": model_version,
            "status": "success",
            "metrics": run["candidate_metrics"],
        }
        write_retraining_log(record, _path(CONFIG["logging"]["history_file"]))
        return {**record, "current_model_file": str(pointer)}

    ready = validate_bootstrap_assets()
    batch = detect_new_batch(ready)
    bronze_batch = bronze(batch)
    silver_batch = silver(bronze_batch)
    valid_batch = quality(silver_batch)
    dataset = gold(valid_batch)
    run = cross_validate(dataset)
    candidate = train(run)
    evaluated = evaluate(candidate)
    register_and_promote(evaluated)


sepsis_retraining_dag = sepsis_retraining()
