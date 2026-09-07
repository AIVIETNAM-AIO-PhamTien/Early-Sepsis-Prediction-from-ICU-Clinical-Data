from __future__ import annotations

from src.inference.loader import ModelLoader
from src.inference.predictor import run_prediction
from src.pipelines.config import load_config, load_schema
from src.pipelines.registry.model_registry import promote_model, register_model
from src.pipelines.training.evaluator import evaluate_model
from src.pipelines.training.trainer import train_final_model


def test_retrained_artifact_can_be_promoted_and_used_for_prediction(
    tmp_path,
    valid_silver_frame,
    sample_patient_df,
) -> None:
    config = load_config()
    schema = load_schema()
    cv_result = {
        "params": {"max_depth": 1, "eta": 0.3, "tree_method": "hist", "seed": 42},
        "mean_best_iteration": 2,
        "best_threshold": 0.5,
        "log_columns": [],
    }
    candidate = tmp_path / "candidate"
    trained = train_final_model(
        valid_silver_frame,
        config,
        schema,
        cv_result,
        candidate,
        "dataset_test",
    )
    evaluate_model(candidate, valid_silver_frame, config, schema)
    registered = register_model(candidate, tmp_path / "models", "model_test")
    manifest = promote_model(
        registered,
        tmp_path / "current_model.json",
        "dataset_test",
        trained["threshold"],
        pipeline="team_v1",
    )

    results, validation, artifact = run_prediction(
        sample_patient_df,
        "sample_patient",
        ModelLoader(manifest),
    )

    assert validation.is_valid
    assert artifact is not None
    assert artifact.pipeline == "team_v1"
    assert artifact.model_format == "xgboost_json"
    assert len(results) == len(sample_patient_df)
    assert results["risk_score"].between(0.0, 1.0).all()
