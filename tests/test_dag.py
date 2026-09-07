from __future__ import annotations

import ast
from importlib.metadata import version
from pathlib import Path

import pytest


AIRFLOW_MAJOR = int(version("apache-airflow").split(".", maxsplit=1)[0])
requires_airflow_3 = pytest.mark.skipif(
    AIRFLOW_MAJOR < 3,
    reason="The DAG uses the Airflow 3 public SDK; install requirements-dev.txt to run this test",
)

EXPECTED_TASKS = [
    "validate_bootstrap_assets",
    "detect_new_batch",
    "bronze",
    "silver",
    "quality",
    "gold",
    "cross_validate",
    "train",
    "evaluate",
    "register_and_promote",
]


def test_dag_source_defines_all_expected_tasks() -> None:
    dag_path = Path(__file__).resolve().parents[1] / "dags" / "sepsis_retraining_dag.py"
    tree = ast.parse(dag_path.read_text(encoding="utf-8"))
    task_functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(decorator, ast.Name) and decorator.id == "task"
            for decorator in node.decorator_list
        )
    }
    assert task_functions == set(EXPECTED_TASKS)


@pytest.fixture(scope="module")
def retraining_dag():
    from airflow.models import DagBag

    dag_folder = Path(__file__).resolve().parents[1] / "dags"
    dag_bag = DagBag(dag_folder=str(dag_folder), include_examples=False)
    assert dag_bag.import_errors == {}
    dag = dag_bag.get_dag("sepsis_retraining")
    assert dag is not None
    return dag


@requires_airflow_3
def test_dag_runtime_settings(retraining_dag) -> None:
    assert retraining_dag.catchup is False
    assert retraining_dag.max_active_runs == 1
    assert retraining_dag.start_date.isoformat() == "2026-01-01T00:00:00+00:00"


@requires_airflow_3
def test_dag_has_expected_linear_topology(retraining_dag) -> None:
    assert retraining_dag.task_ids == EXPECTED_TASKS
    for upstream_id, downstream_id in zip(EXPECTED_TASKS, EXPECTED_TASKS[1:]):
        downstream = retraining_dag.get_task(downstream_id)
        assert downstream.upstream_task_ids == {upstream_id}


@requires_airflow_3
def test_all_tasks_share_retry_policy(retraining_dag) -> None:
    assert all(task.retries == 1 for task in retraining_dag.tasks)
