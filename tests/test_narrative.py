"""Tests for narrative summary and template provider."""

from __future__ import annotations

import pandas as pd
import pytest

from src.narrative.context import build_prediction_summary
from src.narrative.providers import TemplateProvider
from src.narrative.service import generate_narrative


@pytest.fixture
def sample_results() -> pd.DataFrame:
    n = 54
    scores = [0.2] * 50 + [0.56, 0.55, 0.40, 0.34]
    preds = [1 if s >= 0.5 else 0 for s in scores]
    return pd.DataFrame(
        {
            "patient_id": ["p000001"] * n,
            "ICULOS": list(range(1, n + 1)),
            "risk_score": scores,
            "prediction": preds,
            "model_version": ["v1"] * n,
        }
    )


def test_build_prediction_summary(sample_results: pd.DataFrame):
    summary = build_prediction_summary(
        sample_results, patient_id="p000001", threshold=0.5
    )
    assert summary.patient_id == "p000001"
    assert summary.n_hours == 54
    assert summary.latest_score == pytest.approx(0.34)
    assert summary.peak_score == pytest.approx(0.56)
    assert summary.elevated_hours == 2
    assert summary.current_status == "Watch"
    assert summary.latest_iculos == 54
    assert summary.peak_iculos == 51


def test_template_provider_vietnamese(sample_results: pd.DataFrame):
    summary = build_prediction_summary(
        sample_results, patient_id="p000001", threshold=0.5
    )
    text = TemplateProvider().generate(summary, language="vi")
    assert "p000001" in text
    assert "34" in text or "34.0%" in text
    assert "model" in text.lower() or "Model" in text or "nguy cơ" in text


def test_generate_narrative_template(sample_results: pd.DataFrame):
    text, provider = generate_narrative(
        sample_results,
        patient_id="p000001",
        threshold=0.5,
        provider_name="template",
        language="vi",
    )
    assert provider == "template"
    assert len(text) > 50


def test_generate_narrative_empty_raises():
    with pytest.raises(Exception):
        generate_narrative(
            pd.DataFrame(),
            patient_id="x",
            threshold=0.5,
            provider_name="template",
        )
