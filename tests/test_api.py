"""Lightweight API smoke tests (run from project root: pytest tests/)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from ml.ui_mapping import ui_checkin_to_predict_body

CLIENT = TestClient(app)

PREDICT_BODY = ui_checkin_to_predict_body(
    {
        "study_hours": 3.0,
        "absences": 2,
        "sleep_quality": 6,
        "mental_health": 6,
        "financial_stress": 6,
        "electricity_hours": 8,
        "internet_quality": 6,
        "war_stress": 5,
        "motivation": 6,
    }
)


def _skip_if_no_model() -> None:
    r = CLIENT.get("/health")
    assert r.status_code == 200
    if not r.json().get("model_loaded"):
        pytest.skip("No model artifact at artifacts/model.joblib (train first)")


def test_health() -> None:
    r = CLIENT.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("ok", "degraded")
    assert isinstance(data["model_loaded"], bool)
    assert "artifact_path" in data


def test_chat_support() -> None:
    r = CLIENT.post(
        "/chat/support",
        json={"message": "I feel overwhelmed with exams and sleep"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data.get("reply")
    assert data.get("safety_note")
    assert isinstance(data.get("supportive_actions"), list)


def test_predict_when_model_present() -> None:
    _skip_if_no_model()
    r = CLIENT.post("/predict", json=PREDICT_BODY)
    assert r.status_code == 200
    data = r.json()
    assert data["risk_label"] in (0, 1)
    assert 0.0 <= float(data["risk_probability"]) <= 1.0
    assert data["status_label"]
    assert isinstance(data.get("top_factors"), list)


def test_feature_importance_when_model_present() -> None:
    _skip_if_no_model()
    r = CLIENT.get("/feature-importance")
    assert r.status_code == 200
    data = r.json()
    assert data.get("source")
    feats = data.get("features") or []
    assert len(feats) >= 1
    assert all("importance" in row for row in feats)


def test_explain_when_model_present() -> None:
    _skip_if_no_model()
    r = CLIENT.post("/explain?top_k=5", json=PREDICT_BODY)
    assert r.status_code == 200
    data = r.json()
    assert data["method"] in ("shap_tree", "approximate")
    assert data["risk_label"] in (0, 1)
    factors = data.get("factors") or []
    assert len(factors) >= 1
    for f in factors:
        assert f.get("method") in ("shap_tree", "approximate")
