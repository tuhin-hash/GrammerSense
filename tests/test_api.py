"""
API integration tests using FastAPI TestClient.
Tests all required endpoints: /api/health, /api/correct, /api/detect, /api/candidates, /api/evaluate.
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_name" in data
    assert "device" in data
    assert "version" in data


def test_api_correct():
    payload = {
        "text": "She go to college every day.",
        "num_candidates": 2,
        "num_beams": 2
    }
    response = client.post("/api/correct", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "original_text" in data
    assert "corrected_text" in data
    assert "candidates" in data
    assert "detected_errors" in data
    assert "diff_chunks" in data
    assert "metrics" in data
    assert data["original_text"] == payload["text"]
    assert len(data["candidates"]) >= 1


def test_api_detect():
    payload = {"text": "He have two brothers."}
    response = client.post("/api/detect", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "original_text" in data
    assert "detected_errors" in data
    assert "error_count" in data
    assert "has_errors" in data
    assert data["has_errors"] is True


def test_api_candidates():
    payload = {"text": "They was playing cricket.", "num_candidates": 3}
    response = client.post("/api/candidates", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "candidates" in data
    assert len(data["candidates"]) >= 1
    for cand in data["candidates"]:
        assert "candidate_text" in cand
        assert "rank" in cand
        assert "confidence" in cand


def test_api_evaluate_custom_pairs():
    payload = {
        "custom_pairs": [
            {"source": "She go to college.", "target": "She goes to college."},
            {"source": "He have two cars.", "target": "He has two cars."}
        ]
    }
    response = client.post("/api/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "reproducibility" in data
    assert "metrics" in data
    assert "corpus_gleu" in data["metrics"]
    assert "token_f0_5_score_percent" in data["metrics"]
