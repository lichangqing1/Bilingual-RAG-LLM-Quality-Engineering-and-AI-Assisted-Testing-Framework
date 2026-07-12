import json

import pytest


fastapi = pytest.importorskip("fastapi")
testclient = pytest.importorskip("fastapi.testclient")


def test_fastapi_ask_endpoint_logs_request(tmp_path, monkeypatch):
    import api

    log_path = tmp_path / "api_requests.jsonl"
    original_append_jsonl = api.append_jsonl

    def append_to_tmp(path, record):
        return original_append_jsonl(log_path, record)

    monkeypatch.setattr(api, "append_jsonl", append_to_tmp)
    client = testclient.TestClient(api.app)

    response = client.post("/ask", json={"question": "标准配送通常需要多长时间?", "top_k": 3})

    assert response.status_code == 200
    payload = response.json()
    assert "3到5个工作日" in payload["answer"]
    assert "shipping_policy_zh.md" in payload["sources"]
    log_record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert log_record["event"] == "api_request_completed"
    assert log_record["question"] == "标准配送通常需要多长时间?"


def test_fastapi_evaluate_feedback_and_metrics_endpoints(tmp_path, monkeypatch):
    import api

    original_append_jsonl = api.append_jsonl

    def append_to_tmp(path, record):
        return original_append_jsonl(tmp_path / path.name, record)

    monkeypatch.setattr(api, "append_jsonl", append_to_tmp)
    client = testclient.TestClient(api.app)

    evaluation_response = client.post(
        "/evaluate",
        json={
            "question": "标准配送通常需要多长时间?",
            "expected_answer": "标准配送通常需要3到5个工作日。",
            "expected_source": "shipping_policy_zh.md",
            "expected_keywords": "标准配送;3到5个工作日",
            "question_type": "normal",
            "top_k": 3,
        },
    )
    assert evaluation_response.status_code == 200
    metrics = evaluation_response.json()["metrics"]
    assert metrics["ragas_faithfulness"] == 1.0
    assert metrics["overall_pass"] == 1

    feedback_response = client.post(
        "/feedback",
        json={
            "question": "标准配送通常需要多长时间?",
            "answer": evaluation_response.json()["answer"],
            "rating": 5,
            "comment": "Grounded answer",
        },
    )
    assert feedback_response.status_code == 200
    assert feedback_response.json()["status"] == "logged"

    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200
    assert metrics_response.json()["status"] == "ok"
