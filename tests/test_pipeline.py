from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from upi_fraud.features import History
from upi_fraud.generate import generate
from upi_fraud.train import train
from upi_fraud.api import create_app


def transaction(**updates):
    return dict(transaction_id="live-1", timestamp="2026-04-01T12:00:00+05:30", sender_id="u000",
                receiver_id="r1", amount=500., device_type="android", location="Mumbai", **updates)


def test_causal_history_and_windows():
    state = History()
    tx = transaction()
    first = state.observe(tx)
    assert first[2] == 0 and first[1] == .5
    second = state.observe({**tx, "timestamp": "2026-04-01T12:01:00+05:30", "amount": 2000})
    assert second[1] == 4 and second[2] == 1 and second[3] == 1
    third = state.observe({**tx, "timestamp": "2026-04-01T14:00:00+05:30"})
    assert third[2] == 0
    with pytest.raises(ValueError, match="predates"):
        state.observe(tx)


def test_future_does_not_change_past_features():
    rows = generate(200).to_dict("records")
    a, b = History(), History()
    before = [a.observe(row) for row in rows[:100]]
    altered = deepcopy(rows)
    for row in altered[100:]:
        row["amount"] *= 100
    after = [b.observe(row) for row in altered]
    np.testing.assert_equal(before, after[:100])


def test_generator():
    a = generate()
    pd.testing.assert_frame_equal(a, generate())
    assert len(a) == 10_000 and a.transaction_id.is_unique
    assert a.is_fraud.sum() == 400 and a.amount.gt(0).all()


def test_training_and_persistent_api(tmp_path):
    path = tmp_path / "data.csv"
    generate(1000).to_csv(path, index=False)
    report = train(path, tmp_path / "model")
    assert report["split"] == {"train": 600, "calibration": 200, "test": 200}
    model, db = tmp_path / "model/model.joblib", tmp_path / "alerts.sqlite3"
    tx = transaction()
    tx["amount"] = 500_000
    with TestClient(create_app(model, db)) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/transactions/score", json=tx)
        assert response.status_code == 200
        result = response.json()
        assert result["alert"]
        assert client.post("/transactions/score", json=tx).json() == result
        assert client.post("/transactions/score", json={**tx, "amount": 42}).status_code == 409
        assert client.post("/transactions/score", json={**tx, "transaction_id": "old", "timestamp": "2025-01-01T00:00:00Z"}).status_code == 409
        assert client.post("/transactions/score", json={**tx, "amount": -1}).status_code == 422
        assert client.post("/transactions/score", json={**tx, "timestamp": "2026-04-02T12:00:00"}).status_code == 422
        assert len(client.get("/alerts").json()) == 1
    with TestClient(create_app(model, db)) as client:
        assert client.post("/transactions/score", json=tx).json() == result
        next_tx = {**tx, "transaction_id": "live-2", "timestamp": "2026-04-01T12:01:00+05:30"}
        assert client.post("/transactions/score", json=next_tx).json()["features"]["count_hour"] == 1
