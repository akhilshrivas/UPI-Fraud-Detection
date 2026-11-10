"""Model replacement must refresh unused state without erasing scored history."""
import hashlib
import pickle
import sqlite3

import joblib
import pytest
from fastapi.testclient import TestClient

from upi_fraud.api import create_app
from upi_fraud.features import History


@pytest.mark.parametrize("has_transactions", [False, True])
def test_model_replacement(tmp_path, has_transactions):
    model = tmp_path / "model.joblib"
    db = tmp_path / "alerts.sqlite3"
    old_history = History()
    # No predictions needed: this test isolates startup and storage behavior.
    joblib.dump(dict(detector=None, history=old_history, version="old"), model)
    with TestClient(create_app(model, db)):
        pass
    with sqlite3.connect(db) as con:
        if has_transactions:
            # A low-risk transaction also protects history, even with no alerts.
            con.execute("INSERT INTO transactions VALUES ('existing', '{}', '{}', 0)")
        previous = con.execute("SELECT digest, history FROM state WHERE id=1").fetchone()

    new_history = History()
    new_history.observe(dict(timestamp="2026-05-01T12:00:00+05:30", sender_id="u1",
                             receiver_id="r1", amount=100, device_type="android", location="Mumbai"))
    joblib.dump(dict(detector=None, history=new_history, version="new"), model)
    if has_transactions:
        with pytest.raises(RuntimeError, match="contains scored transactions"):
            with TestClient(create_app(model, db)):
                pass
        with sqlite3.connect(db) as con:
            assert con.execute("SELECT digest, history FROM state WHERE id=1").fetchone() == previous
            assert con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 1
    else:
        for _ in range(2):  # The refreshed state also survives a normal restart.
            with TestClient(create_app(model, db)) as client:
                assert client.get("/health").json()["latest_timestamp"] == new_history.latest.isoformat()
        with sqlite3.connect(db) as con:
            digest, history = con.execute("SELECT digest, history FROM state WHERE id=1").fetchone()
            assert digest == hashlib.sha256(model.read_bytes()).hexdigest()
            assert pickle.loads(history).users == new_history.users
