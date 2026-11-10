"""Local demo API with atomic SQLite history updates and idempotent ingestion."""
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
import hashlib
import json
import logging
import os
import pickle
import sqlite3
from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    transaction_id: str = Field(min_length=1, max_length=100)
    timestamp: datetime
    sender_id: str = Field(min_length=1, max_length=100)
    receiver_id: str = Field(min_length=1, max_length=100)
    amount: float = Field(gt=0, le=10_000_000)
    device_type: str = Field(min_length=1, max_length=100)
    location: str = Field(min_length=1, max_length=100)
    merchant_category: str = "unknown"
    transaction_type: str = "payment"
    upi_channel: str = "app"

    @field_validator("timestamp")
    @classmethod
    def aware(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        return value


def create_app(model_path=None, db_path=None):
    model_path = Path(model_path or os.getenv("MODEL_PATH", "artifacts/model.joblib"))
    db_path = Path(db_path or os.getenv("DB_PATH", "data/alerts.sqlite3"))

    @contextmanager
    def connect():
        con = sqlite3.connect(db_path, timeout=30)
        con.row_factory = sqlite3.Row
        try:
            with con:
                yield con
        finally:
            con.close()

    @asynccontextmanager
    async def lifespan(app):
        if not model_path.exists():
            raise RuntimeError("Run python -m upi_fraud.generate and python -m upi_fraud.train first")
        # Only load locally generated trusted artifacts: joblib/pickle execute code.
        bundle = joblib.load(model_path)
        app.state.detector = bundle["detector"]
        app.state.version = bundle["version"]
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with connect() as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute("CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, digest TEXT, history BLOB)")
            con.execute("CREATE TABLE IF NOT EXISTS transactions (id TEXT PRIMARY KEY, payload TEXT, result TEXT, alert INTEGER)")
            con.execute("INSERT OR IGNORE INTO state VALUES (1, ?, ?)", (digest, pickle.dumps(bundle["history"])))
            if con.execute("SELECT digest FROM state WHERE id=1").fetchone()[0] != digest:
                if con.execute("SELECT EXISTS(SELECT 1 FROM transactions)").fetchone()[0]:
                    next_db = db_path.with_name(f"{db_path.stem}-{digest[:12]}{db_path.suffix}")
                    raise RuntimeError(
                        f"Model changed and {db_path} contains scored transactions. "
                        "Existing history has been preserved. Start with a separate database; "
                        f'in PowerShell run: $env:DB_PATH = "{next_db}"; '
                        "python -m uvicorn upi_fraud.api:app"
                    )
                con.execute("UPDATE state SET digest=?, history=? WHERE id=1",
                            (digest, pickle.dumps(bundle["history"])))
                logging.getLogger(__name__).warning(
                    "Model changed; initialized empty database %s with the current model history", db_path
                )
        yield

    app = FastAPI(title="UPI Fraud Detection", version="1.0.0", lifespan=lifespan,
                  description="Synthetic-data portfolio demo. Risk scores are signal votes, not probabilities.")

    @app.get("/health")
    def health():
        with connect() as con:
            history = pickle.loads(con.execute("SELECT history FROM state WHERE id=1").fetchone()[0])
        return dict(status="ok", model_version=app.state.version, latest_timestamp=history.latest.isoformat())

    @app.post("/transactions/score")
    def score(tx: Transaction):
        payload = tx.model_dump(mode="json")
        canonical = json.dumps(payload, sort_keys=True)
        with connect() as con:
            con.execute("BEGIN IMMEDIATE")
            existing = con.execute("SELECT payload, result FROM transactions WHERE id=?", (tx.transaction_id,)).fetchone()
            if existing:
                if existing["payload"] != canonical:
                    raise HTTPException(409, "Transaction ID already used with different data")
                return json.loads(existing["result"])
            state = pickle.loads(con.execute("SELECT history FROM state WHERE id=1").fetchone()[0])
            try:
                features = state.observe(payload)
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            result = dict(transaction_id=tx.transaction_id, model_version=app.state.version,
                          **app.state.detector.predict([features])[0])
            con.execute("INSERT INTO transactions VALUES (?, ?, ?, ?)",
                        (tx.transaction_id, canonical, json.dumps(result), int(result["alert"])))
            con.execute("UPDATE state SET history=? WHERE id=1", (pickle.dumps(state),))
        return result

    @app.get("/alerts")
    def alerts(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
        with connect() as con:
            records = con.execute("SELECT result FROM transactions WHERE alert=1 ORDER BY rowid DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [json.loads(row[0]) for row in records]

    return app


app = create_app()
