import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, confusion_matrix
from .features import History
from .model import Detector


def train(path="data/transactions.csv", output="artifacts"):
    df = pd.read_csv(path).drop_duplicates()
    if df.transaction_id.duplicated().any():
        raise ValueError("Conflicting transaction IDs")
    if df.isna().any().any():
        raise ValueError("Missing values: repair source records before training")
    if not np.isfinite(df.amount).all() or (df.amount <= 0).any():
        raise ValueError("Amounts must be finite and positive")
    df.timestamp = pd.to_datetime(df.timestamp, utc=True)
    df = df.sort_values("timestamp", kind="stable")
    if len(df) < 100:
        raise ValueError("At least 100 rows required")
    state = History()
    x = np.array([state.observe(row) for row in df.to_dict("records")])
    a, b = int(len(df)*.6), int(len(df)*.8)
    detector = Detector().fit(x[:a], x[a:b])
    results = detector.predict(x[b:])
    y = df.is_fraud.to_numpy()[b:]
    report = {"rows": len(df), "split": {"train": a, "calibration": b-a, "test": len(df)-b},
              "test_fraud_count": int(y.sum()), "test_start": str(df.iloc[b].timestamp),
              "test_end": str(df.iloc[-1].timestamp), "methods": {}}
    for method in ["iqr", "isolation_forest", "time_series", "combined"]:
        pred = [r["alert"] if method == "combined" else r["signals"][method] for r in results]
        report["methods"][method] = dict(precision=float(precision_score(y, pred, zero_division=0)),
            recall=float(recall_score(y, pred, zero_division=0)), f1=float(f1_score(y, pred, zero_division=0)),
            alert_rate=float(np.mean(pred)), confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist())
    report["isolation_average_precision"] = float(average_precision_score(y, [r["isolation_score"] for r in results]))
    report["limitations"] = "Synthetic scenario labels only; not evidence of real-world fraud detection performance. Risk score is a signal vote, not a probability."
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(detector=detector, history=state, version="synthetic-v1", report=report), output/"model.joblib")
    (output/"metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame([dict(transaction_id=tid, actual=int(label), **{k:r[k] for k in ["risk_score", "risk_level", "alert"]})
                  for tid, label, r in zip(df.transaction_id.iloc[b:], y, results)]).to_csv(output/"test_predictions.csv", index=False)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/transactions.csv")
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    train(args.data, args.output)
