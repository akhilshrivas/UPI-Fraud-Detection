# UPI Fraud Detection

A runnable data science portfolio project that generates 10,000 synthetic UPI transactions, builds causal behavioral features, evaluates three anomaly detectors, and serves transaction scores through FastAPI with persistent SQLite alerts.

This is a simulation and local deployment demo. Synthetic scenario labels do not establish real-world fraud performance. The output flags transactions for review; it does not block payments.

## Run locally

Use Python 3.12. From this folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m upi_fraud.generate
.\.venv\Scripts\python -m upi_fraud.train
.\.venv\Scripts\python -m pytest -q --basetemp=.pytest_cache/test-temp
.\.venv\Scripts\python -m uvicorn upi_fraud.api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs to try the API. Generated files:

- `data/transactions.csv`: raw simulated records with scenario labels.
- `artifacts/metrics.json`: holdout metrics, split sizes, and limitations.
- `artifacts/test_predictions.csv`: individual holdout predictions.
- `artifacts/model.joblib`: trained detector and history through the end of the simulated dataset.
- `data/alerts.sqlite3`: live scores and behavioral state, created when the API starts.

## Score a transaction

In another PowerShell terminal while the server is running:

```powershell
$transaction = @{
    transaction_id = 'demo-001'
    timestamp = '2026-04-01T12:00:00+05:30'
    sender_id = 'u000'
    receiver_id = 'r1'
    amount = 500000
    device_type = 'android'
    location = 'Mumbai'
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/transactions/score -ContentType 'application/json' -Body $transaction
Invoke-RestMethod http://127.0.0.1:8000/alerts
```

Responses include the risk score, level, individual signals, explanations, model version, and behavioral feature values. `GET /health` reports the latest processed event timestamp. `GET /alerts?limit=50&offset=0` provides paginated alerts, newest first. Alerts are stored for polling; email, SMS, and webhooks are not implemented.

The model starts with historical state through March 1, 2026. New transactions must be at or after the latest processed timestamp, globally. Out-of-order events return HTTP 409 without modifying history. Include an explicit timezone; hour-of-day behavior is computed in India time. Duplicate transaction IDs with identical request data return the original result without updating history. Conflicting data for an existing ID returns 409. Invalid amounts and timezone-free timestamps return 422.

## Data and methodology

The seeded simulator represents 160 senders over 60 days, with different spending baselines, devices, cities, and daily activity patterns. Of 10,000 records, 400 have injected scenario labels: amount spikes, device/location changes, or repeated-recipient bursts. Normal events include occasional large purchases and travel, making some benign and suspicious behavior overlap. These labels are simulation ground truth, not independently confirmed fraud.

Rows are deduplicated and sorted chronologically. Conflicting IDs, missing values, and invalid amounts are rejected for source repair instead of silently imputed. Features use only previous events: amount relative to prior mean, rolling-hour count and spend, repeated recipients, time since previous event, device/location changes, cold-start status, and cyclical hour. The current event is added to history after its features are calculated. Label and scenario columns are excluded from model inputs.

The chronological split is 60% training, 20% calibration, and 20% test. Historical feature state advances through all periods, as it would in a live stream. The Isolation Forest is fitted only on training features. Its anomaly threshold is the 95th percentile of calibration anomaly scores; labels are not used to fit or tune it. No test-set tuning occurs. No scaling is applied because tree-based Isolation Forest does not require it; the amount feature is log-transformed.

Detection combines:

1. **IQR:** amount-to-history ratio exceeds training Q3 plus three IQRs. The wider fence limits sensitivity to ordinary large purchases.
2. **Isolation Forest:** 200 trees detect unusual combinations of behavioral features.
3. **Rolling time-series rules:** prior-hour count reaches the larger of three or the training 99th percentile; a recipient already appeared three times in the hour; or current-plus-recent spend exceeds eight prior-average transactions with at least two preceding events in the hour. This is rolling temporal anomaly detection, not a forecasting model.

One signal produces a review alert; two or three produce high risk. The 0/33/67/100 risk score is the percentage of signals firing, **not a calibrated fraud probability**. Signals are correlated, so agreement does not imply independent evidence.

Evaluation reports precision, recall, F1, alert rate, and confusion matrices for each method and their union. Matrices use `[[TN, FP], [FN, TP]]`. Isolation Forest average precision uses continuous anomaly scores. The union trades more review volume for potentially better recall. Inspect the actual generated metrics before making resume claims.

## Design and limitations

`generate.py` creates data; `features.py` shares the same causal logic between batch training and live scoring; `model.py` combines signals; `train.py` writes evaluation artifacts; `api.py` provides serving and storage. Tests cover reproducibility, causal history, rolling windows, time ordering, validation, duplicate requests, and persistence across restarts.

SQLite transactions serialize updates so that scoring and history commits are atomic. This demo stores the whole history state per request, which is simple but unsuitable for high-volume payment traffic. New users use a fixed INR 1,000 prior and a cold-start feature. Suspicious transactions update history too, so repeated attacks can contaminate the spending baseline. Global event-time ordering requires upstream ordering in a real deployment. The simulator's distribution and strong attack signatures make evaluation optimistic; account takeover changes and legitimate travel can be ambiguous.

The API has no authentication or authorization and should run locally. Public deployment needs access control, transport encryption, retention controls, monitoring, event ordering, more scalable storage, and evaluation on authorized representative data. Only load your own trusted model and database artifacts: joblib/pickle deserialization can execute code. When the model changes, startup refreshes the historical state automatically if the database has no scored transactions. If it contains any scored transactions (including low-risk ones), startup preserves the database and prints a PowerShell command for using a separate `DB_PATH`.

## Container deployment

```powershell
docker build -t upi-fraud .
docker run --rm -p 127.0.0.1:8000:8000 -e DB_PATH=/state/alerts.sqlite3 -v upi-fraud-state:/state upi-fraud
```

The image generates and trains its model during the build. The named volume persists live alerts and history. Use a new volume if a rebuilt image has a different model artifact and the existing database contains scored transactions. Empty databases refresh automatically. Docker deployment is provided as a recipe; it is separate from local Python verification.

## Portfolio wording

- Built an end-to-end UPI anomaly detection demo using 10,000 synthetic transactions, IQR, Isolation Forest, and rolling behavioral signals.
- Engineered causal sender-history features and evaluated precision, recall, and review volume on a chronological holdout set.
- Implemented a FastAPI scoring service with SQLite alert persistence, atomic history updates, and idempotent transaction ingestion.

In an interview, explain why synthetic evaluation cannot establish real-world fraud accuracy, how chronological features prevent leakage, and how alert volume affects manual review capacity. Do not claim production deployment or real financial loss prevention from this demo.
