"""Reproducible behavioral simulation; labels are scenario truth, not real fraud."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate(n=10_000, seed=42):
    if n < 100:
        raise ValueError("Generate at least 100 transactions")
    rng = np.random.default_rng(seed)
    users = 160
    means = rng.lognormal(6.3, .65, users)
    cities = rng.choice(["Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad"], users)
    devices = rng.choice(["android", "ios"], users, p=[.8, .2])
    base = pd.Timestamp("2026-01-01", tz="Asia/Kolkata")
    rows = []
    # Burst rows are generated in coherent sessions, then all events are sorted.
    normal_count = int(n * .96)
    for i in range(normal_count):
        user = int(rng.integers(users))
        day = int(rng.integers(60))
        hour = int(rng.choice(np.arange(24), p=np.array([1]*6+[3]*3+[5]*13+[2]*2)/84))
        stamp = base + pd.Timedelta(days=day, hours=hour, seconds=int(rng.integers(3600)))
        amount = float(rng.lognormal(np.log(means[user]), .55))
        # Legitimate large purchases and travel deliberately overlap anomalies.
        if rng.random() < .015:
            amount *= 8
        location = str(rng.choice(cities)) if rng.random() < .02 else str(cities[user])
        rows.append(_row(user, stamp, amount, f"r{rng.integers(25)}", str(devices[user]), location, "normal", rng))
    while len(rows) < n:
        user = int(rng.integers(users))
        scenario = str(rng.choice(["amount_spike", "account_change", "recipient_burst"]))
        stamp = base + pd.Timedelta(days=int(rng.integers(60)), hours=int(rng.integers(24)))
        burst = min(int(rng.integers(5, 10)), n-len(rows)) if scenario == "recipient_burst" else 1
        receiver = f"r{rng.integers(25)}"
        for k in range(burst):
            amount = means[user] * rng.uniform(12, 25) if scenario == "amount_spike" else means[user]*rng.uniform(1, 4)
            device = "new_device" if scenario == "account_change" else str(devices[user])
            location = "new_location" if scenario == "account_change" else str(cities[user])
            rows.append(_row(user, stamp+pd.Timedelta(seconds=k*40), amount, receiver, device, location, scenario, rng))
    frame = pd.DataFrame(rows).sort_values("timestamp", kind="stable").reset_index(drop=True)
    frame.insert(0, "transaction_id", [f"sim-{i:06d}" for i in range(n)])
    return frame


def _row(user, stamp, amount, receiver, device, location, scenario, rng):
    return dict(timestamp=stamp.isoformat(), sender_id=f"u{user:03d}", receiver_id=receiver,
                amount=round(max(amount, 1), 2), merchant_category=str(rng.choice(["grocery", "travel", "retail", "food"])),
                transaction_type="payment", location=location, device_type=device,
                upi_channel=str(rng.choice(["app", "qr"])), is_fraud=int(scenario != "normal"), scenario=scenario)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/transactions.csv")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    generate(args.rows, args.seed).to_csv(args.output, index=False)
    print(f"Saved {args.rows:,} synthetic transactions to {args.output}")
