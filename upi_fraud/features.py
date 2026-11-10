"""One causal feature implementation shared by training and serving."""
from collections import deque
from datetime import datetime
import math

FEATURES = ["log_amount", "amount_ratio", "count_hour", "repeat_hour",
            "spend_ratio", "gap_hours", "device_changed", "location_changed",
            "night", "hour_sin", "hour_cos", "cold_start"]


class History:
    def __init__(self):
        self.users = {}
        self.latest = None

    def observe(self, tx):
        """Compute from previous events, then update; require event-time order."""
        now = datetime.fromisoformat(str(tx["timestamp"]).replace("Z", "+00:00"))
        if now.tzinfo is None:
            raise ValueError("timestamp must include a timezone")
        if self.latest is not None and now < self.latest:
            raise ValueError("transaction predates the latest processed timestamp")
        amount = float(tx["amount"])
        u = self.users.setdefault(tx["sender_id"], {
            "n": 0, "total": 0., "last": None, "device": None,
            "location": None, "recent": deque()})
        while u["recent"] and (now - u["recent"][0][0]).total_seconds() > 3600:
            u["recent"].popleft()
        avg = u["total"] / u["n"] if u["n"] else 1000.
        count = len(u["recent"])
        repeat = sum(r[2] == tx["receiver_id"] for r in u["recent"])
        spend = sum(r[1] for r in u["recent"])
        # Behavior uses India local time, irrespective of the supplied UTC offset.
        from datetime import timezone, timedelta
        local = now.astimezone(timezone(timedelta(hours=5, minutes=30)))
        hour = local.hour + local.minute / 60
        values = [math.log1p(amount), amount / max(avg, 1), count, repeat,
                  (spend + amount) / max(avg, 1),
                  min((now-u["last"]).total_seconds()/3600, 168) if u["last"] else 168,
                  int(u["n"] > 0 and tx["device_type"] != u["device"]),
                  int(u["n"] > 0 and tx["location"] != u["location"]),
                  int(local.hour < 5), math.sin(hour*math.pi/12),
                  math.cos(hour*math.pi/12), int(u["n"] < 5)]
        u["n"] += 1
        u["total"] += amount
        u["last"], u["device"], u["location"] = now, tx["device_type"], tx["location"]
        u["recent"].append((now, amount, tx["receiver_id"]))
        self.latest = now
        return values
