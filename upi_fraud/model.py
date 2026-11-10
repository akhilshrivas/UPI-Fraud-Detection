import numpy as np
from sklearn.ensemble import IsolationForest
from .features import FEATURES


class Detector:
    def fit(self, train, calibration):
        self.forest = IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=1)
        self.forest.fit(train)
        self.if_threshold = float(np.quantile(-self.forest.score_samples(calibration), .95))
        q1, q3 = np.quantile(train[:, 1], [.25, .75])
        self.iqr_upper = float(q3 + 3 * (q3-q1))
        self.count_upper = max(3., float(np.quantile(train[:, 2], .99)))
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        isolation = -self.forest.score_samples(x)
        output = []
        for row, value in zip(x, isolation):
            iqr = bool(row[1] > self.iqr_upper)
            temporal = bool(row[2] >= self.count_upper or row[3] >= 3 or (row[2] >= 2 and row[4] >= 8))
            forest = bool(value >= self.if_threshold)
            reasons = []
            if iqr:
                reasons.append("Amount unusually high relative to prior sender average")
            if forest:
                reasons.append("Multivariate Isolation Forest anomaly")
            if temporal:
                reasons.append("Unusual rolling-hour activity or repeated recipient")
            votes = int(iqr) + int(forest) + int(temporal)
            output.append(dict(risk_score=round(100*votes/3), risk_level="high" if votes >= 2 else "review" if votes else "low",
                               alert=bool(votes), signals=dict(iqr=iqr, isolation_forest=forest, time_series=temporal),
                               reasons=reasons, isolation_score=float(value),
                               features=dict(zip(FEATURES, row.tolist()))))
        return output
