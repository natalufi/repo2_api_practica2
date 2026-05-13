from __future__ import annotations

import numpy as np
import pandas as pd


class ProbabilityIntervalWrapper:
    """Modelo final servido por la API.

    Encapsula un estimador probabilistico ya ajustado y calcula un intervalo
    conformal simple sobre la probabilidad de impago.
    """

    def __init__(
        self,
        probability_model,
        alpha: float = 0.1,
        delegation_threshold: float = 0.2,
        n_bins: int = 10,
    ):
        self.probability_model = probability_model
        self.alpha = alpha
        self.delegation_threshold = delegation_threshold
        self.n_bins = n_bins
        self.qhat_ = None
        self.q_global_ = None
        self.q_bins_ = {}

    def fit_interval(self, X_cal, y_cal):
        p_cal = self.predict_proba(X_cal)[:, 1]
        y_cal = np.asarray(y_cal).reshape(-1)
        scores = np.abs(y_cal - p_cal)
        self.q_global_ = self._conformal_quantile(scores)
        self.qhat_ = self.q_global_

        bins = np.linspace(0.0, 1.0, self.n_bins + 1)
        self.q_bins_ = {}
        for bin_idx in range(self.n_bins):
            left, right = bins[bin_idx], bins[bin_idx + 1]
            if bin_idx == self.n_bins - 1:
                mask = (p_cal >= left) & (p_cal <= right)
            else:
                mask = (p_cal >= left) & (p_cal < right)

            if mask.sum() >= 20:
                self.q_bins_[bin_idx] = self._conformal_quantile(scores[mask])
            else:
                self.q_bins_[bin_idx] = self.q_global_
        return self

    def _conformal_quantile(self, scores):
        scores = np.asarray(scores)
        if len(scores) == 0:
            return 1.0
        q_level = np.ceil((len(scores) + 1) * (1 - self.alpha)) / len(scores)
        q_level = min(q_level, 1.0)
        return float(np.quantile(scores, q_level, method="higher"))

    def predict_proba(self, X):
        return self.probability_model.predict_proba(X)

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def predict_interval(self, X):
        q_global = getattr(self, "q_global_", None)
        qhat = getattr(self, "qhat_", None)
        if q_global is None and qhat is None:
            raise RuntimeError("El intervalo no esta ajustado. Ejecuta fit_interval primero.")
        p = self.predict_proba(X)[:, 1]
        q_values = self._lookup_q_values(p)
        p_low = np.clip(p - q_values, 0.0, 1.0)
        p_high = np.clip(p + q_values, 0.0, 1.0)
        return p_low, p_high

    def _lookup_q_values(self, probabilities):
        q_bins = getattr(self, "q_bins_", {})
        q_global = getattr(self, "q_global_", None)
        qhat = getattr(self, "qhat_", None)
        n_bins = getattr(self, "n_bins", 10)

        if not q_bins:
            return np.full_like(probabilities, qhat, dtype=float)

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        q_values = []
        for probability in probabilities:
            bin_idx = int(np.digitize(probability, bins, right=False) - 1)
            bin_idx = min(max(bin_idx, 0), n_bins - 1)
            q_values.append(q_bins.get(bin_idx, q_global))
        return np.asarray(q_values, dtype=float)

    def decision(self, X):
        p = self.predict_proba(X)[:, 1]
        p_low, p_high = self.predict_interval(X)
        width = p_high - p_low
        is_agent = width > self.delegation_threshold
        return pd.DataFrame(
            {
                "p_default": p,
                "p_low": p_low,
                "p_high": p_high,
                "width": width,
                "decision": np.where(is_agent, "agent", "auto"),
                "reason": np.where(
                    is_agent,
                    "p_high - p_low > 0.2",
                    "p_high - p_low <= 0.2",
                ),
            }
        )
