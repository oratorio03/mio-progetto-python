"""
m3_gap.py — M3: rating GAP/xG + modello diretto calibrato.

Filosofia opposta a M2: niente modello di punteggio, niente distribuzione di
gol. Si stimano direttamente le probabilità del mercato che interessa, con una
regressione logistica su poche feature costruite dai rating GAP (attacco e
difesa per venue, aggiornati sui gol o sull'xG quando disponibile).

Perché tenerlo nell'ensemble accanto a M1 e M2:
  - è l'approccio con la traccia di profitto di lungo periodo più solida su
    Over 2.5 e Home Win, proprio perché stima l'evento e non il punteggio;
  - è lineare e a bassa varianza: quando il gradient boosting overfitta su
    leghe con pochi dati, M3 tiene;
  - sbaglia in modo diverso dagli altri due, che è la condizione perché uno
    stacking abbia senso.

La parte "calibrata" è interna: la logistica viene addestrata sulla prima
parte del training e la mappa isotonica sull'ultima parte, mai sugli stessi
dati. Serve perché una logistica su feature correlate tende a essere troppo
sicura agli estremi.
"""

import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import BaseModel, empty_predictions, normalize_1x2, time_decay_weights
from ..calibration import IsotonicCalibrator
from ..config import ModelConfig

# Poche feature, tutte interpretabili e derivate dai rating GAP/xG.
GAP_FEATURES = [
    "gap_exp_supr", "gap_exp_total", "gap_log_home", "gap_log_away",
    "gap_ratio", "elo_diff_norm", "pi_exp_gd",
    "form_ppg_diff", "form_gf_diff", "form_ga_diff",
    "league_avg_goals", "league_home_rate", "elo_hfa", "reliability",
]

# Il totale gol atteso è il predittore naturale dei mercati sui gol:
# per Over 2.5 e BTTS si aggiungono i termini quadratici.
TOTAL_FEATURES = GAP_FEATURES + ["exp_total_vs_league", "supr_x_total",
                                 "form_o25_home", "form_o25_away",
                                 "form_btts_home", "form_btts_away"]

_CALIB_TAIL = 0.25   # quota finale del training riservata alla calibrazione


class GapDirectModel(BaseModel):
    """M3 — logistica sui rating GAP/xG con calibrazione isotonica interna."""

    name = "M3_gap"

    def __init__(self, cfg: ModelConfig = None, seed=42):
        super().__init__(cfg or ModelConfig())
        self.seed = seed
        self.models_ = {}
        self.calibrators_ = {}
        self.columns_ = {"1x2": GAP_FEATURES,
                         "over25": TOTAL_FEATURES,
                         "btts": TOTAL_FEATURES}

    def _pipeline(self, multiclass=False):
        # lbfgs su più classi è già multinomiale: il parametro multi_class
        # è deprecato/rimosso nelle versioni recenti di scikit-learn
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(C=self.cfg.m3_c, max_iter=2000,
                                      solver="lbfgs", random_state=self.seed)),
        ])

    def _matrix(self, feats, market):
        cols = self.columns_[market]
        X = feats.reindex(columns=cols).astype(float)
        return X.replace([np.inf, -np.inf], np.nan).to_numpy()

    def fit(self, df, feats):
        self.models_, self.calibrators_ = {}, {}
        w_all = time_decay_weights(df["kickoff"], half_life_days=self.cfg.m3_half_life_days)

        targets = {"1x2": ("y_1x2", True), "over25": ("y_over25", False),
                   "btts": ("y_btts", False)}
        for market, (col, multiclass) in targets.items():
            if col not in df.columns:
                continue
            y = df[col].to_numpy(dtype=float)
            mask = np.isfinite(y)
            if mask.sum() < 150 or len(np.unique(y[mask])) < 2:
                continue

            X = self._matrix(feats, market)[mask]
            yi = y[mask].astype(int)
            w = w_all[mask]

            # split temporale interno: le righe sono già in ordine cronologico
            n = len(yi)
            cut = max(int(n * (1 - _CALIB_TAIL)), 100)
            calibrator = IsotonicCalibrator()

            if n - cut >= 100:
                pipe = self._pipeline(multiclass)
                pipe.fit(X[:cut], yi[:cut], lr__sample_weight=w[:cut])
                p_cal = pipe.predict_proba(X[cut:])
                p_cal = self._expand(p_cal, pipe, multiclass)
                calibrator.fit(p_cal, yi[cut:])

            pipe_full = self._pipeline(multiclass)
            pipe_full.fit(X, yi, lr__sample_weight=w)
            self.models_[market] = pipe_full
            self.calibrators_[market] = calibrator

        self.fitted_ = bool(self.models_)
        return self

    @staticmethod
    def _expand(proba, pipe, multiclass):
        """Reinserisce eventuali classi assenti nel sottoinsieme di training."""
        n_classes = 3 if multiclass else 2
        classes = np.asarray(pipe.named_steps["lr"].classes_, dtype=int)
        if proba.shape[1] == n_classes and set(classes) == set(range(n_classes)):
            return proba
        full = np.full((len(proba), n_classes), 1e-6)
        for i, c in enumerate(classes):
            full[:, int(c)] = proba[:, i]
        return full / full.sum(axis=1, keepdims=True)

    def predict_proba(self, df, feats):
        n = len(df)
        if not self.fitted_ or n == 0:
            return empty_predictions(n)

        out = {}
        for market, n_classes in [("1x2", 3), ("over25", 2), ("btts", 2)]:
            pipe = self.models_.get(market)
            if pipe is None:
                out[market] = (np.tile([0.44, 0.26, 0.30], (n, 1)) if n_classes == 3
                               else np.full(n, 0.5))
                continue
            X = self._matrix(feats, market)
            p = self._expand(pipe.predict_proba(X), pipe, n_classes == 3)
            cal = self.calibrators_.get(market)
            if cal is not None:
                p = cal.transform(p)
            out[market] = normalize_1x2(p) if n_classes == 3 else p[:, 1]
        return out
