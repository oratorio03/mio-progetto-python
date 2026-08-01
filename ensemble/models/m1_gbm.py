"""
m1_gbm.py — M1: gradient boosting su rating, forma e contesto.

È il modello che nella letteratura recente dà il miglior RPS sul 1X2:
alberi in boosting su feature di rating (pi-ratings, Elo, GAP/xG) e forma
battono i modelli di conteggio classici perché catturano interazioni e
non linearità (es. il valore della differenza di rating cambia con il
livello di gol atteso della lega).

Tre teste indipendenti — 1X2 multiclasse, Over 2.5 binaria, BTTS binaria —
perché i tre mercati hanno strutture di dipendenza diverse e forzarli in un
unico modello di punteggio (come fa il Poisson) è proprio il limite che
vogliamo superare.

Backend: CatBoost > XGBoost > LightGBM > sklearn HistGradientBoosting.
Il primo disponibile viene usato; tutti gestiscono i NaN nativamente.
"""

import numpy as np

from dataclasses import replace

from .base import BaseModel, normalize_1x2, time_decay_weights
from ..config import ModelConfig
from ..features import feature_columns, prepare_matrix

_BACKEND_ORDER = ["catboost", "xgboost", "lightgbm", "sklearn"]
_VAL_FRACTION = 0.15      # coda di training usata per l'early stopping
_ES_ROUNDS = 40


def _available(backend):
    try:
        if backend == "catboost":
            import catboost  # noqa: F401
        elif backend == "xgboost":
            import xgboost  # noqa: F401
        elif backend == "lightgbm":
            import lightgbm  # noqa: F401
        elif backend == "sklearn":
            from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: F401
        else:
            return False
        return True
    except Exception:
        return False


def resolve_backend(preferred="auto"):
    """Sceglie il backend disponibile, con fallback garantito su sklearn."""
    if preferred and preferred != "auto":
        if _available(preferred):
            return preferred
        print(f"  [M1] backend '{preferred}' non disponibile → fallback automatico")
    for b in _BACKEND_ORDER:
        if _available(b):
            return b
    raise ImportError("nessun backend di gradient boosting disponibile (serve almeno scikit-learn)")


def _make_estimator(backend, cfg: ModelConfig, n_classes, seed):
    """Costruisce il classificatore con iperparametri equivalenti fra backend."""
    if backend == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=cfg.m1_n_estimators, learning_rate=cfg.m1_learning_rate,
            depth=cfg.m1_max_depth, l2_leaf_reg=max(cfg.m1_l2, 1.0),
            loss_function="MultiClass" if n_classes > 2 else "Logloss",
            random_seed=seed, verbose=False, allow_writing_files=False,
        )
    if backend == "xgboost":
        from xgboost import XGBClassifier
        # l'API sklearn di xgboost deduce num_class da y: passarlo a mano dà errore
        return XGBClassifier(
            n_estimators=cfg.m1_n_estimators, learning_rate=cfg.m1_learning_rate,
            max_depth=cfg.m1_max_depth, subsample=cfg.m1_subsample,
            reg_lambda=cfg.m1_l2, colsample_bytree=0.8,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            random_state=seed, n_jobs=-1, tree_method="hist", verbosity=0,
        )
    if backend == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=cfg.m1_n_estimators, learning_rate=cfg.m1_learning_rate,
            max_depth=cfg.m1_max_depth, num_leaves=2 ** cfg.m1_max_depth,
            min_child_samples=cfg.m1_min_samples, subsample=cfg.m1_subsample,
            subsample_freq=1, colsample_bytree=0.8, reg_lambda=cfg.m1_l2,
            objective="multiclass" if n_classes > 2 else "binary",
            random_state=seed, n_jobs=-1, verbose=-1,
        )
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=cfg.m1_n_estimators, learning_rate=cfg.m1_learning_rate,
        max_depth=cfg.m1_max_depth, min_samples_leaf=cfg.m1_min_samples,
        l2_regularization=cfg.m1_l2, early_stopping=False, random_state=seed,
    )


def _early_stop_rounds(backend, cfg: ModelConfig, n_classes, seed,
                       Xt, yt, wt, Xv, yv):
    """Numero di alberi scelto sulla coda di validazione, per ciascun backend."""
    if backend == "catboost":
        est = _make_estimator(backend, cfg, n_classes, seed)
        est.fit(Xt, yt, sample_weight=wt, eval_set=(Xv, yv),
                early_stopping_rounds=_ES_ROUNDS, verbose=False)
        return int(est.get_best_iteration() or 0) + 1

    if backend == "xgboost":
        from xgboost import XGBClassifier
        params = _make_estimator(backend, cfg, n_classes, seed).get_params()
        params["early_stopping_rounds"] = _ES_ROUNDS
        est = XGBClassifier(**params)
        est.fit(Xt, yt, sample_weight=wt, eval_set=[(Xv, yv)], verbose=False)
        return int(getattr(est, "best_iteration", 0) or 0) + 1

    if backend == "lightgbm":
        import lightgbm as lgb
        est = _make_estimator(backend, cfg, n_classes, seed)
        est.fit(Xt, yt, sample_weight=wt, eval_set=[(Xv, yv)],
                callbacks=[lgb.early_stopping(_ES_ROUNDS, verbose=False)])
        return int(getattr(est, "best_iteration_", 0) or 0)

    # sklearn: early stopping interno, con validazione sulla coda temporale
    from sklearn.ensemble import HistGradientBoostingClassifier
    est = HistGradientBoostingClassifier(
        max_iter=cfg.m1_n_estimators, learning_rate=cfg.m1_learning_rate,
        max_depth=cfg.m1_max_depth, min_samples_leaf=cfg.m1_min_samples,
        l2_regularization=cfg.m1_l2, early_stopping=True,
        validation_fraction=_VAL_FRACTION, n_iter_no_change=_ES_ROUNDS,
        random_state=seed)
    est.fit(np.vstack([Xt, Xv]), np.concatenate([yt, yv]))
    return int(est.n_iter_)


class GBMModel(BaseModel):
    """M1 — gradient boosting multi-mercato."""

    name = "M1_gbm"

    def __init__(self, cfg: ModelConfig = None, seed=42, with_market=False,
                 half_life_days=720.0):
        super().__init__(cfg or ModelConfig())
        self.seed = seed
        self.with_market = with_market
        self.half_life_days = half_life_days
        self.backend_ = None
        self.columns_ = None
        self.models_ = {}
        self.classes_ = {}
        self.priors_ = {}

    # ── training ──────────────────────────────────────────────────────────────
    def fit(self, df, feats):
        self.backend_ = resolve_backend(self.cfg.m1_backend)
        self.columns_ = feature_columns(with_market=self.with_market)
        X, _ = prepare_matrix(feats, self.columns_)
        X = X.to_numpy(dtype=float)

        w = time_decay_weights(df["kickoff"], half_life_days=self.half_life_days)
        self.models_, self.classes_, self.priors_ = {}, {}, {}

        targets = {"1x2": ("y_1x2", 3), "over25": ("y_over25", 2), "btts": ("y_btts", 2)}
        for market, (col, n_classes) in targets.items():
            if col not in df.columns:
                continue
            y = df[col].to_numpy(dtype=float)
            mask = np.isfinite(y)
            if mask.sum() < 100 or len(np.unique(y[mask])) < 2:
                continue

            yi = y[mask].astype(int)
            self.priors_[market] = np.bincount(yi, minlength=n_classes) / len(yi)

            est = self._fit_head(X[mask], yi, np.nan_to_num(w[mask], nan=1.0), n_classes)
            self.models_[market] = est
            self.classes_[market] = np.asarray(est.classes_ if hasattr(est, "classes_")
                                               else np.arange(n_classes))

        self.fitted_ = bool(self.models_)
        return self

    def _fit_head(self, X, y, w, n_classes):
        """
        Addestra una testa con early stopping temporale.

        Il numero di alberi viene scelto su una coda di validazione (le partite
        più recenti del training), poi il modello viene riaddestrato su tutti i
        dati con quel numero: le partite più recenti sono le più informative e
        buttarle via per la validazione costa più di quanto rende.
        """
        n = len(y)
        n_val = int(n * _VAL_FRACTION)
        if n_val < 150 or len(np.unique(y[-n_val:])) < 2:
            est = _make_estimator(self.backend_, self.cfg, n_classes, self.seed)
            est.fit(X, y, sample_weight=w)
            return est

        Xt, yt, wt = X[:-n_val], y[:-n_val], w[:-n_val]
        Xv, yv = X[-n_val:], y[-n_val:]
        best = None
        try:
            best = _early_stop_rounds(self.backend_, self.cfg, n_classes, self.seed,
                                      Xt, yt, wt, Xv, yv)
        except Exception as e:
            print(f"  [M1] early stopping non disponibile ({e}) → numero di alberi fisso")

        cfg = self.cfg
        if best:
            cfg = replace(self.cfg, m1_n_estimators=int(np.clip(best, 40, cfg.m1_n_estimators)))
        est = _make_estimator(self.backend_, cfg, n_classes, self.seed)
        est.fit(X, y, sample_weight=w)
        return est

    # ── inferenza ─────────────────────────────────────────────────────────────
    def _predict_market(self, market, X, n_classes):
        est = self.models_.get(market)
        prior = self.priors_.get(market)
        if est is None:
            base = prior if prior is not None else np.full(n_classes, 1.0 / n_classes)
            return np.tile(base, (len(X), 1))

        proba = est.predict_proba(X)
        classes = self.classes_.get(market, np.arange(proba.shape[1]))
        if proba.shape[1] != n_classes:
            # una classe assente nel training: la si reinserisce con probabilità ~0
            full = np.zeros((len(X), n_classes))
            for i, c in enumerate(classes):
                full[:, int(c)] = proba[:, i]
            proba = np.clip(full, 1e-6, None)
            proba = proba / proba.sum(axis=1, keepdims=True)
        return proba

    def predict_proba(self, df, feats):
        n = len(df)
        if not self.fitted_:
            from .base import empty_predictions
            return empty_predictions(n)

        X, _ = prepare_matrix(feats, self.columns_)
        X = X.to_numpy(dtype=float)

        p_1x2 = normalize_1x2(self._predict_market("1x2", X, 3))
        p_o25 = self._predict_market("over25", X, 2)[:, 1]
        p_btts = self._predict_market("btts", X, 2)[:, 1]
        return {"1x2": p_1x2, "over25": p_o25, "btts": p_btts}

    # ── diagnostica ───────────────────────────────────────────────────────────
    def feature_importance(self, market="1x2", top=20):
        """Importanza delle feature, quando il backend la espone."""
        est = self.models_.get(market)
        if est is None:
            return None
        imp = getattr(est, "feature_importances_", None)
        if imp is None:
            return None
        import pandas as pd
        return (pd.DataFrame({"feature": self.columns_, "importance": imp})
                  .sort_values("importance", ascending=False)
                  .head(top)
                  .reset_index(drop=True))
