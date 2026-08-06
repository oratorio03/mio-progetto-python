"""
base_models.py — Regressione logistica e albero decisionale.

Entrambi lavorano sulla stessa matrice di feature; l'HMM (hmm.py) è il terzo
membro dell'ensemble e lavora invece sulle sequenze temporali.

Gli iperparametri si scelgono con TimeSeriesSplit: split cronologici, mai
casuali. Su dati temporali una CV mescolata addestrerebbe su partite
successive a quelle di test, e i risultati sarebbero fantasia.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .features import FEATURES
from .util import log_loss


def _select(build, grid, X, y, n_splits=3, scale=False):
    """Iperparametro con la log-loss media più bassa in CV temporale."""
    if len(y) < 200 or len(np.unique(y)) < 2:
        return grid[len(grid) // 2]
    splitter = TimeSeriesSplit(n_splits=max(2, min(n_splits, len(y) // 100)))
    best, best_loss = grid[0], np.inf
    for candidate in grid:
        losses = []
        for train_idx, test_idx in splitter.split(X):
            if len(np.unique(y[train_idx])) < 2:
                continue
            X_tr, X_te = X[train_idx], X[test_idx]
            if scale:
                scaler = StandardScaler().fit(X_tr)
                X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
            try:
                model = build(candidate).fit(X_tr, y[train_idx])
                losses.append(log_loss(y[test_idx], model.predict_proba(X_te)[:, 1]))
            except Exception:
                continue
        if losses and np.mean(losses) < best_loss:
            best_loss, best = float(np.mean(losses)), candidate
    return best


class LogisticModel:
    """
    Logistica L2 su feature standardizzate.

    È il modello che sbaglia poco per volta: raramente il migliore, quasi mai
    il peggiore. Serve da spina dorsale dell'ensemble.
    """

    GRID = (0.02, 0.05, 0.1, 0.3, 1.0, 3.0)
    name = "logistica"

    def __init__(self, random_state=42, C=None):
        self.random_state = random_state
        self.C = C
        self.scaler = None
        self.model = None

    def _build(self, C):
        return LogisticRegression(C=C, solver="lbfgs", max_iter=2000,
                                  random_state=self.random_state)

    def select(self, X, y):
        self.C = _select(self._build, self.GRID, X, y, scale=True)
        return self

    def fit(self, X, y):
        if self.C is None:
            self.select(X, y)
        self.scaler = StandardScaler().fit(X)
        self.model = self._build(self.C).fit(self.scaler.transform(X), y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]

    def coefficients(self):
        return (pd.Series(self.model.coef_[0], index=FEATURES)
                .sort_values(key=np.abs, ascending=False))


class TreeModel:
    """
    Albero singolo, potato e poco profondo di proposito.

    Serve a catturare interazioni a soglia — "Elo alto E ospite che perde
    sempre fuori casa" — che la logistica da sola non vede. Se lo si lascia
    crescere impara i nomi delle squadre e smette di generalizzare.
    """

    GRID = tuple({"max_depth": depth, "min_samples_leaf": leaf, "ccp_alpha": alpha}
                 for depth in (3, 4, 5, 6)
                 for leaf in (30, 60, 120)
                 for alpha in (0.0, 0.0005))
    name = "albero"

    def __init__(self, random_state=42, params=None):
        self.random_state = random_state
        self.params = params
        self.model = None

    def _build(self, params):
        return DecisionTreeClassifier(random_state=self.random_state,
                                      criterion="entropy", **params)

    def select(self, X, y):
        self.params = _select(self._build, self.GRID, X, y)
        return self

    def fit(self, X, y):
        if self.params is None:
            self.select(X, y)
        self.model = self._build(self.params).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def importances(self):
        return (pd.Series(self.model.feature_importances_, index=FEATURES)
                .sort_values(ascending=False))
