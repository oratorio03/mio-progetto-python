"""
base.py — interfaccia comune dei modelli base dell'ensemble.

Ogni modello riceve lo stesso input (df + matrice di feature) e restituisce
probabilità per tutti e tre i mercati:

    {"1x2": array (n, 3) in ordine H/D/A,
     "over25": array (n,),
     "btts":   array (n,)}

Questo permette allo stacking di trattarli in modo intercambiabile e di
aggiungerne di nuovi senza toccare il resto della pipeline.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ..config import MARKETS

_EPS = 1e-9


def time_decay_weights(kickoff, reference=None, half_life_days=540.0):
    """
    Pesi con decadimento esponenziale: le partite vecchie contano meno.

    Con half_life_days = 540 una partita di un anno e mezzo fa pesa metà
    di una di oggi. Serve a inseguire i cambi di rosa e di allenatore senza
    buttare via lo storico.
    """
    k = pd.to_datetime(pd.Series(kickoff))
    ref = pd.to_datetime(reference) if reference is not None else k.max()
    age_days = (ref - k).dt.total_seconds() / 86400.0
    age_days = age_days.clip(lower=0.0)
    if not half_life_days or half_life_days <= 0:
        return np.ones(len(k))
    return np.power(0.5, age_days / half_life_days).to_numpy()


def uniform_1x2(n, league_rate=None):
    """Fallback: distribuzione 1X2 media del calcio europeo."""
    base = np.array([0.44, 0.26, 0.30])
    return np.tile(base, (n, 1))


def empty_predictions(n):
    return {
        "1x2":    uniform_1x2(n),
        "over25": np.full(n, 0.52),
        "btts":   np.full(n, 0.50),
    }


def normalize_1x2(p):
    p = np.clip(np.asarray(p, dtype=float), _EPS, None)
    return p / p.sum(axis=1, keepdims=True)


class BaseModel(ABC):
    """Contratto minimo di un modello base."""

    name = "base"
    markets = tuple(MARKETS)

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.fitted_ = False

    @abstractmethod
    def fit(self, df, feats):
        """Addestra sul passato. `df` e `feats` devono avere lo stesso indice."""

    @abstractmethod
    def predict_proba(self, df, feats):
        """Restituisce il dizionario di probabilità per i tre mercati."""

    def fit_predict(self, df_train, feats_train, df_test, feats_test):
        self.fit(df_train, feats_train)
        return self.predict_proba(df_test, feats_test)

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name} fitted={self.fitted_}>"


def score_matrix_to_markets(matrix):
    """
    Da matrice dei punteggi (P[i, j] = prob. di i gol casa e j gol trasferta)
    alle probabilità dei tre mercati.
    """
    m = np.asarray(matrix, dtype=float)
    n = m.shape[0]
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]

    p_home = float(m[i > j].sum())
    p_draw = float(np.trace(m))
    p_away = float(m[i < j].sum())

    p_over25 = float(m[(i + j) > 2].sum())
    p_btts = float(m[1:, 1:].sum())

    p = np.array([p_home, p_draw, p_away], dtype=float)
    s = p.sum()
    if s > 0:
        p = p / s
    return {
        "1x2": p,
        "over25": float(np.clip(p_over25, _EPS, 1 - _EPS)),
        "btts": float(np.clip(p_btts, _EPS, 1 - _EPS)),
    }


def poisson_score_matrix(lam_home, lam_away, max_goals=12, rho=0.0):
    """
    Matrice dei punteggi Poisson con correzione Dixon-Coles sui risultati bassi.

    La correzione tau interviene solo su 0-0, 0-1, 1-0, 1-1, dove il Poisson
    indipendente sbaglia sistematicamente (troppi pochi pareggi bassi).
    """
    lam_home = max(float(lam_home), 1e-4)
    lam_away = max(float(lam_away), 1e-4)
    k = np.arange(max_goals)
    log_fact = np.cumsum(np.log(np.maximum(k, 1)))
    log_ph = k * np.log(lam_home) - lam_home - log_fact
    log_pa = k * np.log(lam_away) - lam_away - log_fact
    m = np.exp(log_ph)[:, None] * np.exp(log_pa)[None, :]

    if rho:
        tau = np.ones_like(m)
        tau[0, 0] = 1.0 - lam_home * lam_away * rho
        tau[0, 1] = 1.0 + lam_home * rho
        tau[1, 0] = 1.0 + lam_away * rho
        tau[1, 1] = 1.0 - rho
        m = m * np.clip(tau, 1e-6, None)

    s = m.sum()
    return m / s if s > 0 else m
