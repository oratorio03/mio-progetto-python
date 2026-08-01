"""
m2_bayes.py — M2: modello bayesiano ibrido con rating dinamici.

L'idea (stessa famiglia del modello "Dolores") è tenere insieme due fonti
di informazione che di solito stanno in modelli separati:

  1. effetti squadra — attacco e difesa stimati dai gol, con shrinkage
     bayesiano verso la media della lega. La penalizzazione L2 su una
     regressione di Poisson è esattamente un prior gaussiano centrato sulla
     media: le squadre con pochi dati vengono tirate verso il valore medio
     invece di produrre stime rumorose.

  2. rating dinamici — Elo, pi-ratings e GAP entrano come covariate continue.
     Sono questi a far funzionare il modello su squadre mai viste (neopromosse,
     leghe nuove, primo turno di stagione): senza dummy di squadra il modello
     ricade sui rating e resta sensato, mentre un Dixon-Coles puro non ha
     nulla da dire.

Sopra i due lambda si applica la correzione Dixon-Coles sui punteggi bassi
(rho stimato per massima verosimiglianza pesata) e si passa alla matrice dei
punteggi, da cui derivano tutti i mercati in modo coerente.

Peso temporale: decadimento esponenziale con emivita configurabile.
"""

import numpy as np
import pandas as pd
from scipy import sparse

from .base import (BaseModel, empty_predictions, poisson_score_matrix,
                   score_matrix_to_markets, time_decay_weights)
from ..config import ModelConfig

# Covariate continue usate come "ponte" verso le squadre senza storico.
RATING_COVARIATES = ["elo_diff_norm", "pi_exp_gd", "gap_log_home", "gap_log_away",
                     "league_avg_goals", "reliability"]

# Le covariate vengono scalate per essere penalizzate molto meno delle dummy
# di squadra: coef -> coef/scale, penalità -> alpha/scale^2.
_COVAR_SCALE = 10.0


class BayesianHybridModel(BaseModel):
    """M2 — Poisson gerarchico con rating dinamici e correzione Dixon-Coles."""

    name = "M2_bayes"

    def __init__(self, cfg: ModelConfig = None):
        super().__init__(cfg or ModelConfig())
        self.team_index_ = {}
        self.league_index_ = {}
        self.model_ = None
        self.rho_ = 0.0
        self.n_teams_ = 0
        self.covariates_ = list(RATING_COVARIATES)
        self.covar_mean_ = None
        self.covar_std_ = None
        self.fallback_lambda_ = (1.45, 1.15)

    # ── costruzione del design ────────────────────────────────────────────────
    def _design(self, df, feats, fit=False):
        """
        Due righe per partita: gol della casa e gol della trasferta.

        Colonne: [attacco per squadra] [difesa per squadra] [intercetta lega]
                 [indicatore casa] [covariate di rating]
        Le squadre sconosciute non attivano alcuna dummy: restano sulla media
        di lega corretta dalle covariate.
        """
        n = len(df)
        nt = self.n_teams_
        nl = len(self.league_index_)

        home_idx = df["home_id"].map(self.team_index_).to_numpy()
        away_idx = df["away_id"].map(self.team_index_).to_numpy()
        lg_idx = df["league_key"].map(self.league_index_).to_numpy()

        rows, cols, vals = [], [], []

        def add(row, col, val=1.0):
            rows.append(row)
            cols.append(col)
            vals.append(val)

        for i in range(n):
            r_home, r_away = 2 * i, 2 * i + 1
            hi, ai = home_idx[i], away_idx[i]
            # attacco della squadra che segna, difesa dell'avversaria
            if hi == hi and not pd.isna(hi):
                add(r_home, int(hi))
                add(r_away, nt + int(hi))
            if ai == ai and not pd.isna(ai):
                add(r_away, int(ai))
                add(r_home, nt + int(ai))
            if lg_idx[i] == lg_idx[i] and not pd.isna(lg_idx[i]):
                add(r_home, 2 * nt + int(lg_idx[i]))
                add(r_away, 2 * nt + int(lg_idx[i]))
            add(r_home, 2 * nt + nl)          # indicatore "gioca in casa"

        base_cols = 2 * nt + nl + 1
        X = sparse.csr_matrix((vals, (rows, cols)), shape=(2 * n, base_cols))
        if not self.covariates_:
            return X

        C = feats.reindex(columns=self.covariates_).to_numpy(dtype=float)
        C = np.where(np.isfinite(C), C, np.nan)
        if fit:
            self.covar_mean_ = np.nanmean(C, axis=0)
            self.covar_std_ = np.nanstd(C, axis=0)
            self.covar_std_ = np.where(self.covar_std_ > 1e-8, self.covar_std_, 1.0)
        C = np.where(np.isfinite(C), C, self.covar_mean_)
        C = (C - self.covar_mean_) / self.covar_std_

        # Le covariate agiscono con segno opposto sulle due righe: descrivono
        # il differenziale casa/trasferta, non un livello assoluto.
        Cfull = np.zeros((2 * n, len(self.covariates_)))
        Cfull[0::2, :] = C * _COVAR_SCALE
        Cfull[1::2, :] = -C * _COVAR_SCALE

        X = sparse.hstack([X, sparse.csr_matrix(Cfull)], format="csr")
        return X

    # ── training ──────────────────────────────────────────────────────────────
    def fit(self, df, feats):
        played = df["home_goals"].notna() & df["away_goals"].notna()
        d = df[played]
        f = feats.loc[d.index]

        if len(d) < self.cfg.m2_min_matches:
            self.fitted_ = False
            return self

        teams = pd.unique(pd.concat([d["home_id"], d["away_id"]]))
        self.team_index_ = {t: i for i, t in enumerate(teams)}
        self.n_teams_ = len(teams)
        self.league_index_ = {k: i for i, k in enumerate(pd.unique(d["league_key"]))}

        X = self._design(d, f, fit=True)
        y = np.empty(2 * len(d))
        y[0::2] = d["home_goals"].to_numpy(dtype=float)
        y[1::2] = d["away_goals"].to_numpy(dtype=float)

        w_match = time_decay_weights(d["kickoff"], half_life_days=self.cfg.m2_half_life_days)
        w = np.repeat(w_match, 2)

        try:
            from sklearn.linear_model import PoissonRegressor
            model = PoissonRegressor(alpha=self.cfg.m2_alpha, fit_intercept=True,
                                     max_iter=300, tol=1e-6)
            model.fit(X, y, sample_weight=w)
            self.model_ = model
        except Exception as e:
            print(f"  [M2] fit fallito ({e}) → il modello resta disattivato")
            self.fitted_ = False
            return self

        self.fallback_lambda_ = (
            float(np.average(d["home_goals"], weights=w_match)),
            float(np.average(d["away_goals"], weights=w_match)),
        )

        lam = self._lambdas(d, f)
        self.rho_ = self._fit_rho(lam, d, w_match)
        self.fitted_ = True
        return self

    def _lambdas(self, df, feats):
        X = self._design(df, feats, fit=False)
        pred = self.model_.predict(X)
        lam_home = np.clip(pred[0::2], 0.05, 8.0)
        lam_away = np.clip(pred[1::2], 0.05, 8.0)
        return lam_home, lam_away

    def _fit_rho(self, lam, df, weights):
        """
        Stima rho della correzione Dixon-Coles per massima verosimiglianza pesata,
        su griglia. Solo i punteggi bassi contribuiscono davvero.
        """
        lam_home, lam_away = lam
        hg = df["home_goals"].to_numpy(dtype=float)
        ag = df["away_goals"].to_numpy(dtype=float)
        low = (hg <= 1) & (ag <= 1)
        if low.sum() < 100:
            return 0.0

        lh, la = lam_home[low], lam_away[low]
        h, a, w = hg[low].astype(int), ag[low].astype(int), weights[low]

        best_rho, best_ll = 0.0, -np.inf
        for rho in self.cfg.m2_rho_grid:
            tau = np.ones(len(h))
            tau = np.where((h == 0) & (a == 0), 1.0 - lh * la * rho, tau)
            tau = np.where((h == 0) & (a == 1), 1.0 + lh * rho, tau)
            tau = np.where((h == 1) & (a == 0), 1.0 + la * rho, tau)
            tau = np.where((h == 1) & (a == 1), 1.0 - rho, tau)
            if np.any(tau <= 0):
                continue
            ll = float(np.sum(w * np.log(tau)))
            if ll > best_ll:
                best_ll, best_rho = ll, rho
        return best_rho

    # ── inferenza ─────────────────────────────────────────────────────────────
    def predict_proba(self, df, feats):
        n = len(df)
        if not self.fitted_ or n == 0:
            return empty_predictions(n)

        lam_home, lam_away = self._lambdas(df, feats)

        p_1x2 = np.empty((n, 3))
        p_o25 = np.empty(n)
        p_btts = np.empty(n)
        for i in range(n):
            m = poisson_score_matrix(lam_home[i], lam_away[i],
                                     max_goals=self.cfg.m2_max_goals, rho=self.rho_)
            mk = score_matrix_to_markets(m)
            p_1x2[i] = mk["1x2"]
            p_o25[i] = mk["over25"]
            p_btts[i] = mk["btts"]

        return {"1x2": p_1x2, "over25": p_o25, "btts": p_btts,
                "lambda_home": lam_home, "lambda_away": lam_away}

    def expected_goals(self, df, feats):
        """Gol attesi secondo M2 — confrontabili con i lambda di Dixon-Coles."""
        if not self.fitted_:
            return np.full(len(df), np.nan), np.full(len(df), np.nan)
        return self._lambdas(df, feats)
