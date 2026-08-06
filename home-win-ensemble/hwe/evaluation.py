"""
evaluation.py — Metriche, backtest walk-forward, valutazione economica.

Il backtest è walk-forward puro: a ogni passo il modello viene riaddestrato
solo sulle partite precedenti alla finestra da prevedere. Nessuna riga di test
ha mai influenzato i parametri che la predicono.
"""

import numpy as np
import pandas as pd

from . import features as F
from .ensemble import HomeWinEnsemble
from .util import brier, log_loss

SOGLIE = (0.50, 0.55, 0.60, 0.65, 0.70)


# ── metriche ──────────────────────────────────────────────────────────────────

def auc(y, p):
    """Area sotto la ROC, con gestione dei pari merito. Nessuna dipendenza."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    positives, negatives = y == 1, y == 0
    n_pos, n_neg = positives.sum(), negatives.sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    ranks = pd.DataFrame({"p": p, "r": ranks}).groupby("p")["r"].transform("mean")
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def metrics(y, p, label=""):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    base = float(y.mean())
    # "costante" = prevedere sempre la frequenza DEL CAMPIONE STESSO: è un
    # oracolo, quel numero non si conosce prima delle partite. Va letto come
    # tetto pessimistico. Il confronto onesto è la riga p_prior del backtest.
    constant = log_loss(y, np.full(len(y), base))
    loss = log_loss(y, p)
    return {
        "modello":       label,
        "n":             int(len(y)),
        "base_rate":     round(base, 4),
        "log_loss":      round(loss, 4),
        "brier":         round(brier(y, p), 4),
        "auc":           round(auc(y, p), 4) if len(y) else np.nan,
        "accuratezza":   round(float(np.mean((p >= 0.5) == (y == 1))), 4),
        "p_media":       round(float(p.mean()), 4),
        "log_loss_cost": round(constant, 4),
        "skill":         round(1.0 - loss / constant, 4) if constant > 0 else np.nan,
    }


def report(preds, columns=("p_ensemble", "p_logistica", "p_hmm", "p_albero",
                           "p_prior")):
    y = preds["target"].to_numpy(dtype=float)
    rows = [metrics(y, preds[c].to_numpy(dtype=float), c)
            for c in columns if c in preds.columns]
    if "odds_home" in preds.columns and preds["odds_home"].notna().any():
        mask = preds["odds_home"].notna().to_numpy()
        implied = 1.0 / preds.loc[mask, "odds_home"].to_numpy(dtype=float)
        rows.append(metrics(y[mask], implied, "quota (implicita)"))
    return pd.DataFrame(rows)


def threshold_table(y, p, soglie=SOGLIE):
    """Quante segnalazioni sopra soglia, e con che percentuale di successo."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    rows = []
    for soglia in soglie:
        mask = p >= soglia
        rows.append({
            "soglia":    soglia,
            "n":         int(mask.sum()),
            "quota_%":   round(100.0 * float(mask.mean()), 1) if len(mask) else 0.0,
            "riuscite_%": round(100.0 * float(y[mask].mean()), 1) if mask.sum() else np.nan,
            "attese_%":  round(100.0 * float(p[mask].mean()), 1) if mask.sum() else np.nan,
        })
    return pd.DataFrame(rows)


def calibration_table(y, p, bins=10):
    """Quanto le probabilità dichiarate corrispondono alle frequenze osservate."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = index == b
        if not mask.any():
            continue
        rows.append({
            "fascia":    f"{edges[b]:.1f}–{edges[b+1]:.1f}",
            "n":         int(mask.sum()),
            "p_media":   round(float(p[mask].mean()), 3),
            "osservata": round(float(y[mask].mean()), 3),
            "scarto":    round(float(y[mask].mean() - p[mask].mean()), 3),
        })
    return pd.DataFrame(rows)


# ── valutazione economica ─────────────────────────────────────────────────────

def value_bets(preds, prob="p_ensemble", odds="odds_home", min_edge=0.05,
               min_prob=0.0, stake=1.0):
    """
    Puntata piatta sulla vittoria interna quando prob × quota − 1 ≥ min_edge.
    Ritorna (riepilogo, giocate).
    """
    d = preds[preds[odds].notna() & preds[prob].notna() &
              preds["target"].notna()].copy()
    if d.empty:
        return {"n_giocate": 0, "nota": "nessuna quota disponibile"}, d

    d["edge"] = d[prob] * d[odds] - 1.0
    bets = d[(d["edge"] >= min_edge) & (d[prob] >= min_prob)].copy()
    if bets.empty:
        return {"n_candidate": int(len(d)), "n_giocate": 0}, bets

    bets["risultato"] = np.where(bets["target"] == 1,
                                 stake * (bets[odds] - 1.0), -stake)
    total = stake * len(bets)
    return {
        "n_candidate":  int(len(d)),
        "n_giocate":    int(len(bets)),
        "riuscite_%":   round(100.0 * float(bets["target"].mean()), 1),
        "quota_media":  round(float(bets[odds].mean()), 2),
        "edge_medio":   round(float(bets["edge"].mean()), 3),
        "profitto":     round(float(bets["risultato"].sum()), 2),
        "roi_%":        round(100.0 * float(bets["risultato"].sum()) / total, 2),
    }, bets


# ── walk-forward ──────────────────────────────────────────────────────────────

def walk_forward(played, min_train=600, step_days=30, start=None, n_states=3,
                 max_steps=None, verbose=True):
    """
    Riaddestra e prevede a finestre successive.

    min_train : partite minime prima di iniziare a prevedere
    step_days : ampiezza in giorni della finestra prevista a ogni passo
    start     : data da cui iniziare (default: subito dopo min_train partite)
    """
    played = played.copy()
    played["date"] = pd.to_datetime(played["date"])
    played = played.sort_values("date").reset_index(drop=True)

    table = F.build(played)
    table = table[table["target"].notna()].reset_index(drop=True)

    if start is None:
        if len(table) <= min_train:
            raise ValueError(f"dati insufficienti: {len(table)} partite con "
                             f"risultato, min_train={min_train}")
        cursor = pd.Timestamp(table["date"].iloc[min_train])
    else:
        cursor = pd.Timestamp(start)

    last = pd.Timestamp(table["date"].max())
    step = pd.Timedelta(days=step_days)
    chunks = []

    while cursor <= last:
        stop = cursor + step
        test = table[(table["date"] >= cursor) & (table["date"] < stop)]
        train = table[table["date"] < cursor]
        if test.empty or len(train) < min_train:
            cursor = stop
            continue

        past = played[played["date"] < cursor]
        try:
            ens = HomeWinEnsemble(n_states=n_states, verbose=False).fit(
                past, table=train)
            ens.set_history(played)      # taglio per data: nessuna sbirciata
            detail = ens.predict_proba(test, detail=True)
        except Exception as error:
            if verbose:
                print(f"  [{cursor.date()}] passo saltato: {error}")
            cursor = stop
            continue

        fold = test.reset_index(drop=True).copy()
        for column in detail.columns:
            fold[column] = detail[column].to_numpy()
        fold["fold"] = cursor
        # baseline onesta: la frequenza di vittorie casalinghe nota al momento
        # dell'addestramento, l'unica cosa che si potrebbe davvero puntare
        fold["p_prior"] = float(train["target"].mean())
        chunks.append(fold)

        if verbose:
            print(f"  [{cursor.date()}] training {len(train):5d} → test "
                  f"{len(test):4d} | log-loss "
                  f"{log_loss(fold['target'], fold['p_ensemble']):.4f}")

        cursor = stop
        if max_steps and len(chunks) >= max_steps:
            break

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
