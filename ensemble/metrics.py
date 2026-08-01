"""
metrics.py — metriche di valutazione probabilistica.

Il riferimento per il 1X2 è il Ranked Probability Score: tiene conto
dell'ordinamento naturale degli esiti (H, D, A) e penalizza meno un errore
"vicino" di uno "lontano". È la metrica usata nella Soccer Prediction Challenge.

Per i mercati binari si usano log loss e Brier score; l'ECE misura quanto le
probabilità dichiarate corrispondono alle frequenze osservate.
"""

import numpy as np
import pandas as pd

_EPS = 1e-15


def _as_prob_matrix(p):
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = np.column_stack([1.0 - p, p])
    return p


def rps(probs, y, n_classes=None):
    """
    Ranked Probability Score medio (0 = perfetto).

    probs: matrice (n, k) con le classi in ordine naturale
    y:     indici di classe osservati
    """
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    if probs.size == 0:
        return np.nan
    k = n_classes or probs.shape[1]
    obs = np.zeros_like(probs)
    obs[np.arange(len(y)), y] = 1.0
    cum_p = np.cumsum(probs, axis=1)[:, :-1]
    cum_o = np.cumsum(obs, axis=1)[:, :-1]
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / (k - 1)))


def log_loss(probs, y):
    """Log loss (entropia incrociata) media."""
    probs = _as_prob_matrix(probs)
    y = np.asarray(y, dtype=int)
    if probs.size == 0:
        return np.nan
    p = np.clip(probs[np.arange(len(y)), y], _EPS, 1.0)
    return float(-np.mean(np.log(p)))


def brier(probs, y):
    """Brier score multiclasse (per i binari coincide con l'MSE sulla classe 1)."""
    probs = _as_prob_matrix(probs)
    y = np.asarray(y, dtype=int)
    if probs.size == 0:
        return np.nan
    obs = np.zeros_like(probs)
    obs[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - obs) ** 2, axis=1)))


def accuracy(probs, y):
    probs = _as_prob_matrix(probs)
    if probs.size == 0:
        return np.nan
    return float(np.mean(np.argmax(probs, axis=1) == np.asarray(y, dtype=int)))


def ece(p_binary, y, n_bins=10):
    """
    Expected Calibration Error su un mercato binario.

    Media pesata dello scarto |probabilità dichiarata − frequenza osservata|
    dentro bin di probabilità.
    """
    p = np.asarray(p_binary, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    p, y = p[mask], y[mask]
    if len(p) == 0:
        return np.nan
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        sel = idx == b
        if not np.any(sel):
            continue
        total += np.sum(sel) / len(p) * abs(p[sel].mean() - y[sel].mean())
    return float(total)


def reliability_table(p_binary, y, n_bins=10):
    """Tabella di affidabilità: previsto vs osservato per bin di probabilità."""
    p = np.asarray(p_binary, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    p, y = p[mask], y[mask]
    if len(p) == 0:
        return pd.DataFrame(columns=["bin", "n", "p_medio", "freq_osservata", "scarto"])
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        sel = idx == b
        if not np.any(sel):
            continue
        rows.append({
            "bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
            "n": int(sel.sum()),
            "p_medio": float(p[sel].mean()),
            "freq_osservata": float(y[sel].mean()),
            "scarto": float(y[sel].mean() - p[sel].mean()),
        })
    return pd.DataFrame(rows)


def evaluate(probs, y, market="1x2", n_bins=10):
    """Pacchetto di metriche per un mercato."""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    if probs.ndim == 1:
        mask &= np.isfinite(probs)
        probs = probs[mask]
    else:
        mask &= np.isfinite(probs).all(axis=1)
        probs = probs[mask]
    y = y[mask].astype(int)
    if len(y) == 0:
        return {"n": 0}

    out = {
        "n": int(len(y)),
        "log_loss": log_loss(probs, y),
        "brier": brier(probs, y),
        "accuracy": accuracy(probs, y),
    }
    if market == "1x2":
        out["rps"] = rps(probs, y, n_classes=3)
        out["ece"] = ece(probs[:, 0], (y == 0).astype(float), n_bins)
    else:
        p1 = probs if probs.ndim == 1 else probs[:, 1]
        out["rps"] = rps(np.column_stack([1 - p1, p1]), y, n_classes=2)
        out["ece"] = ece(p1, y, n_bins)
    return out


def compare(results: dict, market="1x2"):
    """Confronto tabellare fra modelli (dict nome → (probs, y))."""
    rows = []
    for name, (probs, y) in results.items():
        m = evaluate(probs, y, market=market)
        m["modello"] = name
        rows.append(m)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    cols = ["modello", "n", "rps", "log_loss", "brier", "accuracy", "ece"]
    return df[[c for c in cols if c in df.columns]].sort_values("rps")
