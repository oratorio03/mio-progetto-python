"""
evaluate.py — Metriche, backtest walk-forward e valutazione economica.

Il backtest è walk-forward puro: ad ogni passo il modello viene riaddestrato
solo sulle partite precedenti alla finestra da prevedere. Nessuna riga di test
ha mai influenzato i parametri che la predicono.
"""

import numpy as np
import pandas as pd

from . import features as F
from .models import HomeWinEnsemble, log_loss_safe

THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70)


# ── metriche ──────────────────────────────────────────────────────────────────

def auc_score(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # media dei ranghi per i valori a pari merito
    df = pd.DataFrame({"p": p, "r": ranks})
    ranks = df.groupby("p")["r"].transform("mean").to_numpy()
    n_pos, n_neg = pos.sum(), neg.sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def metrics(y, p, label=""):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    base = float(y.mean())
    out = {
        "label":     label,
        "n":         int(len(y)),
        "base_rate": base,
        "log_loss":  log_loss_safe(y, p),
        "brier":     float(np.mean((p - y) ** 2)),
        "auc":       auc_score(y, p),
        "accuracy":  float(np.mean((p >= 0.5) == (y == 1))),
        "mean_pred": float(p.mean()),
    }
    # Riferimento "climatologia": prevedere sempre la base rate DEL CAMPIONE
    # STESSO. È un oracolo — quel numero non è noto prima delle partite — quindi
    # va letto come tetto pessimistico, non come avversario onesto. Il confronto
    # onesto è con p_prior (base rate del training), riga a parte in report().
    out["log_loss_baseline"] = log_loss_safe(y, np.full(len(y), base))
    out["skill"] = 1.0 - out["log_loss"] / out["log_loss_baseline"] \
        if out["log_loss_baseline"] > 0 else np.nan
    return out


def threshold_table(y, p, thresholds=THRESHOLDS):
    """Quante segnalazioni sopra soglia e con che hit rate."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    rows = []
    for t in thresholds:
        m = p >= t
        rows.append({
            "soglia":   t,
            "n":        int(m.sum()),
            "quota_%":  round(100.0 * m.mean(), 1) if len(m) else 0.0,
            "hit_%":    round(100.0 * y[m].mean(), 1) if m.sum() else np.nan,
            "atteso_%": round(100.0 * p[m].mean(), 1) if m.sum() else np.nan,
        })
    return pd.DataFrame(rows)


def calibration_table(y, p, bins=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({
            "bin":       f"{edges[b]:.1f}–{edges[b+1]:.1f}",
            "n":         int(m.sum()),
            "p_medio":   round(float(p[m].mean()), 3),
            "osservato": round(float(y[m].mean()), 3),
            "gap":       round(float(y[m].mean() - p[m].mean()), 3),
        })
    return pd.DataFrame(rows)


# ── valutazione economica ─────────────────────────────────────────────────────

def value_bets(df, prob_col="p_ensemble", odds_col="q1", target_col="target",
               min_edge=0.05, min_prob=0.0, stake=1.0):
    """
    Puntata piatta su Home Win quando prob * quota - 1 >= min_edge.
    Ritorna (riepilogo, dataframe delle giocate).
    """
    d = df[df[odds_col].notna() & df[prob_col].notna()].copy()
    if target_col in d.columns:
        d = d[d[target_col].notna()]
    if d.empty:
        return {"n_bets": 0}, d

    d["edge"] = d[prob_col] * d[odds_col] - 1.0
    d["implied"] = 1.0 / d[odds_col]
    bets = d[(d["edge"] >= min_edge) & (d[prob_col] >= min_prob)].copy()
    if bets.empty:
        return {"n_bets": 0, "n_candidati": int(len(d))}, bets

    bets["pnl"] = np.where(bets[target_col] == 1,
                           stake * (bets[odds_col] - 1.0), -stake)
    staked = stake * len(bets)
    summary = {
        "n_candidati": int(len(d)),
        "n_bets":      int(len(bets)),
        "hit_%":       round(100.0 * float(bets[target_col].mean()), 1),
        "quota_media": round(float(bets[odds_col].mean()), 2),
        "edge_medio":  round(float(bets["edge"].mean()), 3),
        "profitto":    round(float(bets["pnl"].sum()), 2),
        "roi_%":       round(100.0 * float(bets["pnl"].sum()) / staked, 2),
    }
    return summary, bets


# ── backtest walk-forward ─────────────────────────────────────────────────────

def walk_forward(results, min_train=600, step_days=30, start_date=None,
                 n_states=3, verbose=True, max_steps=None):
    """
    Riaddestra e prevede a finestre successive.

    min_train  : partite minime prima di iniziare a prevedere
    step_days  : ampiezza in giorni della finestra prevista ad ogni passo
    start_date : data da cui iniziare a prevedere (default: dopo min_train)

    Ritorna un DataFrame con le predizioni out-of-sample di ogni passo.
    """
    res = results.copy()
    res["date"] = pd.to_datetime(res["date"])
    res = res[(res.get("played", 1) == 1) &
              res["home_goals"].notna() & res["away_goals"].notna()]
    res = res.sort_values("date").reset_index(drop=True)

    table = F.build_feature_table(res)
    table = table[table["target"].notna()].reset_index(drop=True)

    if start_date is None:
        if len(table) <= min_train:
            raise ValueError(f"dati insufficienti: {len(table)} partite, "
                             f"min_train={min_train}")
        start = pd.Timestamp(table["date"].iloc[min_train])
    else:
        start = pd.Timestamp(start_date)

    end = pd.Timestamp(table["date"].max())
    step = pd.Timedelta(days=step_days)

    chunks = []
    cursor = start
    n_step = 0
    while cursor <= end:
        stop = cursor + step
        test = table[(table["date"] >= cursor) & (table["date"] < stop)]
        train = table[table["date"] < cursor]
        if len(test) == 0 or len(train) < min_train:
            cursor = stop
            continue

        res_train = res[res["date"] < cursor]
        try:
            ens = HomeWinEnsemble(n_states=n_states, verbose=False).fit(
                res_train, table=train)
            # I PARAMETRI dell'HMM vengono solo da res_train. Qui passiamo tutte
            # le partite perché il filtraggio taglia per data (searchsorted su
            # date < data partita): per prevedere la giornata a metà finestra si
            # usano i risultati delle giornate precedenti della stessa finestra,
            # esattamente come fanno le feature rolling e come accade in
            # produzione. Congelare lo storico a inizio finestra non sarebbe più
            # prudente: sarebbe solo incoerente fra i tre modelli.
            ens.prepare_history(res)
            detail = ens.predict_proba(test, detail=True)
        except Exception as exc:
            if verbose:
                print(f"  [{cursor.date()}] passo saltato: {exc}")
            cursor = stop
            continue

        out = test.reset_index(drop=True).copy()
        for col in detail.columns:
            out[col] = detail[col].to_numpy()
        out["fold_start"] = cursor
        # baseline onesta: la frequenza di vittorie casalinghe nota al momento
        # dell'addestramento, l'unica cosa che si potrebbe davvero scommettere
        out["p_prior"] = float(train["target"].mean())
        chunks.append(out)
        n_step += 1
        if verbose:
            print(f"  [{cursor.date()}] train {len(train):5d} → test {len(test):4d} "
                  f"| log-loss {log_loss_safe(out['target'], out['p_ensemble']):.4f}")
        cursor = stop
        if max_steps and n_step >= max_steps:
            break

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def report(preds, prob_cols=("p_ensemble", "p_logistic", "p_hmm", "p_tree",
                             "p_prior")):
    """Riepilogo metriche per l'ensemble, per ogni modello base e per il prior."""
    y = preds["target"].to_numpy(dtype=float)
    rows = [metrics(y, preds[c].to_numpy(dtype=float), c)
            for c in prob_cols if c in preds.columns]
    if "imp_h" in preds.columns and preds["imp_h"].notna().any():
        m = preds["imp_h"].notna()
        rows.append(metrics(y[m.to_numpy()],
                            preds.loc[m, "imp_h"].to_numpy(dtype=float),
                            "bookmaker (imp_h)"))
    return pd.DataFrame(rows)
