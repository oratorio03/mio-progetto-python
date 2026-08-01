"""
value.py — dalla probabilità alla giocata.

Tre passaggi distinti, volutamente separati:

  1. de-vig       togliere il margine del bookmaker dalle quote per ottenere
                  la probabilità "vera" implicita nel mercato
  2. filtro value confrontare la probabilità del modello con quella di mercato
                  e tenere solo gli scarti abbastanza grandi da non essere rumore
  3. staking      Kelly frazionario con cap, cioè quanto puntare

Il KPI di processo non è il ROI di breve periodo ma il Closing Line Value:
se le giocate battono sistematicamente la quota di chiusura, l'edge esiste
anche quando la varianza nasconde il profitto.
"""

import numpy as np
import pandas as pd

from .config import BettingConfig

_EPS = 1e-12


# ── De-vig ────────────────────────────────────────────────────────────────────
def implied(odds):
    """Probabilità implicite grezze (somma > 1 per via del margine)."""
    o = np.asarray(odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(o > 1.0, 1.0 / o, np.nan)
    return p


def devig_multiplicative(raw):
    """Normalizzazione proporzionale: veloce, ma sovrastima gli outsider."""
    raw = np.asarray(raw, dtype=float)
    s = np.nansum(raw)
    if not np.isfinite(s) or s <= 0:
        return np.full_like(raw, np.nan)
    return raw / s


def devig_power(raw, tol=1e-10, max_iter=100):
    """Metodo power: p_i = raw_i^k con k tale che la somma faccia 1."""
    raw = np.asarray(raw, dtype=float)
    if np.any(~np.isfinite(raw)) or np.nansum(raw) <= 0:
        return np.full_like(raw, np.nan)
    lo, hi = 0.2, 5.0
    for _ in range(max_iter):
        k = 0.5 * (lo + hi)
        s = np.sum(raw ** k)
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:
            lo = k          # somma troppo alta: serve un esponente maggiore
        else:
            hi = k
    return raw ** k / np.sum(raw ** k)


def devig_shin(raw, tol=1e-10, max_iter=200):
    """
    Metodo di Shin (1993): modella il margine come protezione contro
    gli scommettitori informati. Corregge il favourite-longshot bias
    meglio della normalizzazione proporzionale.
    """
    raw = np.asarray(raw, dtype=float)
    if np.any(~np.isfinite(raw)):
        return np.full_like(raw, np.nan)
    total = np.sum(raw)
    if total <= 0:
        return np.full_like(raw, np.nan)
    if total <= 1.0 + 1e-9:
        return devig_multiplicative(raw)

    def probs(z):
        inner = z ** 2 + 4.0 * (1.0 - z) * raw ** 2 / total
        return (np.sqrt(np.maximum(inner, 0.0)) - z) / (2.0 * (1.0 - z))

    lo, hi = 0.0, 0.9
    for _ in range(max_iter):
        z = 0.5 * (lo + hi)
        s = np.sum(probs(z))
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:
            lo = z
        else:
            hi = z
    p = probs(z)
    ssum = np.sum(p)
    return p / ssum if ssum > 0 else np.full_like(raw, np.nan)


_DEVIG = {
    "multiplicative": devig_multiplicative,
    "power":          devig_power,
    "shin":           devig_shin,
}


def devig(odds, method="shin"):
    """De-vig di un mercato completo (tutte le selezioni)."""
    raw = implied(odds)
    if np.any(~np.isfinite(raw)):
        return np.full(len(raw), np.nan)
    fn = _DEVIG.get(method, devig_shin)
    return fn(raw)


def devig_two_way(odd_yes, odd_no=None, assumed_margin=0.045, method="shin"):
    """
    De-vig di un mercato a due esiti quando spesso è nota solo una quota.

    Con entrambe le quote si usa il metodo scelto. Con la sola quota "yes"
    si scala la probabilità grezza per un margine assunto: approssimazione,
    ma neutra rispetto al confronto fra modello e mercato.
    """
    if odd_yes is None or not np.isfinite(odd_yes) or odd_yes <= 1.0:
        return np.nan
    if odd_no is not None and np.isfinite(odd_no) and odd_no > 1.0:
        return float(devig([odd_yes, odd_no], method=method)[0])
    p = 1.0 / odd_yes
    return float(np.clip(p / (1.0 + assumed_margin), _EPS, 1.0 - _EPS))


def margin_from_1x2(q1, qx, q2):
    """Margine (overround − 1) del mercato 1X2, usato per stimare quello dei mercati a due vie."""
    try:
        s = 1.0 / q1 + 1.0 / qx + 1.0 / q2
        return float(s - 1.0) if np.isfinite(s) else np.nan
    except Exception:
        return np.nan


def two_way_margin_from_3way(margin_3way, default=0.045):
    """
    Il margine su un mercato a 2 esiti è tipicamente ~2/3 di quello a 3 esiti
    (stesso margine per esito, un esito in meno).
    """
    if margin_3way is None or not np.isfinite(margin_3way) or margin_3way <= 0:
        return default
    return float(np.clip(margin_3way * 2.0 / 3.0, 0.005, 0.20))


# ── Edge e staking ────────────────────────────────────────────────────────────
def edge(p_model, odds):
    """Valore atteso per unità puntata: p·quota − 1."""
    p_model = np.asarray(p_model, dtype=float)
    odds    = np.asarray(odds, dtype=float)
    with np.errstate(invalid="ignore"):
        return p_model * odds - 1.0


def kelly(p_model, odds, fraction=1.0, cap=1.0):
    """
    Frazione di bankroll da puntare secondo Kelly, scalata e cappata.

        f* = (p·q − 1) / (q − 1)

    Negativa o nulla se non c'è vantaggio: in quel caso non si gioca.
    """
    p_model = np.asarray(p_model, dtype=float)
    odds    = np.asarray(odds, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = (p_model * odds - 1.0) / (odds - 1.0)
    f = np.where(np.isfinite(f), f, 0.0)
    f = np.clip(f, 0.0, None) * fraction
    return np.clip(f, 0.0, cap)


def clv(bet_odds, closing_odds, devig_bet=None, devig_close=None):
    """
    Closing Line Value.

    In quota:  bet/close − 1  (positivo se abbiamo preso una quota migliore
    della chiusura). Se sono note anche le probabilità de-viggate, la versione
    in probabilità è più pulita perché immune ai cambi di margine.
    """
    out = {}
    try:
        out["clv_odds"] = float(bet_odds) / float(closing_odds) - 1.0
    except Exception:
        out["clv_odds"] = np.nan
    if devig_bet is not None and devig_close is not None:
        try:
            out["clv_prob"] = float(devig_close) - float(devig_bet)
        except Exception:
            out["clv_prob"] = np.nan
    else:
        out["clv_prob"] = np.nan
    return out


# ── Selezione delle giocate ───────────────────────────────────────────────────
def select_bets(df, cfg: BettingConfig = None):
    """
    Applica filtro value e staking a un DataFrame di candidati.

    Colonne richieste: p_model, odds, market, selection, kickoff.
    Colonne opzionali: p_market (probabilità de-viggata), close_odds.

    Restituisce solo le righe giocabili, con edge, frazione di Kelly e stake.
    """
    cfg = cfg or BettingConfig()
    if df.empty:
        return df.assign(edge=[], kelly_frac=[], stake=[])

    out = df.copy()
    out["odds"]    = pd.to_numeric(out["odds"], errors="coerce")
    out["p_model"] = pd.to_numeric(out["p_model"], errors="coerce")
    out["edge"]    = edge(out["p_model"], out["odds"])
    out["kelly_frac"] = kelly(out["p_model"], out["odds"],
                              fraction=cfg.kelly_fraction, cap=cfg.max_stake_pct)

    keep = (
        out["odds"].between(cfg.min_odds, cfg.max_odds)
        & (out["p_model"] >= cfg.min_prob)
        & (out["edge"] >= cfg.min_edge)
        & (out["kelly_frac"] > 0)
    )
    out = out[keep].copy()
    if out.empty:
        return out.assign(stake=[])

    # cap giornaliero: si tengono le giocate con edge maggiore
    if cfg.max_bets_per_day and "kickoff" in out.columns:
        out["_day"] = pd.to_datetime(out["kickoff"]).dt.date
        out = (out.sort_values(["_day", "edge"], ascending=[True, False])
                  .groupby("_day", group_keys=False)
                  .head(cfg.max_bets_per_day)
                  .drop(columns="_day"))

    out["stake"] = (out["kelly_frac"] * cfg.bankroll).round(2)
    return out.sort_values(["kickoff", "edge"], ascending=[True, False])


def simulate_bankroll(bets, cfg: BettingConfig = None, compounding=True):
    """
    Simulazione sequenziale del bankroll sulle giocate risolte.

    `bets` deve avere: kickoff, odds, kelly_frac, won (0/1).
    Con compounding=True lo stake è ricalcolato sul bankroll corrente.
    """
    cfg = cfg or BettingConfig()
    if bets.empty:
        return pd.DataFrame(columns=["kickoff", "stake", "pnl", "bankroll"]), {}

    b = bets.sort_values("kickoff").copy()
    b = b[b["won"].notna()]
    if b.empty:
        return pd.DataFrame(columns=["kickoff", "stake", "pnl", "bankroll"]), {}

    bankroll = cfg.bankroll
    rows, peak, max_dd = [], bankroll, 0.0
    for r in b.itertuples(index=False):
        stake = (r.kelly_frac * bankroll) if compounding else (r.kelly_frac * cfg.bankroll)
        pnl   = stake * (r.odds - 1.0) if r.won == 1 else -stake
        bankroll += pnl
        peak = max(peak, bankroll)
        max_dd = max(max_dd, (peak - bankroll) / peak if peak > 0 else 0.0)
        rows.append({"kickoff": r.kickoff, "market": getattr(r, "market", None),
                     "odds": r.odds, "stake": stake, "won": r.won,
                     "pnl": pnl, "bankroll": bankroll})

    curve = pd.DataFrame(rows)
    staked = curve["stake"].sum()
    stats = {
        "n_bets":      len(curve),
        "hit_rate":    float(curve["won"].mean()),
        "avg_odds":    float(curve["odds"].mean()),
        "turnover":    float(staked),
        "profit":      float(curve["pnl"].sum()),
        "yield_pct":   float(curve["pnl"].sum() / staked * 100.0) if staked > 0 else np.nan,
        "roi_pct":     float((bankroll / cfg.bankroll - 1.0) * 100.0),
        "max_drawdown_pct": float(max_dd * 100.0),
        "final_bankroll":   float(bankroll),
    }
    return curve, stats
