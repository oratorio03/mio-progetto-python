"""
pipeline.py — orchestrazione: dati → feature → modelli → giocate.

Tre funzioni principali:

  build_all      carica i CSV e calcola una volta sola rating e feature
  walk_forward   backtest onesto: riaddestra periodicamente e predice solo
                 partite successive al training
  run_predict    previsioni sulle fixture future e lista di giocate value

I rating vengono calcolati in un unico passaggio cronologico su tutto lo
storico: ogni riga usa solo le partite precedenti, quindi possono essere
riusati in tutte le finestre di backtest senza reintrodurre look-ahead.
"""

import numpy as np
import pandas as pd

from .config import (MARKETS, MARKET_TARGET_COLS, OUTCOME_1X2, EnsembleConfig,
                     MODEL_DIR, OUTPUT_DIR)
from .data import load_dataset
from .features import build_features
from .metrics import compare, evaluate
from .models.baseline import DixonColesBaseline
from .stacking import StackedEnsemble, market_probs
from .value import (devig, devig_two_way, margin_from_1x2, select_bets,
                    simulate_bankroll, two_way_margin_from_3way)

PRED_ID_COLS = ["fixture_id", "kickoff", "nation", "league_name", "tier",
                "home_name", "away_name", "home_id", "away_id"]


# ── dati e feature ────────────────────────────────────────────────────────────
def build_all(nations=None, root=None, cfg: EnsembleConfig = None,
              include_fixtures=True, verbose=True):
    """Carica tutto e calcola le feature. Restituisce (df, feats, engine)."""
    cfg = cfg or EnsembleConfig()
    if verbose:
        print("\n  Caricamento dati…")
    df = load_dataset(nations, root=root, include_fixtures=include_fixtures,
                      verbose=verbose)
    if df.empty:
        return df, pd.DataFrame(), None

    if verbose:
        print(f"  Totale: {len(df):,} righe  "
              f"({int((df['played'] == 1).sum()):,} giocate)")
        print("  Calcolo rating e feature…")
    feats, engine = build_features(df, cfg)
    return df, feats, engine


# ── raccolta predizioni ───────────────────────────────────────────────────────
def _rows_from_probs(df, probs, model_name):
    """Trasforma il dizionario di probabilità in righe tabellari."""
    p1x2 = np.asarray(probs.get("1x2"))
    out = pd.DataFrame({
        "model": model_name,
        "p_h": p1x2[:, 0], "p_d": p1x2[:, 1], "p_a": p1x2[:, 2],
        "p_o25": np.asarray(probs.get("over25")).reshape(-1),
        "p_btts": np.asarray(probs.get("btts")).reshape(-1),
    }, index=df.index)

    for col in PRED_ID_COLS:
        out[col] = df[col].to_numpy() if col in df.columns else np.nan
    for col in ["y_1x2", "y_over25", "y_btts", "home_goals", "away_goals"]:
        out[col] = df[col].to_numpy() if col in df.columns else np.nan
    for col in ["q1", "qx", "q2", "odd_o25", "odd_btts", "odd_1x",
                "close_q1", "close_qx", "close_q2", "close_odd_o25", "close_odd_btts"]:
        out[col] = df[col].to_numpy() if col in df.columns else np.nan
    return out


def walk_forward(df, feats, cfg: EnsembleConfig = None, start=None, end=None,
                 include_baseline=True, include_base_models=True, verbose=True):
    """
    Backtest walk-forward.

    Il periodo [start, end] viene diviso in finestre di `refit_days` giorni.
    Per ogni finestra si addestra su tutte le partite precedenti e si predice
    la finestra. Nessun dato della finestra entra nel training.
    """
    cfg = cfg or EnsembleConfig()
    played = df[df["played"] == 1].copy()
    if played.empty:
        return pd.DataFrame()

    start = pd.to_datetime(start) if start else played["kickoff"].quantile(0.6)
    end = pd.to_datetime(end) if end else played["kickoff"].max()

    windows = []
    cur = pd.Timestamp(start)
    step = pd.Timedelta(days=max(int(cfg.stack.refit_days), 1))
    while cur <= end:
        windows.append((cur, min(cur + step, end + pd.Timedelta(days=1))))
        cur += step

    if verbose:
        print(f"\n  Walk-forward: {start.date()} → {end.date()}  "
              f"({len(windows)} finestre da {cfg.stack.refit_days}gg)")

    all_rows = []
    for i, (w_start, w_end) in enumerate(windows, 1):
        train = played[played["kickoff"] < w_start]
        test = played[(played["kickoff"] >= w_start) & (played["kickoff"] < w_end)]
        if test.empty:
            continue
        if len(train) < cfg.stack.min_train_matches:
            if verbose:
                print(f"  [{i:>2}/{len(windows)}] {w_start.date()}  "
                      f"training insufficiente ({len(train)}) → salto")
            continue

        if verbose:
            print(f"  [{i:>2}/{len(windows)}] {w_start.date()} → {w_end.date()}  "
                  f"train {len(train):>6}  test {len(test):>4}", flush=True)

        f_train, f_test = feats.loc[train.index], feats.loc[test.index]

        ens = StackedEnsemble(cfg, verbose=False)
        ens.fit(train, f_train)
        probs, base = ens.predict_proba(test, f_test, return_base=True)

        all_rows.append(_rows_from_probs(test, probs, "ENSEMBLE"))
        if include_base_models:
            for name, pred in base.items():
                all_rows.append(_rows_from_probs(test, pred, name))

        if include_baseline:
            dc = DixonColesBaseline(cfg.models)
            dc.fit(train, f_train)
            all_rows.append(_rows_from_probs(test, dc.predict_proba(test, f_test), dc.name))

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


# ── valutazione ───────────────────────────────────────────────────────────────
def evaluate_predictions(preds, market="1x2"):
    """Tabella di confronto fra modelli su un mercato."""
    if preds.empty:
        return pd.DataFrame()
    results = {}
    for name, g in preds.groupby("model"):
        if market == "1x2":
            P = g[["p_h", "p_d", "p_a"]].to_numpy()
            y = g["y_1x2"].to_numpy()
        elif market == "over25":
            P = g["p_o25"].to_numpy()
            y = g["y_over25"].to_numpy()
        else:
            P = g["p_btts"].to_numpy()
            y = g["y_btts"].to_numpy()
        results[name] = (P, y)
    return compare(results, market=market)


def market_baseline_metrics(preds, market="1x2", devig_method="shin"):
    """Metriche della quota de-viggata: il vero avversario da battere."""
    if preds.empty:
        return {}
    g = preds[preds["model"] == "ENSEMBLE"].copy()
    if market == "1x2":
        ok = g[["q1", "qx", "q2"]].notna().all(axis=1) & g["y_1x2"].notna()
        g = g[ok]
        if g.empty:
            return {}
        P = np.vstack([devig(r, method=devig_method)
                       for r in g[["q1", "qx", "q2"]].to_numpy()])
        return evaluate(P, g["y_1x2"].to_numpy(), market="1x2")

    odds_col, y_col = (("odd_o25", "y_over25") if market == "over25"
                       else ("odd_btts", "y_btts"))
    g = g[g[odds_col].notna() & g[y_col].notna()]
    if g.empty:
        return {}
    margin3 = [margin_from_1x2(a, b, c) for a, b, c in
               g[["q1", "qx", "q2"]].to_numpy()]
    p = np.array([devig_two_way(o, assumed_margin=two_way_margin_from_3way(m))
                  for o, m in zip(g[odds_col].to_numpy(), margin3)])
    return evaluate(p, g[y_col].to_numpy(), market=market)


# ── giocate ───────────────────────────────────────────────────────────────────
def build_candidates(preds, model="ENSEMBLE", devig_method="shin"):
    """
    Da predizioni a candidati giocabili: una riga per (partita, selezione).

    Include la probabilità de-viggata del mercato, per misurare lo scarto
    modello-mercato e, quando disponibili le quote di chiusura, il CLV.
    """
    if preds.empty:
        return pd.DataFrame()
    g = preds[preds["model"] == model].copy()
    rows = []

    for r in g.itertuples(index=False):
        q1, qx, q2 = getattr(r, "q1", np.nan), getattr(r, "qx", np.nan), getattr(r, "q2", np.nan)
        m3 = margin_from_1x2(q1, qx, q2)
        p_mkt_1x2 = (devig([q1, qx, q2], method=devig_method)
                     if np.all(np.isfinite([q1, qx, q2])) else [np.nan] * 3)

        base = {
            "fixture_id": r.fixture_id, "kickoff": r.kickoff, "nation": r.nation,
            "league_name": r.league_name, "home_name": r.home_name,
            "away_name": r.away_name,
        }

        specs = [
            ("1x2", "H", getattr(r, "q1", np.nan), r.p_h, p_mkt_1x2[0],
             (r.y_1x2 == 0) if np.isfinite(r.y_1x2) else np.nan,
             getattr(r, "close_q1", np.nan)),
            ("1x2", "D", getattr(r, "qx", np.nan), r.p_d, p_mkt_1x2[1],
             (r.y_1x2 == 1) if np.isfinite(r.y_1x2) else np.nan,
             getattr(r, "close_qx", np.nan)),
            ("1x2", "A", getattr(r, "q2", np.nan), r.p_a, p_mkt_1x2[2],
             (r.y_1x2 == 2) if np.isfinite(r.y_1x2) else np.nan,
             getattr(r, "close_q2", np.nan)),
            ("over25", "Over 2.5", getattr(r, "odd_o25", np.nan), r.p_o25,
             devig_two_way(getattr(r, "odd_o25", np.nan),
                           assumed_margin=two_way_margin_from_3way(m3)),
             r.y_over25, getattr(r, "close_odd_o25", np.nan)),
            ("btts", "BTTS", getattr(r, "odd_btts", np.nan), r.p_btts,
             devig_two_way(getattr(r, "odd_btts", np.nan),
                           assumed_margin=two_way_margin_from_3way(m3)),
             r.y_btts, getattr(r, "close_odd_btts", np.nan)),
        ]

        for market, sel, odds, p_model, p_mkt, won, close_odds in specs:
            if odds is None or not np.isfinite(odds) or odds <= 1.0:
                continue
            rows.append({**base, "market": market, "selection": sel,
                         "odds": float(odds), "p_model": float(p_model),
                         "p_market": float(p_mkt) if np.isfinite(p_mkt) else np.nan,
                         "close_odds": float(close_odds) if close_odds is not None
                         and np.isfinite(close_odds) else np.nan,
                         "won": float(won) if won is not None and np.isfinite(float(won))
                         else np.nan})

    return pd.DataFrame(rows)


def add_clv(bets, devig_method="shin"):
    """Closing Line Value per le giocate con quota di chiusura disponibile."""
    if bets.empty or "close_odds" not in bets.columns:
        return bets.assign(clv_odds=np.nan, clv_prob=np.nan)
    out = bets.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["clv_odds"] = out["odds"] / out["close_odds"] - 1.0
    # In probabilità: quanto il mercato si è mosso verso di noi dopo la giocata
    out["clv_prob"] = np.where(
        out["close_odds"].notna(),
        1.0 / out["close_odds"] - 1.0 / out["odds"],
        np.nan,
    )
    return out


def run_value_analysis(preds, cfg: EnsembleConfig = None, model="ENSEMBLE"):
    """Filtro value + staking + simulazione bankroll sulle predizioni di backtest."""
    cfg = cfg or EnsembleConfig()
    cand = build_candidates(preds, model=model, devig_method=cfg.betting.devig_method)
    if cand.empty:
        return cand, pd.DataFrame(), {}
    bets = select_bets(cand, cfg.betting)
    bets = add_clv(bets, cfg.betting.devig_method)
    curve, stats = simulate_bankroll(bets, cfg.betting)
    if not bets.empty and bets["clv_prob"].notna().any():
        stats["clv_prob_medio"] = float(bets["clv_prob"].mean())
        stats["clv_positivo_pct"] = float((bets["clv_prob"] > 0).mean() * 100)
    return bets, curve, stats


# ── produzione ────────────────────────────────────────────────────────────────
def run_predict(nations=None, root=None, cfg: EnsembleConfig = None,
                date_from=None, date_to=None, verbose=True):
    """
    Addestra su tutto lo storico e predice le fixture future.

    Restituisce (fixture con probabilità, giocate value).
    """
    cfg = cfg or EnsembleConfig()
    df, feats, _ = build_all(nations, root, cfg, include_fixtures=True, verbose=verbose)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    played = df[df["played"] == 1]
    future = df[df["played"] != 1].copy()

    if date_from:
        future = future[future["kickoff"] >= pd.to_datetime(date_from)]
    if date_to:
        future = future[future["kickoff"] <= pd.to_datetime(date_to) + pd.Timedelta(days=1)]

    if future.empty:
        if verbose:
            print("  Nessuna fixture futura nel periodo richiesto")
        return pd.DataFrame(), pd.DataFrame()

    if verbose:
        print(f"\n  Training su {len(played):,} partite, "
              f"previsione di {len(future):,} fixture")

    ens = StackedEnsemble(cfg, verbose=verbose)
    ens.fit(played, feats.loc[played.index])
    probs = ens.predict_proba(future, feats.loc[future.index])

    preds = _rows_from_probs(future, probs, "ENSEMBLE")
    cand = build_candidates(preds, devig_method=cfg.betting.devig_method)
    bets = select_bets(cand, cfg.betting) if not cand.empty else cand
    return preds, bets


# ── persistenza ───────────────────────────────────────────────────────────────
def save_model(ens, path=None):
    """Salva l'ensemble addestrato (joblib)."""
    import joblib
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    path = path or (MODEL_DIR / "ensemble.joblib")
    joblib.dump(ens, path)
    return path


def load_model(path=None):
    import joblib
    path = path or (MODEL_DIR / "ensemble.joblib")
    return joblib.load(path)
