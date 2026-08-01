"""
9_ensemble.py — ensemble M1/M2/M3 come alternativa a Poisson/Dixon-Coles.

Si aggancia alla pipeline esistente: legge gli stessi CSV prodotti dagli
script 1-7 (all_results.csv, all_fixtures.csv, odds.csv) e scrive in output/.
Non tocca 8_predict.py: i due sistemi possono girare in parallelo e essere
confrontati sullo stesso periodo.

Comandi:

    python 9_ensemble.py demo
        Autotest su campionati sintetici: verifica che tutta la catena giri
        e stampa il confronto fra modelli. Non serve alcun dato reale.

    python 9_ensemble.py backtest --start=2025-08-01 --end=2026-03-01
        Backtest walk-forward: riaddestra ogni 28 giorni e predice solo
        partite successive. Confronta ensemble, modelli base, Dixon-Coles
        e quota de-viggata; poi applica filtro value e Kelly frazionario.

    python 9_ensemble.py predict --from=2026-03-01 --to=2026-03-08
        Previsioni sulle fixture future e lista di giocate value.

    python 9_ensemble.py ratings --nations=italy,england
        Fotografia dei rating correnti (Elo, pi-ratings, GAP).

Opzioni comuni:
    --nations=it,en     limita le nazioni (default: tutte quelle in config/)
    --root=/path        radice alternativa del progetto (cartelle config/ e data/)
    --refit=28          giorni fra un riaddestramento e l'altro nel backtest
    --min-edge=0.03     edge minimo per giocare
    --kelly=0.25        frazione di Kelly
    --bankroll=1000     bankroll iniziale della simulazione
    --market-features   include le quote fra le componenti dell'ensemble
    --fast              backtest più rapido (meno fold, finestre più larghe)
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from ensemble.config import MARKETS, OUTPUT_DIR, EnsembleConfig  # noqa: E402
from ensemble.metrics import compare, evaluate                    # noqa: E402
from ensemble.pipeline import (build_all, evaluate_predictions,   # noqa: E402
                               market_baseline_metrics, run_predict,
                               run_value_analysis, save_model, walk_forward)

MARKET_LABEL = {"1x2": "1X2", "over25": "OVER 2.5", "btts": "BTTS"}


# ── argomenti ─────────────────────────────────────────────────────────────────
def parse_args(argv):
    cmd = argv[0] if argv and not argv[0].startswith("-") else "demo"
    params, flags = {}, set()
    for a in argv:
        if a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
            params[k] = v
        elif a.startswith("--"):
            flags.add(a[2:])
    return cmd, params, flags


def build_config(params, flags):
    cfg = EnsembleConfig()
    if "refit" in params:
        cfg.stack.refit_days = int(params["refit"])
    if "min-edge" in params:
        cfg.betting.min_edge = float(params["min-edge"])
    if "kelly" in params:
        cfg.betting.kelly_fraction = float(params["kelly"])
    if "bankroll" in params:
        cfg.betting.bankroll = float(params["bankroll"])
    if "devig" in params:
        cfg.betting.devig_method = params["devig"]
    if "max-odds" in params:
        cfg.betting.max_odds = float(params["max-odds"])
    if "market-features" in flags:
        cfg.stack.use_market_features = True
    if "no-calibration" in flags:
        cfg.stack.calibration = "none"
    if "fast" in flags:
        cfg.stack.oof_folds = 3
        cfg.stack.refit_days = max(cfg.stack.refit_days, 56)
        cfg.models.m1_n_estimators = 250
    return cfg


def root_from(params):
    """Radice alternativa del progetto (utile per test e dataset separati)."""
    return Path(params["root"]).resolve() if "root" in params else None


def output_dir(params):
    """Cartella di output: sotto la radice indicata, altrimenti output/ del progetto."""
    root = root_from(params)
    return (root / "output") if root else OUTPUT_DIR


def nations_from(params):
    if "nations" not in params:
        return None
    return [n.strip() for n in params["nations"].split(",") if n.strip()]


# ── stampa ────────────────────────────────────────────────────────────────────
def header(title):
    print(f"\n{'=' * 78}")
    print(f"  {title}")
    print(f"{'=' * 78}")


def print_metrics(preds, cfg):
    """Confronto per mercato: ensemble vs modelli base vs Dixon-Coles vs mercato."""
    for market in MARKETS:
        tab = evaluate_predictions(preds, market)
        if tab.empty:
            continue
        mkt = market_baseline_metrics(preds, market, cfg.betting.devig_method)
        if mkt:
            mkt = {**mkt, "modello": "QUOTA DE-VIGGATA"}
            tab = pd.concat([tab, pd.DataFrame([mkt])], ignore_index=True)
            tab = tab.sort_values("rps")

        print(f"\n  {MARKET_LABEL[market]} — più basso è meglio (RPS è la metrica di riferimento)")
        print(f"  {'-' * 74}")
        print(f"  {'modello':<18} {'n':>6} {'RPS':>8} {'logloss':>9} "
              f"{'Brier':>8} {'acc':>7} {'ECE':>7}")
        for r in tab.itertuples(index=False):
            print(f"  {r.modello:<18} {int(r.n):>6} {r.rps:>8.4f} {r.log_loss:>9.4f} "
                  f"{r.brier:>8.4f} {r.accuracy * 100:>6.1f}% {r.ece:>7.4f}")


def print_betting(bets, stats, cfg):
    if bets.empty:
        print("\n  Nessuna giocata supera il filtro value "
              f"(edge minimo {cfg.betting.min_edge:.1%}).")
        return

    header("GIOCATE VALUE — filtro + Kelly frazionario")
    print(f"  Giocate            {stats.get('n_bets', 0):,}")
    print(f"  Hit rate           {stats.get('hit_rate', float('nan')) * 100:.1f}%")
    print(f"  Quota media        {stats.get('avg_odds', float('nan')):.2f}")
    print(f"  Turnover           {stats.get('turnover', 0):,.0f}")
    print(f"  Profitto           {stats.get('profit', 0):,.0f}")
    print(f"  Yield              {stats.get('yield_pct', float('nan')):.2f}%")
    print(f"  Bankroll finale    {stats.get('final_bankroll', 0):,.0f} "
          f"(da {cfg.betting.bankroll:,.0f})")
    print(f"  Max drawdown       {stats.get('max_drawdown_pct', float('nan')):.1f}%")
    if "clv_prob_medio" in stats:
        print(f"\n  CLV medio (prob.)  {stats['clv_prob_medio'] * 100:+.2f} punti")
        print(f"  Giocate con CLV+   {stats['clv_positivo_pct']:.1f}%")
        print("  Il CLV è il KPI di processo: se è positivo l'edge è reale,")
        print("  anche quando il profitto di breve periodo è negativo.")
    else:
        print("\n  CLV non calcolabile: manca odds_closing.csv "
              "(quote di chiusura) nei dati.")

    print(f"\n  Per mercato:")
    print(f"  {'mercato':<10} {'n':>5} {'hit%':>7} {'quota':>7} {'yield%':>8}")
    for market, g in bets.groupby("market"):
        res = g[g["won"].notna()]
        if res.empty:
            continue
        stake = res["kelly_frac"] * cfg.betting.bankroll
        pnl = np.where(res["won"] == 1, stake * (res["odds"] - 1), -stake)
        print(f"  {MARKET_LABEL.get(market, market):<10} {len(res):>5} "
              f"{res['won'].mean() * 100:>6.1f}% {res['odds'].mean():>7.2f} "
              f"{pnl.sum() / stake.sum() * 100:>7.2f}%")


# ── comandi ───────────────────────────────────────────────────────────────────
def cmd_demo(params, flags):
    """Autotest end-to-end su campionati sintetici."""
    from ensemble.features import build_features
    from ensemble.stacking import StackedEnsemble
    from ensemble.synthetic import generate_league
    from ensemble.value import devig

    header("DEMO — campionati sintetici (nessun dato reale richiesto)")
    seasons = int(params.get("seasons", 8))
    cfg = build_config(params, flags)

    print("  Generazione dello storico…")
    df = generate_league(n_seasons=seasons, n_teams=18, n_tiers=3,
                         seed=int(params.get("seed", 7)))
    print(f"  {len(df):,} partite, {df['home_id'].nunique()} squadre, "
          f"media gol {df['total_goals'].mean():.2f}")

    feats, _ = build_features(df, cfg)
    cut = int(len(df) * 0.8)
    tr, te = df.iloc[:cut], df.iloc[cut:]

    print(f"\n  Training su {len(tr):,} partite, test su {len(te):,}")
    ens = StackedEnsemble(cfg, verbose=True)
    ens.fit(tr, feats.iloc[:cut])
    probs, base = ens.predict_proba(te, feats.iloc[cut:], return_base=True)

    ycols = {"1x2": "y_1x2", "over25": "y_over25", "btts": "y_btts"}
    for market in MARKETS:
        y = te[ycols[market]].to_numpy()
        res = {name: (p[market], y) for name, p in base.items()}
        res["ENSEMBLE"] = (probs[market], y)
        if market == "1x2":
            res["QUOTA DE-VIGGATA"] = (
                np.vstack([devig(r) for r in te[["q1", "qx", "q2"]].to_numpy()]), y)
            res["PROB. VERE"] = (te[["_p_h", "_p_d", "_p_a"]].to_numpy(), y)
        else:
            res["PROB. VERE"] = (te["_p_" + ("o25" if market == "over25" else "btts")]
                                 .to_numpy(), y)
        print(f"\n  {MARKET_LABEL[market]}")
        print(compare(res, market).to_string(index=False))

    print("\n  Pesi stimati dall'ensemble:")
    print(ens.weights_report().to_string(index=False))
    print("\n  'PROB. VERE' sono le probabilità con cui i risultati sono stati")
    print("  generati: nessun modello può fare meglio, è il limite teorico.")


def cmd_backtest(params, flags):
    cfg = build_config(params, flags)
    nations = nations_from(params)

    header("BACKTEST WALK-FORWARD — ensemble M1/M2/M3")
    df, feats, _ = build_all(nations, root=root_from(params), cfg=cfg,
                             include_fixtures=False)
    if df.empty:
        print("  Nessun dato trovato in data/*/processed/ — "
              "lancia prima gli script 1-7 della pipeline.")
        return

    n_played = int((df["played"] == 1).sum())
    if n_played > 20000:
        print(f"\n  Attenzione: {n_played:,} partite in archivio. Ogni finestra "
              f"riaddestra tre modelli più i fold out-of-fold,\n"
              f"  quindi il backtest completo può richiedere ore. Per un giro "
              f"veloce: --fast --refit=56 --nations=...")

    preds = walk_forward(df, feats, cfg,
                         start=params.get("start"), end=params.get("end"))
    if preds.empty:
        print("  Nessuna predizione prodotta: periodo troppo corto o "
              "storico insufficiente.")
        return

    header("QUALITÀ DELLE PROBABILITÀ")
    print_metrics(preds, cfg)

    bets, curve, stats = run_value_analysis(preds, cfg)
    print_betting(bets, stats, cfg)

    out_dir = output_dir(params)
    out_dir.mkdir(exist_ok=True, parents=True)
    tag = f"{preds['kickoff'].min().date()}_{preds['kickoff'].max().date()}"
    p1 = out_dir / f"ensemble_backtest_preds_{tag}.csv"
    preds.to_csv(p1, index=False)
    print(f"\n[SALVATO] {p1}")
    if not bets.empty:
        p2 = out_dir / f"ensemble_backtest_bets_{tag}.csv"
        bets.to_csv(p2, index=False)
        print(f"[SALVATO] {p2}")
    if not curve.empty:
        p3 = out_dir / f"ensemble_backtest_bankroll_{tag}.csv"
        curve.to_csv(p3, index=False)
        print(f"[SALVATO] {p3}")


def cmd_predict(params, flags):
    cfg = build_config(params, flags)
    nations = nations_from(params)

    header("PREVISIONI ENSEMBLE — fixture future")
    preds, bets = run_predict(nations, root=root_from(params), cfg=cfg,
                              date_from=params.get("from"),
                              date_to=params.get("to"))
    if preds.empty:
        return

    view = preds[["kickoff", "nation", "league_name", "home_name", "away_name",
                  "p_h", "p_d", "p_a", "p_o25", "p_btts"]].copy()
    for c in ["p_h", "p_d", "p_a", "p_o25", "p_btts"]:
        view[c] = (view[c] * 100).round(1)
    print(f"\n  {len(view)} fixture previste (prime 15):")
    print(view.head(15).to_string(index=False))

    out_dir = output_dir(params)
    out_dir.mkdir(exist_ok=True, parents=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    p1 = out_dir / f"ensemble_predictions_{stamp}.csv"
    preds.to_csv(p1, index=False)
    print(f"\n[SALVATO] {p1}")

    if bets.empty:
        print(f"\n  Nessuna giocata supera il filtro value "
              f"(edge ≥ {cfg.betting.min_edge:.1%}).")
        return

    header("GIOCATE VALUE")
    show = bets[["kickoff", "home_name", "away_name", "market", "selection",
                 "odds", "p_model", "p_market", "edge", "stake"]].copy()
    show["odds"] = show["odds"].round(2)
    show["p_model"] = (show["p_model"] * 100).round(1)
    show["p_market"] = (show["p_market"] * 100).round(1)
    show["edge"] = (show["edge"] * 100).round(1)
    show = show.rename(columns={"p_model": "p_modello%", "p_market": "p_mercato%",
                                "edge": "edge%"})
    print(show.to_string(index=False))
    p2 = out_dir / f"ensemble_bets_{stamp}.csv"
    bets.to_csv(p2, index=False)
    print(f"\n[SALVATO] {p2}")
    print(f"  Stake calcolati su bankroll {cfg.betting.bankroll:,.0f} "
          f"con Kelly {cfg.betting.kelly_fraction:.0%}.")


def cmd_ratings(params, flags):
    cfg = build_config(params, flags)
    header("RATING CORRENTI")
    df, feats, engine = build_all(nations_from(params), root=root_from(params),
                                  cfg=cfg, include_fixtures=False)
    if df.empty or engine is None:
        print("  Nessun dato disponibile.")
        return

    snap = engine.team_snapshot()
    names = (df.melt(id_vars=[], value_vars=["home_id", "away_id"], value_name="team_id")
             .drop_duplicates("team_id"))
    name_map = pd.concat([
        df[["home_id", "home_name"]].rename(columns={"home_id": "team_id",
                                                     "home_name": "team_name"}),
        df[["away_id", "away_name"]].rename(columns={"away_id": "team_id",
                                                     "away_name": "team_name"}),
    ]).drop_duplicates("team_id")
    snap = snap.merge(name_map, on="team_id", how="left")
    snap = snap[snap["n_matches"] >= 10].sort_values("elo", ascending=False)

    print(f"\n  Top 20 per Elo ({len(snap)} squadre con almeno 10 partite):")
    cols = ["team_name", "league_key", "n_matches", "elo", "pi_h", "pi_a",
            "gap_att_h", "gap_def_h"]
    print(snap[cols].head(20).round(3).to_string(index=False))

    out_dir = output_dir(params)
    out_dir.mkdir(exist_ok=True, parents=True)
    path = out_dir / "ensemble_ratings.csv"
    snap.to_csv(path, index=False)
    print(f"\n[SALVATO] {path}")


def cmd_train(params, flags):
    from ensemble.stacking import StackedEnsemble

    cfg = build_config(params, flags)
    header("TRAINING ENSEMBLE")
    df, feats, _ = build_all(nations_from(params), root=root_from(params),
                             cfg=cfg, include_fixtures=False)
    if df.empty:
        print("  Nessun dato disponibile.")
        return
    played = df[df["played"] == 1]
    ens = StackedEnsemble(cfg, verbose=True)
    ens.fit(played, feats.loc[played.index])
    print("\n  Pesi stimati:")
    print(ens.weights_report().to_string(index=False))
    path = save_model(ens)
    print(f"\n[SALVATO] {path}")


COMMANDS = {
    "demo": cmd_demo,
    "backtest": cmd_backtest,
    "predict": cmd_predict,
    "ratings": cmd_ratings,
    "train": cmd_train,
}


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("--help", "-h", "help"):
        print(__doc__)
        return
    cmd, params, flags = parse_args(argv)
    fn = COMMANDS.get(cmd)
    if fn is None:
        print(f"Comando sconosciuto: {cmd}\n")
        print(__doc__)
        sys.exit(1)
    fn(params, flags)


if __name__ == "__main__":
    main()
