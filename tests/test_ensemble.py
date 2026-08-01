"""
test_ensemble.py — verifica della pipeline ensemble su dati sintetici.

Gira sia con pytest (`pytest tests/`) sia da solo (`python tests/test_ensemble.py`),
senza toccare i dati reali: tutto lo storico è generato da ensemble/synthetic.py,
dove le probabilità vere sono note e si può quindi verificare non solo che il
codice giri, ma che i numeri abbiano senso.

Le verifiche che contano davvero sono due:
  - assenza di look-ahead nei rating (test_ratings_sono_causali)
  - l'ensemble batte i riferimenti banali e non peggiora rispetto ai suoi
    componenti in modo grossolano (test_ensemble_batte_i_riferimenti)
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ensemble.calibration import IsotonicCalibrator                      # noqa: E402
from ensemble.config import EnsembleConfig                               # noqa: E402
from ensemble.data import load_dataset                                   # noqa: E402
from ensemble.features import build_features                             # noqa: E402
from ensemble.metrics import evaluate, rps                               # noqa: E402
from ensemble.models import BayesianHybridModel, GBMModel, GapDirectModel  # noqa: E402
from ensemble.pipeline import build_candidates, run_value_analysis, walk_forward  # noqa: E402
from ensemble.pooling import LogPoolMeta, stack_components               # noqa: E402
from ensemble.ratings import RatingEngine                                # noqa: E402
from ensemble.stacking import StackedEnsemble                            # noqa: E402
from ensemble.synthetic import generate_league, split_future             # noqa: E402
from ensemble.value import (devig, devig_two_way, kelly, select_bets,    # noqa: E402
                            simulate_bankroll)

CFG = EnsembleConfig()


# ── fixture condivise (calcolate una volta sola: generare è costoso) ───────────
_CACHE = {}


def dataset(seasons=6, seed=7):
    key = (seasons, seed)
    if key not in _CACHE:
        df = generate_league(n_seasons=seasons, n_teams=16, n_tiers=2, seed=seed)
        feats, _ = build_features(df, CFG)
        _CACHE[key] = (df, feats)
    df, feats = _CACHE[key]
    return df.copy(), feats.copy()


# ── dati ──────────────────────────────────────────────────────────────────────
def test_dataset_sintetico_coerente():
    df, feats = dataset()
    assert len(df) > 1500
    assert df["kickoff"].is_monotonic_increasing
    # i target devono corrispondere ai gol
    assert ((df["home_goals"] > df["away_goals"]) == (df["y_1x2"] == 0)).all()
    assert (((df["home_goals"] + df["away_goals"]) > 2.5) == (df["y_over25"] == 1)).all()
    assert (((df["home_goals"] > 0) & (df["away_goals"] > 0)) == (df["y_btts"] == 1)).all()
    assert feats.shape[0] == df.shape[0]
    print("  ok  dataset sintetico coerente")


def test_lettura_da_csv(tmp_root=None):
    """I CSV scritti nel formato della pipeline devono essere riletti identici."""
    tmp = Path(tmp_root or tempfile.mkdtemp())
    proc = tmp / "data" / "synth" / "processed"
    proc.mkdir(parents=True, exist_ok=True)

    df, _ = dataset()
    cols = ["fixture_id", "league_id", "league_name", "tier", "season", "round",
            "date", "time", "played", "home_id", "home_name", "away_id",
            "away_name", "home_goals", "away_goals"]
    df[cols].to_csv(proc / "all_results.csv", index=False)
    df[["fixture_id", "q1", "qx", "q2", "odd_o25", "odd_btts"]].to_csv(
        proc / "odds.csv", index=False)
    df[["fixture_id", "close_q1", "close_qx", "close_q2"]].rename(
        columns={"close_q1": "q1", "close_qx": "qx", "close_q2": "q2"}
    ).to_csv(proc / "odds_closing.csv", index=False)

    loaded = load_dataset(["synth"], root=tmp, include_fixtures=False, verbose=False)
    assert len(loaded) == len(df)
    assert loaded["q1"].notna().all()
    assert "close_q1" in loaded.columns and loaded["close_q1"].notna().all()
    print("  ok  lettura da CSV nel formato della pipeline")
    if tmp_root is None:
        shutil.rmtree(tmp, ignore_errors=True)


# ── rating ────────────────────────────────────────────────────────────────────
def test_ratings_sono_causali():
    """
    Le feature di una partita non devono cambiare se il futuro cambia.

    È il test che protegge dal look-ahead: si calcolano i rating su tutto lo
    storico e poi solo sul prefisso, e le righe comuni devono coincidere.
    """
    df, _ = dataset()
    cut = int(len(df) * 0.6)

    full = RatingEngine(CFG.ratings).transform(df)
    prefix = RatingEngine(CFG.ratings).transform(df.iloc[:cut])

    common = full.iloc[:cut]
    diff = (common - prefix).abs().max().max()
    assert diff < 1e-9, f"le feature dipendono dal futuro (scarto max {diff})"
    print("  ok  rating causali (nessun look-ahead)")


def test_ratings_recuperano_la_forza_vera():
    """
    Elo deve ordinare le squadre come la loro forza osservata.

    Il confronto si fa dentro la stessa lega: fra tier diversi la differenza
    reti non è confrontabile (una squadra forte di seconda divisione ha una
    differenza reti alta contro avversari deboli), ed è proprio l'informazione
    che l'Elo tiene separata.
    """
    df, feats = dataset()
    eng = RatingEngine(CFG.ratings)
    eng.transform(df)
    snap = eng.team_snapshot().set_index("team_id")

    last = df[df["season"] == df["season"].max()]
    gd, lega = {}, {}
    for r in last.itertuples(index=False):
        gd.setdefault(r.home_id, []).append(r.home_goals - r.away_goals)
        gd.setdefault(r.away_id, []).append(r.away_goals - r.home_goals)
        lega[r.home_id] = lega[r.away_id] = r.league_key

    obs = pd.DataFrame({
        "gd_medio": {k: np.mean(v) for k, v in gd.items() if len(v) >= 10},
        "lega": lega,
    }).dropna()
    merged = snap.join(obs, how="inner")

    corrs = {}
    for key, g in merged.groupby("lega"):
        if len(g) >= 8:
            corrs[key] = g["elo"].corr(g["gd_medio"])
    assert corrs, "nessuna lega con squadre sufficienti per il confronto"
    peggiore = min(corrs.values())
    assert peggiore > 0.6, f"correlazione Elo/forza osservata troppo bassa: {corrs}"
    print(f"  ok  Elo correlato alla forza osservata per lega "
          f"(r={ {k: round(v, 2) for k, v in corrs.items()} })")


# ── modelli base ──────────────────────────────────────────────────────────────
def test_modelli_base_battono_il_prior():
    """Ogni modello deve battere la distribuzione media del calcio (44/26/30)."""
    df, feats = dataset()
    cut = int(len(df) * 0.75)
    tr, te = df.iloc[:cut], df.iloc[cut:]
    y = te["y_1x2"].to_numpy().astype(int)

    prior = np.tile([0.44, 0.26, 0.30], (len(te), 1))
    rps_prior = rps(prior, y, 3)

    for M in (GBMModel, BayesianHybridModel, GapDirectModel):
        m = M(CFG.models)
        m.fit(tr, feats.iloc[:cut])
        assert m.fitted_, f"{m.name} non si è addestrato"
        p = m.predict_proba(te, feats.iloc[cut:])
        assert p["1x2"].shape == (len(te), 3)
        assert np.allclose(p["1x2"].sum(axis=1), 1.0, atol=1e-6)
        assert np.all((p["over25"] > 0) & (p["over25"] < 1))
        r = rps(p["1x2"], y, 3)
        assert r < rps_prior, f"{m.name}: RPS {r:.4f} non batte il prior {rps_prior:.4f}"
        print(f"  ok  {m.name}: RPS {r:.4f} < prior {rps_prior:.4f}")


def test_m2_gestisce_squadre_mai_viste():
    """
    Il modello ibrido deve produrre probabilità sensate anche per squadre
    assenti dal training: è il motivo per cui esiste il ponte sui rating.
    """
    df, feats = dataset()
    cut = int(len(df) * 0.7)
    tr = df.iloc[:cut]

    # partite di test con almeno una squadra mai vista nel training
    visti = set(tr["home_id"]) | set(tr["away_id"])
    te = df.iloc[cut:]
    nuovi = te[~te["home_id"].isin(visti) | ~te["away_id"].isin(visti)]
    if nuovi.empty:                      # con promozioni/retrocessioni è raro ma possibile
        nuovi = te.tail(20).copy()
        nuovi["home_id"] = -1            # squadra sicuramente sconosciuta

    m = BayesianHybridModel(CFG.models)
    m.fit(tr, feats.iloc[:cut])
    p = m.predict_proba(nuovi, feats.loc[nuovi.index])
    assert np.all(np.isfinite(p["1x2"]))
    assert np.allclose(p["1x2"].sum(axis=1), 1.0, atol=1e-6)
    assert np.all(p["1x2"] > 0.01), "probabilità degenerate su squadre sconosciute"
    print(f"  ok  M2 gestisce {len(nuovi)} partite con squadre sconosciute")


# ── pool e calibrazione ───────────────────────────────────────────────────────
def test_pool_pesa_i_modelli():
    """Con un modello informativo e uno casuale, il pool deve preferire il primo."""
    rng = np.random.default_rng(0)
    n = 3000
    y = rng.integers(0, 3, n)
    buono = np.full((n, 3), 0.15)
    buono[np.arange(n), y] = 0.70
    casuale = rng.dirichlet([4, 4, 4], n)

    LP = stack_components([buono, casuale], 3)
    pool = LogPoolMeta(l2=0.01).fit(LP, y)
    w = pool.weights
    assert w[0] > w[1] * 2, f"pesi non discriminanti: {w}"

    p = pool.predict_proba(LP)
    assert rps(p, y, 3) < rps(casuale, y, 3)
    print(f"  ok  pool logaritmico pesa i modelli (pesi {np.round(w, 2)})")


def test_calibrazione_corregge_lo_sbilanciamento():
    """Su probabilità sistematicamente gonfiate l'isotonica deve ridurre l'ECE."""
    rng = np.random.default_rng(1)
    n = 4000
    p_vera = rng.uniform(0.1, 0.8, n)
    y = (rng.uniform(size=n) < p_vera).astype(int)
    p_gonfia = np.clip(p_vera + 0.12, 0.01, 0.99)     # modello troppo ottimista

    cal = IsotonicCalibrator().fit(p_gonfia[:2000], y[:2000])
    p_cal = cal.transform(p_gonfia[2000:])[:, 1]

    ece_prima = evaluate(p_gonfia[2000:], y[2000:], "over25")["ece"]
    ece_dopo = evaluate(p_cal, y[2000:], "over25")["ece"]
    assert ece_dopo < ece_prima * 0.6, f"ECE {ece_prima:.3f} → {ece_dopo:.3f}"
    assert np.all(p_cal > 0), "la calibrazione non deve produrre probabilità nulle"
    print(f"  ok  isotonica: ECE {ece_prima:.3f} → {ece_dopo:.3f}")


# ── ensemble ──────────────────────────────────────────────────────────────────
def test_ensemble_batte_i_riferimenti():
    """
    L'ensemble deve battere il prior e restare vicino al miglior modello base
    (che ex ante non si conosce: è il senso stesso di combinarli).
    """
    df, feats = dataset(seasons=8)
    cut = int(len(df) * 0.8)
    tr, te = df.iloc[:cut], df.iloc[cut:]

    ens = StackedEnsemble(CFG, verbose=False)
    ens.fit(tr, feats.iloc[:cut])
    probs, base = ens.predict_proba(te, feats.iloc[cut:], return_base=True)

    y = te["y_1x2"].to_numpy().astype(int)
    rps_ens = rps(probs["1x2"], y, 3)
    rps_prior = rps(np.tile([0.44, 0.26, 0.30], (len(te), 1)), y, 3)
    rps_base = {n: rps(p["1x2"], y, 3) for n, p in base.items()}

    assert rps_ens < rps_prior
    assert rps_ens <= min(rps_base.values()) * 1.02, (
        f"ensemble {rps_ens:.4f} molto peggio del miglior base "
        f"{min(rps_base.values()):.4f}")
    for market in ("over25", "btts"):
        assert np.all((probs[market] > 0) & (probs[market] < 1))
    print(f"  ok  ensemble RPS {rps_ens:.4f} (base: "
          f"{ {k: round(v, 4) for k, v in rps_base.items()} })")


def test_walk_forward_non_usa_il_futuro():
    """Ogni predizione del backtest deve venire da un training che la precede."""
    df, feats = dataset(seasons=8)
    cfg = EnsembleConfig()
    cfg.stack.refit_days = 120
    cfg.stack.oof_folds = 3
    cfg.models.m1_n_estimators = 150

    start = df["kickoff"].quantile(0.75)
    preds = walk_forward(df, feats, cfg, start=start, include_base_models=False,
                         include_baseline=False, verbose=False)
    assert not preds.empty
    assert preds["kickoff"].min() >= start
    # ogni partita predetta una sola volta per modello
    assert not preds.duplicated(["fixture_id", "model"]).any()
    print(f"  ok  walk-forward: {len(preds)} predizioni, tutte dopo {start.date()}")


# ── value betting ─────────────────────────────────────────────────────────────
def test_devig_somma_a_uno():
    for q in ([2.0, 3.4, 4.0], [1.25, 6.0, 12.0], [1.9, 3.6, 4.2]):
        for method in ("multiplicative", "power", "shin"):
            p = devig(q, method=method)
            assert abs(p.sum() - 1.0) < 1e-6, f"{method}: somma {p.sum()}"
            assert np.all(p > 0)
    # Shin deve ridurre la probabilità dell'outsider rispetto al proporzionale
    q = [1.25, 6.0, 12.0]
    assert devig(q, "shin")[2] < devig(q, "multiplicative")[2]
    assert abs(devig_two_way(1.90, 1.95) + devig_two_way(1.95, 1.90) - 1.0) < 1e-6
    print("  ok  de-vig coerente (somma 1, Shin corregge il longshot bias)")


def test_kelly_e_filtro():
    # quota equa: nessuna puntata
    assert kelly(0.5, 2.0)[()] if False else True
    assert float(np.asarray(kelly(0.5, 2.0))) == 0.0
    # 60% su quota 2.0 → Kelly pieno 20%
    assert abs(float(np.asarray(kelly(0.6, 2.0))) - 0.2) < 1e-9
    # frazionario e cap
    assert abs(float(np.asarray(kelly(0.6, 2.0, fraction=0.25))) - 0.05) < 1e-9
    assert float(np.asarray(kelly(0.9, 5.0, fraction=1.0, cap=0.02))) == 0.02

    cand = pd.DataFrame({
        "kickoff": pd.to_datetime(["2026-01-01"] * 3),
        "market": ["1x2"] * 3, "selection": ["H", "D", "A"],
        "odds": [2.10, 1.20, 12.0], "p_model": [0.55, 0.30, 0.10],
    })
    bets = select_bets(cand, CFG.betting)
    assert list(bets["selection"]) == ["H"], "il filtro deve tenere solo la giocata value"
    assert bets["stake"].iloc[0] > 0
    print("  ok  Kelly frazionario e filtro value")


def test_simulazione_bankroll():
    rng = np.random.default_rng(3)
    n = 500
    odds = np.full(n, 2.0)
    won = (rng.uniform(size=n) < 0.55).astype(float)   # edge reale del 10%
    bets = pd.DataFrame({
        "kickoff": pd.date_range("2025-01-01", periods=n, freq="D"),
        "odds": odds, "kelly_frac": np.full(n, 0.02), "won": won, "market": "1x2",
    })
    curve, stats = simulate_bankroll(bets, CFG.betting)
    assert len(curve) == n
    assert stats["final_bankroll"] > CFG.betting.bankroll, "con edge positivo il bankroll cresce"
    assert 0 <= stats["max_drawdown_pct"] <= 100
    print(f"  ok  simulazione bankroll (finale {stats['final_bankroll']:.0f}, "
          f"drawdown {stats['max_drawdown_pct']:.1f}%)")


def test_clv_e_candidati():
    """Le giocate devono portarsi dietro quota di chiusura e CLV."""
    df, feats = dataset(seasons=8)
    cfg = EnsembleConfig()
    cfg.stack.refit_days = 150
    cfg.stack.oof_folds = 3
    cfg.models.m1_n_estimators = 150
    cfg.betting.min_edge = 0.02

    preds = walk_forward(df, feats, cfg, start=df["kickoff"].quantile(0.8),
                         include_base_models=False, include_baseline=False,
                         verbose=False)
    cand = build_candidates(preds)
    assert not cand.empty
    assert cand["p_market"].notna().any()

    bets, curve, stats = run_value_analysis(preds, cfg)
    if not bets.empty:
        assert "clv_prob" in bets.columns
        assert bets["odds"].between(cfg.betting.min_odds, cfg.betting.max_odds).all()
        assert (bets["edge"] >= cfg.betting.min_edge - 1e-9).all()
        print(f"  ok  {len(bets)} giocate con CLV calcolato "
              f"(CLV medio {stats.get('clv_prob_medio', float('nan')) * 100:+.2f} punti)")
    else:
        print("  ok  nessuna giocata oltre soglia (accettabile: mercato sintetico efficiente)")


def test_fixture_future_senza_risultato():
    """Le partite non giocate devono ricevere probabilità ma non aggiornare i rating."""
    df, _ = dataset()
    df2, truth = split_future(df, n_future=40)
    feats2, _ = build_features(df2, CFG)

    played = df2[df2["played"] == 1]
    future = df2[df2["played"] == 0]
    assert len(future) == 40

    ens = StackedEnsemble(CFG, verbose=False)
    ens.fit(played, feats2.loc[played.index])
    probs = ens.predict_proba(future, feats2.loc[future.index])
    assert np.all(np.isfinite(probs["1x2"]))
    assert np.allclose(probs["1x2"].sum(axis=1), 1.0, atol=1e-6)

    # i rating delle fixture future non devono essere aggiornati dai loro risultati
    eng_a = RatingEngine(CFG.ratings)
    eng_a.transform(df2)
    eng_b = RatingEngine(CFG.ratings)
    eng_b.transform(df2[df2["played"] == 1])
    a = eng_a.team_snapshot().set_index("team_id")["elo"]
    b = eng_b.team_snapshot().set_index("team_id")["elo"]
    assert (a - b).abs().max() < 1e-9
    print("  ok  fixture future: previste ma non usate per aggiornare i rating")


# ── runner ────────────────────────────────────────────────────────────────────
TESTS = [
    test_dataset_sintetico_coerente,
    test_lettura_da_csv,
    test_ratings_sono_causali,
    test_ratings_recuperano_la_forza_vera,
    test_modelli_base_battono_il_prior,
    test_m2_gestisce_squadre_mai_viste,
    test_pool_pesa_i_modelli,
    test_calibrazione_corregge_lo_sbilanciamento,
    test_ensemble_batte_i_riferimenti,
    test_walk_forward_non_usa_il_futuro,
    test_devig_somma_a_uno,
    test_kelly_e_filtro,
    test_simulazione_bankroll,
    test_clv_e_candidati,
    test_fixture_future_senza_risultato,
]


def main():
    print(f"\n{'=' * 70}\n  TEST ENSEMBLE — dati sintetici\n{'=' * 70}")
    falliti = []
    for t in TESTS:
        print(f"\n{t.__name__}")
        try:
            t()
        except AssertionError as e:
            falliti.append((t.__name__, str(e)))
            print(f"  FALLITO: {e}")
        except Exception as e:
            falliti.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"  ERRORE: {type(e).__name__}: {e}")

    print(f"\n{'=' * 70}")
    if falliti:
        print(f"  {len(falliti)}/{len(TESTS)} test falliti:")
        for name, err in falliti:
            print(f"    {name}: {err}")
        sys.exit(1)
    print(f"  Tutti i {len(TESTS)} test superati")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
