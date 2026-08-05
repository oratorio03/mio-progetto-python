"""
test_ensemble.py — Verifica end-to-end su dati sintetici.

Non servono i CSV reali: generiamo un campionato finto con forza di squadra +
uno stato di forma latente markoviano (esattamente la struttura che l'HMM
dovrebbe recuperare), lo scriviamo nel layout data/{nation}/processed/ dentro
una cartella temporanea e ci facciamo girare sopra l'intera pipeline.

Uso:
    python tests/test_ensemble.py
    pytest tests/test_ensemble.py -q
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# la root del progetto va fissata PRIMA di importare ensemble.paths
_TMP = Path(tempfile.mkdtemp(prefix="ensemble_test_"))
os.environ["BETPRO_ROOT"] = str(_TMP)

from ensemble import features as F          # noqa: E402
from ensemble import evaluate as E          # noqa: E402
from ensemble import paths as P             # noqa: E402
from ensemble.hmm import FormHMM, HMMHomeModel, encode_symbol   # noqa: E402
from ensemble.models import HomeWinEnsemble, optimise_weights   # noqa: E402


# ── generatore di dati sintetici ──────────────────────────────────────────────

N_TEAMS   = 20
SEASONS   = (2023, 2024, 2025)
HOME_ADV  = 0.30
FORM_A    = np.array([[0.80, 0.18, 0.02],
                      [0.12, 0.76, 0.12],
                      [0.02, 0.18, 0.80]])
FORM_SHIFT = np.array([-0.30, 0.0, 0.30])


def _round_robin(teams, rng):
    """Calendario andata/ritorno con ordine randomizzato."""
    fixtures = []
    for i, h in enumerate(teams):
        for j, a in enumerate(teams):
            if i != j:
                fixtures.append((h, a))
    rng.shuffle(fixtures)
    return fixtures


def make_synthetic(nation="testland", seed=7):
    rng = np.random.default_rng(seed)
    teams = [(1000 + i, f"Team {i:02d}") for i in range(N_TEAMS)]
    att = {tid: rng.normal(0, 0.32) for tid, _ in teams}
    dfn = {tid: rng.normal(0, 0.26) for tid, _ in teams}
    form = {tid: rng.integers(0, 3) for tid, _ in teams}

    rows = []
    fid = 1
    day = pd.Timestamp("2023-08-05")
    for season in SEASONS:
        for tid, _ in teams:                     # la forma si rimescola in estate
            form[tid] = rng.integers(0, 3)
        fixtures = _round_robin(teams, rng)
        per_day = max(1, N_TEAMS // 2)
        for k in range(0, len(fixtures), per_day):
            batch = fixtures[k:k + per_day]
            for (hid, hname), (aid, aname) in [((h[0], h[1]), (a[0], a[1]))
                                               for h, a in batch]:
                lam_h = np.exp(0.15 + HOME_ADV + att[hid] + FORM_SHIFT[form[hid]]
                               - dfn[aid])
                lam_a = np.exp(0.15 + att[aid] + FORM_SHIFT[form[aid]] - dfn[hid])
                hg = int(rng.poisson(lam_h))
                ag = int(rng.poisson(lam_a))
                rows.append({
                    "fixture_id": fid, "league_id": 1, "league_name": "Test Lega",
                    "tier": 1, "season": season, "round": f"R{k//per_day+1}",
                    "date": day.strftime("%Y-%m-%d"), "time": "15:00",
                    "status": "FT", "played": 1,
                    "home_id": hid, "home_name": hname,
                    "away_id": aid, "away_name": aname,
                    "home_goals": hg, "away_goals": ag,
                    "home_ht": hg // 2, "away_ht": ag // 2,
                    "home_2h": hg - hg // 2, "away_2h": ag - ag // 2,
                    "total_goals": hg + ag,
                    "result_1x2": "1" if hg > ag else ("X" if hg == ag else "2"),
                    "over15": int(hg + ag > 1.5), "over25": int(hg + ag > 2.5),
                    "over35": int(hg + ag > 3.5),
                    "btts": int(hg > 0 and ag > 0),
                    "true_p_home": np.nan,
                })
                fid += 1
            for tid, _ in teams:                 # la forma evolve fra i turni
                form[tid] = rng.choice(3, p=FORM_A[form[tid]])
            day += pd.Timedelta(days=7)
        day += pd.Timedelta(days=45)             # pausa estiva

    df = pd.DataFrame(rows).drop(columns=["true_p_home"])

    # ultime 40 partite → fixture "future" (senza risultato)
    played = df.iloc[:-40].copy()
    future = df.iloc[-40:].copy()
    truth  = future[["fixture_id", "home_goals", "away_goals"]].copy()
    future = future.assign(played=0, status="NS",
                           home_goals=np.nan, away_goals=np.nan,
                           home_ht=np.nan, away_ht=np.nan,
                           home_2h=np.nan, away_2h=np.nan,
                           total_goals=np.nan, result_1x2=np.nan,
                           over15=np.nan, over25=np.nan, over35=np.nan,
                           btts=np.nan)

    proc = P.data_dir(nation)
    proc.mkdir(parents=True, exist_ok=True)
    played.to_csv(proc / "all_results.csv", index=False)
    future.to_csv(proc / "all_fixtures.csv", index=False)

    # quote sintetiche: probabilità empirica rumorosa + margine 6%
    rng2 = np.random.default_rng(99)
    p_noisy = np.clip(rng2.normal(0.45, 0.12, len(future)), 0.12, 0.80)
    pd.DataFrame({
        "fixture_id": future["fixture_id"].to_numpy(),
        "q1": np.round(1.0 / (p_noisy * 1.06), 2),
        "qx": 3.4, "q2": np.round(1.0 / ((0.85 - p_noisy) * 1.06), 2),
        "imp_h": np.round(p_noisy * 1.06, 4),
        "bookmaker": "Test",
    }).to_csv(proc / "odds.csv", index=False)

    (P.ROOT / "config").mkdir(exist_ok=True)
    (P.ROOT / "config" / f"{nation}.json").write_text(
        '{"nation": "Testland", "leagues": [{"id": 1, "name": "Test Lega"}]}')
    return played, future, truth


# ── test ──────────────────────────────────────────────────────────────────────

def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok — {msg}")


def test_features_are_causal():
    """La feature riga i non deve cambiare se cancello il futuro dopo i."""
    print("\n[1] feature strettamente causali")
    played, _, _ = make_synthetic("testland")
    full = F.build_feature_table(played)
    trunc = F.build_feature_table(played.iloc[:400])
    n = len(trunc)
    a = full.iloc[:n][F.FEATURE_COLS].to_numpy(dtype=float)
    b = trunc[F.FEATURE_COLS].to_numpy(dtype=float)
    same = np.allclose(np.nan_to_num(a, nan=-999), np.nan_to_num(b, nan=-999))
    _check(same, "le prime 400 righe sono identiche con e senza dati futuri")

    first = full.iloc[0]
    _check(pd.isna(first["h2h_home_rate"]) and first["matches_played_home"] == 0,
           "la prima partita non ha storia (nessun valore inventato)")
    _check(full["target"].notna().all(), "target presente su tutte le partite giocate")


def test_hmm_recovers_form():
    """L'HMM deve ritrovare stati ordinati per forza crescente."""
    print("\n[2] HMM sugli stati di forma")
    played, _, _ = make_synthetic("testland")
    hmm = FormHMM(n_states=3, n_restarts=2, n_iter=25).fit(
        [(v["venues"], v["symbols"])
         for v in __import__("ensemble.hmm", fromlist=["x"])
         .build_sequences(played).values()])

    grid = np.arange(5)
    strength = (hmm.B[:, 0, :] @ grid + hmm.B[:, 1, :] @ grid) / 2.0
    _check(np.all(np.diff(strength) > 0),
           f"stati ordinati per forza crescente: {np.round(strength, 2)}")
    _check(np.mean(np.diag(hmm.A)) > 0.35,
           f"transizioni persistenti (diag media {np.mean(np.diag(hmm.A)):.2f})")

    p_win_home = hmm.B[:, 0, 3:].sum(axis=1)
    p_win_away = hmm.B[:, 1, 3:].sum(axis=1)
    _check(np.all(p_win_home > p_win_away),
           "in ogni stato si vince più in casa che fuori (vantaggio campo appreso)")
    _check(np.isfinite(hmm.loglik_), f"log-likelihood finita ({hmm.loglik_:.0f})")

    _check(encode_symbol(3, 0) == 4 and encode_symbol(1, 1) == 2
           and encode_symbol(0, 1) == 1, "codifica simboli coerente")


def test_ensemble_beats_baseline():
    """L'ensemble deve battere la base rate e non peggiorare i modelli base."""
    print("\n[3] ensemble su hold-out temporale")
    played, _, _ = make_synthetic("testland")
    table = F.build_feature_table(played)
    table = table[table["target"].notna()].reset_index(drop=True)

    cut = int(len(table) * 0.75)
    split_date = table["date"].iloc[cut]
    train_tab, test_tab = table.iloc[:cut], table.iloc[cut:]
    res_train = played[pd.to_datetime(played["date"]) < split_date]

    ens = HomeWinEnsemble(verbose=True).fit(res_train, table=train_tab)
    ens.prepare_history(res_train)
    detail = ens.predict_proba(test_tab, detail=True)
    y = test_tab["target"].to_numpy(dtype=float)

    rep = E.report(detail.assign(target=y))
    print(rep.round(4).to_string(index=False))

    ll_ens = E.log_loss_safe(y, detail["p_ensemble"])
    ll_base = E.log_loss_safe(y, np.full(len(y), float(y.mean())))
    _check(ll_ens < ll_base,
           f"log-loss ensemble {ll_ens:.4f} < base rate {ll_base:.4f}")

    worst = max(E.log_loss_safe(y, detail[f"p_{n}"]) for n in ("logistic", "hmm", "tree"))
    _check(ll_ens <= worst,
           f"ensemble non peggiore del peggior modello base ({worst:.4f})")

    auc = E.auc_score(y, detail["p_ensemble"])
    _check(auc > 0.55, f"AUC {auc:.3f} sopra il caso")

    _check(abs(ens.weights.sum() - 1.0) < 1e-6 and (ens.weights >= 0).all(),
           f"pesi validi e normalizzati: {np.round(ens.weights, 3)}")

    cal = E.calibration_table(y, detail["p_ensemble"].to_numpy())
    big = cal[cal["n"] >= 30]
    if len(big):
        _check(big["gap"].abs().max() < 0.20,
               f"calibrazione entro 20pp nei bin popolati (max {big['gap'].abs().max():.3f})")


def test_weight_optimiser():
    """Con un modello perfetto e uno casuale il peso deve andare al primo."""
    print("\n[4] ottimizzatore dei pesi")
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500).astype(float)
    good = np.clip(y * 0.85 + 0.075 + rng.normal(0, 0.05, 500), 0.01, 0.99)
    junk = np.full(500, 0.5)
    w = optimise_weights(np.column_stack([good, junk]), y)
    _check(w[0] > 0.8, f"peso al modello informativo {w[0]:.2f} > 0.80")


def test_prediction_on_fixtures():
    """Predizione su fixture future: nessun NaN, probabilità valide."""
    print("\n[5] predizione su fixture future")
    played, future, truth = make_synthetic("testland")
    ens = HomeWinEnsemble(verbose=False).fit(played)
    ens.prepare_history(played)

    table = F.build_feature_table(played, future)
    fut = table[table["played"] == 0]
    _check(len(fut) == len(future), f"tutte le {len(future)} fixture hanno feature")

    detail = ens.predict_proba(fut, detail=True)
    p = detail["p_ensemble"].to_numpy()
    _check(np.isfinite(p).all(), "nessun NaN nelle probabilità")
    _check(((p > 0) & (p < 1)).all(), "probabilità dentro (0,1)")
    _check(p.std() > 0.02, f"il modello discrimina (std {p.std():.3f})")

    real = fut[["fixture_id"]].merge(truth, on="fixture_id")
    y = (real["home_goals"] > real["away_goals"]).to_numpy(dtype=float)
    print(f"  hit rate reale sulle fixture: {y.mean():.1%}, "
          f"probabilità media prevista {p.mean():.1%}")


def test_walk_forward():
    """Il backtest walk-forward gira e produce metriche sensate."""
    print("\n[6] backtest walk-forward")
    played, _, _ = make_synthetic("testland")
    preds = E.walk_forward(played, min_train=500, step_days=60, verbose=True)
    _check(not preds.empty, f"{len(preds)} predizioni out-of-sample")
    _check(preds["p_ensemble"].between(0, 1).all(), "probabilità valide")
    _check(preds["fixture_id"].is_unique, "nessuna partita prevista due volte")

    rep = E.report(preds)
    print(rep.round(4).to_string(index=False))
    ll_ens = E.log_loss_safe(preds["target"], preds["p_ensemble"])
    ll_prior = E.log_loss_safe(preds["target"], preds["p_prior"])
    _check(ll_ens < ll_prior,
           f"log-loss {ll_ens:.4f} < prior noto al training {ll_prior:.4f}")
    _check(E.auc_score(preds["target"], preds["p_ensemble"]) > 0.55,
           f"AUC out-of-sample {E.auc_score(preds['target'], preds['p_ensemble']):.3f}")

    summary, bets = E.value_bets(
        preds.assign(q1=1.0 / np.clip(preds["p_ensemble"] * 0.95, 0.05, 0.95)),
        min_edge=0.02)
    _check(summary["n_bets"] >= 0, f"valutazione economica eseguita: {summary}")


def test_cli():
    """train → inspect → predict → backtest dalla riga di comando."""
    print("\n[7] interfaccia a riga di comando")
    make_synthetic("testland")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "runner", _ROOT / "9_ensemble_home.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    for argv in (["train", "testland"],
                 ["inspect", "testland"],
                 ["predict", "testland", "--top=5"],
                 ["backtest", "testland", "--min-train=500", "--step-days=90",
                  "--max-steps=2"]):
        rc = runner.main(argv)
        _check(rc == 0, f"comando {' '.join(argv)} → uscita {rc}")

    _check((P.models_dir() / "testland_ensemble.pkl").exists(), "modello salvato")
    out = P.output_dir() / "ensemble_home.csv"
    _check(out.exists(), "output/ensemble_home.csv scritto")
    df = pd.read_csv(out)
    _check(len(df) == 40 and df["p_ensemble"].notna().all(),
           f"{len(df)} previsioni complete nel CSV finale")


def test_model_roundtrip():
    """Il modello salvato su disco predice come quello in memoria."""
    print("\n[8] salvataggio e ricarica del modello")
    import pickle
    played, future, _ = make_synthetic("testland")
    ens = HomeWinEnsemble(verbose=False).fit(played)
    ens.prepare_history(played)
    table = F.build_feature_table(played, future)
    fut = table[table["played"] == 0]
    p1 = ens.predict_proba(fut)

    blob = pickle.dumps(ens)
    ens2 = pickle.loads(blob)
    _check(not ens2.hmm_model.sequences, "il pickle non trascina i CSV dentro")
    ens2.prepare_history(played)
    p2 = ens2.predict_proba(fut)
    _check(np.allclose(p1, p2), "predizioni identiche dopo il round-trip")


TESTS = [
    test_features_are_causal,
    test_hmm_recovers_form,
    test_ensemble_beats_baseline,
    test_weight_optimiser,
    test_prediction_on_fixtures,
    test_walk_forward,
    test_cli,
    test_model_roundtrip,
]


def main():
    print(f"Root temporanea: {P.ROOT}")
    failed = []
    for t in TESTS:
        try:
            t()
        except AssertionError as exc:
            print(f"  FALLITO — {exc}")
            failed.append((t.__name__, str(exc)))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            failed.append((t.__name__, repr(exc)))
    print("\n" + "=" * 70)
    if failed:
        print(f"{len(failed)}/{len(TESTS)} test falliti:")
        for name, msg in failed:
            print(f"  - {name}: {msg}")
        return 1
    print(f"Tutti i {len(TESTS)} test superati.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
