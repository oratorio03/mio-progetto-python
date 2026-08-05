"""
9_ensemble_home.py — Ensemble Home Win: Logistica + HMM + Albero decisionale.

Step 9 della pipeline: gira dopo credibility.py, legge gli stessi CSV
(all_results.csv, all_fixtures.csv, odds.csv) e non fa nessuna chiamata API.

I tre modelli base guardano la stessa partita da angoli diversi:
    logistica  segnale lineare su forza, forma, riposo, scontri diretti
    HMM        stato di forma latente, con emissioni diverse in casa e fuori
    albero     interazioni a soglia fra le stesse feature

Vengono calibrati singolarmente e poi fusi con pesi stimati per minimizzare la
log-loss su una finestra di validazione temporalmente successiva al training.

Comandi:
    python 9_ensemble_home.py train    --all
    python 9_ensemble_home.py predict  italy --min-edge=0.05
    python 9_ensemble_home.py backtest --all --step-days=30
    python 9_ensemble_home.py inspect  italy

Opzioni utili:
    --cutoff=2026-03-01   ignora tutto ciò che è successo dopo (simulazioni)
    --states=3            stati latenti dell'HMM
    --min-train=600       partite minime prima di iniziare a prevedere
    --top=25              quante righe mostrare a schermo

Legge:  data/{nation}/processed/all_results.csv
        data/{nation}/processed/all_fixtures.csv
        data/{nation}/processed/odds.csv           (facoltativo)
Salva:  models/{nation}_ensemble.pkl
        data/{nation}/processed/ensemble_home.csv
        output/ensemble_home.csv                   (aggregato cross-nazione)
        output/ensemble_backtest_{nation}.csv
"""

import sys
import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from ensemble import paths as P                      # noqa: E402
from ensemble import features as F                   # noqa: E402
from ensemble import evaluate as E                   # noqa: E402
from ensemble.models import HomeWinEnsemble, BASE_NAMES   # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_results(nation, cutoff=None):
    path = P.results_path(nation)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]
    if "played" in df.columns:
        df = df[df["played"] == 1]
    df = df[df["home_goals"].notna() & df["away_goals"].notna()]
    if cutoff is not None:
        df = df[df["date"] < pd.Timestamp(cutoff)]
    return df.sort_values("date").reset_index(drop=True)


def load_fixtures(nation, cutoff=None):
    path = P.fixtures_path(nation)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]
    if "played" in df.columns:
        df = df[df["played"] != 1]
    if cutoff is not None:
        df = df[df["date"] >= pd.Timestamp(cutoff)]
    return df.sort_values("date").reset_index(drop=True)


def load_odds(nation):
    path = P.odds_path(nation)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    keep = [c for c in ("fixture_id", "q1", "qx", "q2", "imp_h", "bookmaker")
            if c in df.columns]
    return df[keep] if keep else None


def model_path(nation):
    return P.models_dir() / f"{nation}_ensemble.pkl"


def save_model(nation, ens):
    with open(model_path(nation), "wb") as f:
        pickle.dump(ens, f)


def load_model(nation):
    path = model_path(nation)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_nations(args):
    if args.all:
        return P.list_nations()
    if args.nations:
        return args.nations
    return P.list_nations()


# ── comandi ───────────────────────────────────────────────────────────────────

def cmd_train(args):
    nations = resolve_nations(args)
    if not nations:
        print("Nessuna nazione trovata (controlla config/ o data/).")
        return 1

    ok = 0
    for nation in nations:
        res = load_results(nation, args.cutoff)
        if res is None or len(res) < 200:
            n = 0 if res is None else len(res)
            print(f"[SKIP] {nation}: {n} partite disponibili (minimo 200)")
            continue
        print(f"\n[{nation}] {len(res)} partite fino a {res['date'].max().date()}")
        try:
            ens = HomeWinEnsemble(n_states=args.states).fit(res)
        except Exception as exc:
            print(f"  [ERRORE] {exc}")
            continue
        save_model(nation, ens)
        print(f"  salvato → {model_path(nation).relative_to(P.ROOT)}")
        ok += 1

    print(f"\nModelli addestrati: {ok}/{len(nations)}")
    return 0 if ok else 1


def cmd_predict(args):
    nations = resolve_nations(args)
    all_rows = []

    for nation in nations:
        res = load_results(nation, args.cutoff)
        fix = load_fixtures(nation, args.cutoff)
        if res is None or len(res) < 200:
            continue
        if fix is None or fix.empty:
            print(f"[{nation}] nessuna fixture futura")
            continue

        ens = load_model(nation)
        stale = ens is None or (args.retrain) or (
            ens.train_end is not None and
            (res["date"].max() - pd.Timestamp(ens.train_end)).days > args.max_age)
        if stale:
            print(f"[{nation}] addestramento modello…")
            try:
                ens = HomeWinEnsemble(n_states=args.states, verbose=False).fit(res)
            except Exception as exc:
                print(f"  [ERRORE] {exc}")
                continue
            save_model(nation, ens)

        table = F.build_feature_table(res, fix)
        future = table[table["played"] == 0].copy()
        if future.empty:
            continue

        ens.prepare_history(res)
        detail = ens.predict_proba(future, detail=True)
        out = future[F.META_COLS].reset_index(drop=True)
        for col in ("p_logistic", "p_hmm", "p_tree", "p_ensemble", "spread"):
            out[col] = detail[col].to_numpy()
        out["nation"] = nation

        odds = load_odds(nation)
        if odds is not None:
            out = out.merge(odds, on="fixture_id", how="left")
            if "q1" in out.columns:
                out["edge"] = out["p_ensemble"] * out["q1"] - 1.0
                out["value"] = np.where(out["edge"] >= args.min_edge, "VALUE", "")

        out = out.sort_values("p_ensemble", ascending=False)
        dest = P.data_dir(nation) / "ensemble_home.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(dest, index=False)
        all_rows.append(out)
        print(f"[{nation}] {len(out)} fixture previste → {dest.relative_to(P.ROOT)}")

    if not all_rows:
        print("Nessuna previsione generata.")
        return 1

    allp = pd.concat(all_rows, ignore_index=True).sort_values(
        "p_ensemble", ascending=False)
    dest = P.output_dir() / "ensemble_home.csv"
    allp.to_csv(dest, index=False)

    cols = ["nation", "date", "home_name", "away_name",
            "p_ensemble", "p_logistic", "p_hmm", "p_tree", "spread"]
    cols += [c for c in ("q1", "edge", "value") if c in allp.columns]
    show = allp.head(args.top).copy()
    show["date"] = pd.to_datetime(show["date"]).dt.strftime("%Y-%m-%d")
    for c in ("p_ensemble", "p_logistic", "p_hmm", "p_tree", "spread", "edge"):
        if c in show.columns:
            show[c] = show[c].round(3)
    print(f"\nTop {len(show)} Home Win — {dest.relative_to(P.ROOT)}\n")
    print(show[cols].to_string(index=False))
    return 0


def cmd_backtest(args):
    nations = resolve_nations(args)
    all_preds = []

    for nation in nations:
        res = load_results(nation, args.cutoff)
        if res is None or len(res) < args.min_train + 100:
            n = 0 if res is None else len(res)
            print(f"[SKIP] {nation}: {n} partite (servono almeno "
                  f"{args.min_train + 100})")
            continue

        print(f"\n[{nation}] walk-forward, passo {args.step_days} giorni")
        preds = E.walk_forward(res, min_train=args.min_train,
                               step_days=args.step_days, n_states=args.states,
                               start_date=args.start, max_steps=args.max_steps)
        if preds.empty:
            print("  nessun passo eseguito")
            continue

        odds = load_odds(nation)
        if odds is not None:
            preds = preds.merge(odds, on="fixture_id", how="left")

        preds["nation"] = nation
        dest = P.output_dir() / f"ensemble_backtest_{nation}.csv"
        preds.to_csv(dest, index=False)
        all_preds.append(preds)

        print(f"\n  {len(preds)} predizioni out-of-sample → "
              f"{dest.relative_to(P.ROOT)}")
        print(E.report(preds).round(4).to_string(index=False))
        print("\n  Soglie:")
        print(E.threshold_table(preds["target"], preds["p_ensemble"])
              .to_string(index=False))

    if not all_preds:
        return 1

    allp = pd.concat(all_preds, ignore_index=True)
    if len(all_preds) > 1:
        print("\n" + "=" * 78)
        print(f"AGGREGATO — {len(allp)} predizioni, {len(all_preds)} nazioni")
        print(E.report(allp).round(4).to_string(index=False))
        print("\nSoglie:")
        print(E.threshold_table(allp["target"], allp["p_ensemble"])
              .to_string(index=False))

    print("\nCalibrazione:")
    print(E.calibration_table(allp["target"], allp["p_ensemble"])
          .to_string(index=False))

    if "q1" in allp.columns and allp["q1"].notna().any():
        summary, bets = E.value_bets(allp, min_edge=args.min_edge)
        print(f"\nValue bet (edge ≥ {args.min_edge:.0%}, puntata piatta):")
        for k, v in summary.items():
            print(f"  {k:12s} {v}")

    dest = P.output_dir() / "ensemble_backtest_all.csv"
    allp.to_csv(dest, index=False)
    print(f"\nSalvato → {dest.relative_to(P.ROOT)}")
    return 0


def cmd_inspect(args):
    nations = resolve_nations(args)
    for nation in nations:
        ens = load_model(nation)
        if ens is None:
            print(f"[{nation}] nessun modello salvato — lancia prima 'train'")
            continue
        print("\n" + "=" * 78)
        print(f"[{nation}]")
        print(ens.summary())

        print("\nHMM — matrice di transizione fra stati di forma:")
        print(ens.hmm_model.transition_matrix().to_string())
        print("\nHMM — emissioni per stato e campo:")
        print(ens.hmm_model.describe_states().to_string(index=False))
        print("\nHMM — fusione delle due letture: "
              f"peso casa {ens.hmm_model.coef_[0]:.3f}, "
              f"peso ospite {ens.hmm_model.coef_[1]:.3f}, "
              f"intercetta {ens.hmm_model.intercept_:.3f}")

        print("\nLogistica — coefficienti principali:")
        print(ens.logistic.coefficients().head(12).round(3).to_string())
        print("\nAlbero — importanza feature:")
        imp = ens.tree.importances()
        print(imp[imp > 0].head(12).round(3).to_string())
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="Ensemble Home Win (logistica + HMM + albero)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("train", "predict", "backtest", "inspect"))
    p.add_argument("nations", nargs="*", help="codici nazione (default: tutte)")
    p.add_argument("--all", action="store_true", help="tutte le nazioni")
    p.add_argument("--cutoff", default=None,
                   help="ignora le partite dalla data indicata in poi")
    p.add_argument("--start", default=None,
                   help="backtest: data da cui iniziare a prevedere")
    p.add_argument("--states", type=int, default=3, help="stati latenti HMM")
    p.add_argument("--min-train", type=int, default=600,
                   help="partite minime prima di prevedere (backtest)")
    p.add_argument("--step-days", type=int, default=30,
                   help="ampiezza finestra walk-forward in giorni")
    p.add_argument("--max-steps", type=int, default=None,
                   help="limita il numero di passi del backtest")
    p.add_argument("--min-edge", type=float, default=0.05,
                   help="edge minimo per marcare una value bet")
    p.add_argument("--max-age", type=int, default=7,
                   help="giorni oltre i quali il modello salvato è riaddestrato")
    p.add_argument("--retrain", action="store_true",
                   help="forza il riaddestramento in predict")
    p.add_argument("--top", type=int, default=25, help="righe da mostrare")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    print(f"Root progetto: {P.ROOT}")
    return {
        "train":    cmd_train,
        "predict":  cmd_predict,
        "backtest": cmd_backtest,
        "inspect":  cmd_inspect,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
