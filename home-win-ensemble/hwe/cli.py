"""
cli.py — Interfaccia a riga di comando.

    python -m hwe demo                          prova tutto senza avere dati
    python -m hwe schema    --data partite.csv  come viene letto il file
    python -m hwe train     --data partite.csv
    python -m hwe predict   --data partite.csv
    python -m hwe backtest  --data partite.csv
    python -m hwe inspect
"""

import argparse
import pickle
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import evaluation as E
from . import features as F
from . import schema as S
from . import synthetic
from .ensemble import HomeWinEnsemble

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

DEFAULT_MODEL = "model.pkl"


# ── utilità ───────────────────────────────────────────────────────────────────

def _load(args, verbose=None):
    overrides = S.parse_overrides(args.map)
    matches = D.load_matches(
        args.data, overrides=overrides, dayfirst=args.dayfirst,
        verbose=args.verbose if verbose is None else verbose)
    played, future = D.split_played(matches)
    return matches, played, future


def _save_model(ens, path):
    with open(path, "wb") as handle:
        pickle.dump(ens, handle)


def _load_model(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _percent(frame, columns):
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].astype(float).round(3)
    return out


# ── comandi ───────────────────────────────────────────────────────────────────

def cmd_schema(args):
    raw = pd.read_csv(args.data, nrows=200)
    mapping = S.detect(raw.columns, S.parse_overrides(args.map))
    print(f"File: {args.data}")
    print(S.describe(mapping, list(raw.columns)))
    matches, played, future = _load(args, verbose=False)   # già stampato sopra
    print("\nDati letti:")
    for key, value in D.summary(matches).items():
        print(f"  {key:16s} {value}")
    return 0


def cmd_train(args):
    matches, played, _ = _load(args)
    print(f"{len(played)} partite con risultato, "
          f"dal {played['date'].min().date()} al {played['date'].max().date()}")
    ens = HomeWinEnsemble(n_states=args.states, n_folds=args.folds).fit(played)
    _save_model(ens, args.model)
    print(f"\n{ens.summary()}")
    print(f"\nModello salvato in {args.model}")
    return 0


def cmd_predict(args):
    matches, played, future = _load(args)
    if future.empty:
        print("Nessuna partita da prevedere: nel file tutte le righe hanno "
              "già il risultato.\nAggiungi le partite future con le colonne "
              "dei gol vuote.")
        return 1

    path = Path(args.model)
    if path.exists() and not args.retrain:
        ens = _load_model(path)
        print(f"Modello caricato da {path} ({ens.n_train} partite di training)")
    else:
        print("Addestramento…")
        ens = HomeWinEnsemble(n_states=args.states, n_folds=args.folds,
                              verbose=args.verbose).fit(played)
        _save_model(ens, path)

    table = F.build(matches)
    rows = table[table["played"] == 0].reset_index(drop=True)
    ens.set_history(played)
    detail = ens.predict_proba(rows, detail=True)

    out = rows[["date", "league", "home", "away", "odds_home"]].copy()
    for column in ("p_ensemble", "p_logistica", "p_hmm", "p_albero", "disaccordo"):
        out[column] = detail[column].to_numpy()
    if out["odds_home"].notna().any():
        out["edge"] = out["p_ensemble"] * out["odds_home"] - 1.0
        out["value"] = np.where(out["edge"] >= args.min_edge, "VALUE", "")

    out = out.sort_values("p_ensemble", ascending=False).reset_index(drop=True)
    out.to_csv(args.out, index=False)

    show = _percent(out.head(args.top), ["p_ensemble", "p_logistica", "p_hmm",
                                         "p_albero", "disaccordo", "edge"])
    show["date"] = pd.to_datetime(show["date"]).dt.strftime("%Y-%m-%d")
    print(f"\nProbabilità di vittoria in casa — prime {len(show)} di {len(out)}\n")
    print(show.to_string(index=False))
    print(f"\nTutte le previsioni in {args.out}")
    return 0


def cmd_backtest(args):
    matches, played, _ = _load(args)
    preds = E.walk_forward(played, min_train=args.min_train,
                           step_days=args.step_days, n_states=args.states,
                           start=args.start, max_steps=args.max_steps)
    if preds.empty:
        print("Nessun passo eseguito: servono più partite o un --min-train "
              "più basso.")
        return 1

    preds.to_csv(args.out, index=False)
    print(f"\n{len(preds)} predizioni out-of-sample, "
          f"dal {preds['date'].min().date()} al {preds['date'].max().date()}")
    print("\nMetriche per modello:")
    print(E.report(preds).to_string(index=False))
    print("\nSoglie di segnalazione:")
    print(E.threshold_table(preds["target"], preds["p_ensemble"]).to_string(index=False))
    print("\nCalibrazione:")
    print(E.calibration_table(preds["target"], preds["p_ensemble"]).to_string(index=False))

    if preds["odds_home"].notna().any():
        summary, _ = E.value_bets(preds, min_edge=args.min_edge)
        print(f"\nValue bet (edge ≥ {args.min_edge:.0%}, puntata piatta):")
        for key, value in summary.items():
            print(f"  {key:13s} {value}")

    print(f"\nPredizioni salvate in {args.out}")
    return 0


def cmd_inspect(args):
    path = Path(args.model)
    if not path.exists():
        print(f"Nessun modello in {path}: lancia prima 'train'.")
        return 1
    ens = _load_model(path)
    print(ens.summary())
    print("\nHMM — transizioni fra stati di forma "
          "(s0 = in crisi, s2 = in fiducia):")
    print(ens.hmm.transitions_table().to_string())
    print("\nHMM — cosa emette ogni stato, per campo:")
    print(ens.hmm.emissions_table().to_string(index=False))
    print(f"\nHMM — fusione delle due letture: peso casa {ens.hmm.coef_[0]:.3f}, "
          f"peso ospite {ens.hmm.coef_[1]:.3f}, "
          f"intercetta {ens.hmm.intercept_:.3f}")
    print("\nLogistica — coefficienti più pesanti:")
    print(ens.logistica.coefficients().head(12).round(3).to_string())
    print("\nAlbero — feature usate:")
    importances = ens.albero.importances()
    print(importances[importances > 0].head(12).round(3).to_string())
    return 0


def cmd_demo(args):
    """Genera un campionato finto e ci fa girare sopra tutta la pipeline."""
    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(
        prefix="hwe_demo_"))
    workdir.mkdir(parents=True, exist_ok=True)
    csv = workdir / "partite_demo.csv"
    synthetic.write_csv(csv, seasons=args.seasons)
    print(f"Campionato sintetico generato in {csv}")
    print("(colonne in stile football-data.co.uk: Div, Date, HomeTeam, "
          "AwayTeam, FTHG, FTAG, B365H…)\n")

    # verbose spento: lo schema lo mostra il passo 1, ripeterlo a ogni passo
    # sporcherebbe l'output della demo
    shared = dict(data=str(csv), map=[], dayfirst=None, verbose=False,
                  states=args.states, folds=3, min_edge=0.05,
                  model=str(workdir / DEFAULT_MODEL))

    print("=" * 76 + "\n1. LETTURA DEL FILE\n" + "=" * 76)
    cmd_schema(argparse.Namespace(**shared))

    print("\n" + "=" * 76 + "\n2. ADDESTRAMENTO\n" + "=" * 76)
    cmd_train(argparse.Namespace(**shared))

    print("\n" + "=" * 76 + "\n3. COSA HA IMPARATO\n" + "=" * 76)
    cmd_inspect(argparse.Namespace(**shared))

    print("\n" + "=" * 76 + "\n4. PREVISIONI SULLE PARTITE DA GIOCARE\n" + "=" * 76)
    cmd_predict(argparse.Namespace(retrain=False, top=15,
                                   out=str(workdir / "previsioni.csv"), **shared))

    print("\n" + "=" * 76 + "\n5. BACKTEST WALK-FORWARD\n" + "=" * 76)
    cmd_backtest(argparse.Namespace(
        min_train=args.min_train, step_days=60, start=None, max_steps=args.max_steps,
        out=str(workdir / "backtest.csv"), **shared))

    print(f"\nTutti i file della demo sono in {workdir}")
    return 0


# ── parser ────────────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        prog="hwe",
        description="Ensemble per la probabilità di vittoria in casa nel calcio "
                    "(regressione logistica + HMM + albero decisionale)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub, needs_data=True):
        if needs_data:
            sub.add_argument("--data", required=True,
                             help="CSV delle partite (giocate e/o da giocare)")
            sub.add_argument("--map", action="append", default=[],
                             metavar="canonica=colonna",
                             help="forza una colonna, es. --map home=Squadra1")
            sub.add_argument("--dayfirst", type=lambda v: v.lower() == "true",
                             default=None,
                             help="true/false per il formato delle date")
        sub.add_argument("--states", type=int, default=3,
                         help="stati latenti dell'HMM (default 3)")
        sub.add_argument("--folds", type=int, default=3,
                         help="fold out-of-fold per calibrazione e pesi")
        sub.add_argument("--verbose", action="store_true")

    p = subparsers.add_parser("schema", help="mostra come viene letto il CSV")
    add_common(p)

    p = subparsers.add_parser("train", help="addestra e salva il modello")
    add_common(p)
    p.add_argument("--model", default=DEFAULT_MODEL)

    p = subparsers.add_parser("predict", help="prevede le partite da giocare")
    add_common(p)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default="previsioni.csv")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--min-edge", type=float, default=0.05,
                   help="edge minimo per marcare VALUE (serve la quota)")
    p.add_argument("--retrain", action="store_true",
                   help="riaddestra anche se esiste un modello salvato")

    p = subparsers.add_parser("backtest", help="walk-forward su tutto lo storico")
    add_common(p)
    p.add_argument("--out", default="backtest.csv")
    p.add_argument("--min-train", type=int, default=600,
                   help="partite minime prima di iniziare a prevedere")
    p.add_argument("--step-days", type=int, default=30,
                   help="giorni previsti a ogni passo")
    p.add_argument("--start", default=None, help="data da cui iniziare")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--min-edge", type=float, default=0.05)

    p = subparsers.add_parser("inspect", help="cosa ha imparato il modello")
    add_common(p, needs_data=False)
    p.add_argument("--model", default=DEFAULT_MODEL)

    p = subparsers.add_parser(
        "demo", help="genera dati finti e mostra l'intera pipeline")
    add_common(p, needs_data=False)
    p.add_argument("--workdir", default=None)
    p.add_argument("--seasons", type=int, default=3)
    p.add_argument("--min-train", type=int, default=600)
    p.add_argument("--max-steps", type=int, default=None)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    handlers = {
        "schema": cmd_schema, "train": cmd_train, "predict": cmd_predict,
        "backtest": cmd_backtest, "inspect": cmd_inspect, "demo": cmd_demo,
    }
    try:
        return handlers[args.command](args)
    except (S.SchemaError, D.DataError) as error:
        print(f"\nErrore nei dati: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
