"""
data.py — caricamento e normalizzazione dei dati per l'ensemble.

Legge i CSV già prodotti dalla pipeline esistente:
    data/{nazione}/processed/all_results.csv    partite giocate
    data/{nazione}/processed/all_fixtures.csv   partite future
    data/{nazione}/processed/odds.csv           quote (snapshot di raccolta)
    data/{nazione}/processed/odds_closing.csv   quote di chiusura (opzionale, per il CLV)

Produce un unico DataFrame ordinato cronologicamente con schema stabile,
usato da ratings.py, features.py e dai modelli.

Colonne xG (home_xg / away_xg) sono opzionali: se presenti nei CSV vengono
usate dai rating GAP, altrimenti si ricade sui gol effettivi.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG_DIR, DATA_DIR

warnings.filterwarnings("ignore")

# Schema minimo richiesto in all_results.csv / all_fixtures.csv
BASE_COLS = [
    "fixture_id", "league_id", "league_name", "tier", "season", "round",
    "date", "time", "played",
    "home_id", "home_name", "away_id", "away_name",
    "home_goals", "away_goals",
]

ODDS_COLS = ["q1", "qx", "q2", "odd_o25", "odd_o15", "odd_btts", "odd_1x", "overround"]
XG_COLS   = ["home_xg", "away_xg"]


# ── Scoperta nazioni ──────────────────────────────────────────────────────────
def discover_nations(root=None):
    """Elenca i codici nazione disponibili (da config/*.json, altrimenti da data/)."""
    cfg_dir  = Path(root) / "config" if root else CONFIG_DIR
    data_dir = Path(root) / "data" if root else DATA_DIR

    if cfg_dir.exists():
        nations = sorted(p.stem for p in cfg_dir.glob("*.json"))
        if nations:
            return nations
    if data_dir.exists():
        return sorted(p.name for p in data_dir.iterdir()
                      if p.is_dir() and (p / "processed").exists())
    return []


def nation_label(nation_code, root=None):
    """Nome leggibile della nazione (dal config, altrimenti il codice capitalizzato)."""
    cfg_dir = Path(root) / "config" if root else CONFIG_DIR
    path    = cfg_dir / f"{nation_code}.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("nation", nation_code.capitalize())
        except Exception:
            pass
    return nation_code.capitalize()


def processed_dir(nation_code, root=None):
    base = Path(root) / "data" if root else DATA_DIR
    return base / nation_code / "processed"


# ── Caricamento ───────────────────────────────────────────────────────────────
def _read_csv(path):
    if not Path(path).exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"  [WARN] impossibile leggere {path}: {e}")
        return pd.DataFrame()


def _normalize(df, nation_code, nation_name):
    """Porta un CSV grezzo allo schema comune."""
    if df.empty:
        return df

    for col in BASE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df = df.copy()
    df["nation"]      = nation_name
    df["nation_code"] = nation_code
    df["date"]        = pd.to_datetime(df["date"], errors="coerce")
    df["time"]        = df["time"].fillna("00:00").astype(str)

    for col in ["home_goals", "away_goals", "home_id", "away_id",
                "league_id", "tier", "season", "played"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in XG_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan

    df = df[df["date"].notna()]
    df = df[df["home_id"].notna() & df["away_id"].notna()]

    # Chiavi intere e stabili: i team_id sono globali (API-Football)
    df["home_id"] = df["home_id"].astype("int64")
    df["away_id"] = df["away_id"].astype("int64")
    df["league_key"] = (df["nation_code"].astype(str) + "|" +
                        df["league_id"].fillna(-1).astype("int64").astype(str))
    return df


def load_odds(nation_code, root=None):
    """Quote di raccolta + eventuali quote di chiusura (per il CLV)."""
    proc = processed_dir(nation_code, root)
    odds = _read_csv(proc / "odds.csv")
    if not odds.empty:
        keep = ["fixture_id"] + [c for c in ODDS_COLS if c in odds.columns]
        odds = odds[keep].copy()
        odds["fixture_id"] = pd.to_numeric(odds["fixture_id"], errors="coerce")
        odds = odds.dropna(subset=["fixture_id"]).drop_duplicates("fixture_id", keep="last")

    closing = _read_csv(proc / "odds_closing.csv")
    if not closing.empty:
        keep = ["fixture_id"] + [c for c in ODDS_COLS if c in closing.columns]
        closing = closing[keep].copy()
        closing["fixture_id"] = pd.to_numeric(closing["fixture_id"], errors="coerce")
        closing = (closing.dropna(subset=["fixture_id"])
                          .drop_duplicates("fixture_id", keep="last"))
        closing = closing.rename(columns={c: f"close_{c}" for c in ODDS_COLS
                                          if c in closing.columns})
        odds = (closing if odds.empty
                else odds.merge(closing, on="fixture_id", how="outer"))
    return odds


def load_nation(nation_code, root=None, include_fixtures=True):
    """Carica risultati (+ fixture future) di una nazione, con quote agganciate."""
    proc   = processed_dir(nation_code, root)
    name   = nation_label(nation_code, root)

    results = _normalize(_read_csv(proc / "all_results.csv"), nation_code, name)
    if not results.empty:
        results = results[results["home_goals"].notna() & results["away_goals"].notna()]
        results["played"] = 1

    frames = [results]
    if include_fixtures:
        fixtures = _normalize(_read_csv(proc / "all_fixtures.csv"), nation_code, name)
        if not fixtures.empty:
            fixtures["played"] = 0
            fixtures["home_goals"] = np.nan
            fixtures["away_goals"] = np.nan
            # una fixture già presente fra i risultati non va duplicata
            if not results.empty:
                fixtures = fixtures[~fixtures["fixture_id"].isin(results["fixture_id"])]
            frames.append(fixtures)

    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if df.empty:
        return df

    odds = load_odds(nation_code, root)
    if not odds.empty:
        df["fixture_id"] = pd.to_numeric(df["fixture_id"], errors="coerce")
        df = df.merge(odds, on="fixture_id", how="left")

    for col in ODDS_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def load_dataset(nations=None, root=None, include_fixtures=True, verbose=True):
    """
    Carica e concatena più nazioni in un unico DataFrame ordinato per data.

    L'ordinamento cronologico è la precondizione di tutto il resto: i rating
    e le predizioni vengono generati in un singolo passaggio in avanti nel tempo.
    """
    nations = nations or discover_nations(root)
    frames  = []
    for code in nations:
        df = load_nation(code, root, include_fixtures=include_fixtures)
        if df.empty:
            if verbose:
                print(f"  [SKIP] {code}: nessun dato")
            continue
        if verbose:
            n_played = int((df["played"] == 1).sum())
            print(f"  {code:<12} {n_played:>6} giocate  {len(df) - n_played:>4} future")
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = add_targets(df)
    df = sort_chronological(df)
    return df


def sort_chronological(df):
    """Ordina per data/ora/fixture_id — ordine deterministico e senza look-ahead."""
    df = df.copy()
    df["kickoff"] = pd.to_datetime(
        df["date"].dt.strftime("%Y-%m-%d") + " " + df["time"].str.slice(0, 5),
        errors="coerce",
    )
    df["kickoff"] = df["kickoff"].fillna(df["date"])
    df = (df.sort_values(["kickoff", "fixture_id"], kind="mergesort")
            .reset_index(drop=True))
    return df


def add_targets(df):
    """Aggiunge le colonne obiettivo dei tre mercati (NaN per le partite future)."""
    df = df.copy()
    hg, ag = df["home_goals"], df["away_goals"]
    played = hg.notna() & ag.notna()

    total = hg + ag
    # 1X2 codificato 0=H, 1=D, 2=A (coerente con OUTCOME_1X2)
    y_1x2 = np.where(hg > ag, 0, np.where(hg == ag, 1, 2)).astype(float)
    df["y_1x2"]    = np.where(played, y_1x2, np.nan)
    df["y_over25"] = np.where(played, (total > 2.5).astype(float), np.nan)
    df["y_btts"]   = np.where(played, ((hg > 0) & (ag > 0)).astype(float), np.nan)
    df["total_goals"] = np.where(played, total, np.nan)
    return df


def train_test_split_by_date(df, cutoff):
    """Split temporale puro: train < cutoff <= test. Nessuna partita a cavallo."""
    cutoff = pd.to_datetime(cutoff)
    return df[df["kickoff"] < cutoff].copy(), df[df["kickoff"] >= cutoff].copy()
