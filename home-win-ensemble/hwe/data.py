"""
data.py — Caricamento e pulizia delle partite.

Un solo file CSV può contenere sia le partite giocate sia quelle da giocare:
la riga con i gol è un risultato, la riga senza gol è una partita futura.
Niente formati separati, niente cartelle obbligatorie.
"""

import numpy as np
import pandas as pd

from . import schema as S


class DataError(ValueError):
    pass


_NON_NOMI = {"", "nan", "none", "null", "na", "-", "?"}


def _clean_team(series):
    """
    Nomi normalizzati e segnaposto ridotti a valore mancante.

    Una cella vuota può arrivare come NaN o come stringa "nan" a seconda del
    dtype con cui pandas ha letto la colonna: vanno intercettate entrambe,
    altrimenti "nan" diventa una squadra a tutti gli effetti.
    """
    cleaned = (series.astype("string").str.strip()
               .str.replace(r"\s+", " ", regex=True))
    return cleaned.mask(cleaned.str.lower().isin(_NON_NOMI))


def load_matches(path, overrides=None, dayfirst=None, verbose=False):
    """
    Legge un CSV di partite e lo normalizza nello schema interno.

    Ritorna un DataFrame con: match_id, date, league, season, home, away,
    home_goals, away_goals, played, odds_home/draw/away.
    Ordinato per data, con `played` a 1 dove il risultato è noto.
    """
    raw = pd.read_csv(path)
    if raw.empty:
        raise DataError(f"{path}: file vuoto")

    mapping = S.detect(raw.columns, overrides)
    if verbose:
        print(f"Schema riconosciuto in {path}:")
        print(S.describe(mapping, list(raw.columns)))

    df = pd.DataFrame(index=raw.index)
    df["date"] = _parse_dates(raw[mapping["date"]], dayfirst)
    df["home"] = _clean_team(raw[mapping["home"]])
    df["away"] = _clean_team(raw[mapping["away"]])

    for canon in ("home_goals", "away_goals"):
        df[canon] = (pd.to_numeric(raw[mapping[canon]], errors="coerce")
                     if canon in mapping else np.nan)

    df["league"] = (raw[mapping["league"]].astype(str).str.strip()
                    if "league" in mapping else "—")
    df["season"] = (raw[mapping["season"]] if "season" in mapping
                    else _infer_season(df["date"]))

    for canon in ("odds_home", "odds_draw", "odds_away"):
        df[canon] = (pd.to_numeric(raw[mapping[canon]], errors="coerce")
                     if canon in mapping else np.nan)

    if "match_id" in mapping:
        df["match_id"] = raw[mapping["match_id"]].astype(str)
    else:
        df["match_id"] = (df["date"].dt.strftime("%Y%m%d") + "_" +
                          df["home"].str.replace(" ", "") + "_" +
                          df["away"].str.replace(" ", ""))

    return _finalise(df, path)


def _parse_dates(series, dayfirst=None):
    """
    Le date arrivano in ogni formato immaginabile. Se non è specificato,
    proviamo giorno-per-primo e mese-per-primo e teniamo quella che ne
    interpreta di più (i CSV inglesi usano dd/mm, quelli americani mm/dd).
    """
    if dayfirst is not None:
        return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)
    a = pd.to_datetime(series, errors="coerce", dayfirst=True)
    b = pd.to_datetime(series, errors="coerce", dayfirst=False)
    return a if a.notna().sum() >= b.notna().sum() else b


def _infer_season(dates):
    """Stagione europea: agosto-luglio, etichettata con l'anno d'inizio."""
    year = dates.dt.year
    return np.where(dates.dt.month >= 7, year, year - 1)


def _finalise(df, source):
    bad_date = int(df["date"].isna().sum())
    df = df[df["date"].notna()]

    bad_team = int((df["home"].isna() | df["away"].isna()).sum())
    df = df[df["home"].notna() & df["away"].notna()]
    df = df[df["home"] != df["away"]]        # una squadra non gioca contro sé stessa
    df["home"] = df["home"].astype(str)
    df["away"] = df["away"].astype(str)

    df["played"] = (df["home_goals"].notna() & df["away_goals"].notna()).astype(int)
    df = df.sort_values(["date", "played", "match_id"],
                        ascending=[True, False, True])
    df = df.drop_duplicates(subset="match_id", keep="first").reset_index(drop=True)

    if df["played"].sum() == 0:
        raise DataError(
            f"{source}: nessuna partita con risultato. Servono le colonne dei "
            f"gol (home_goals/away_goals, FTHG/FTAG, …) per addestrare.")

    if bad_date or bad_team:
        print(f"  attenzione: scartate {bad_date} righe con data illeggibile "
              f"e {bad_team} con squadra mancante")
    return df


def split_played(df):
    """(partite giocate, partite da giocare)."""
    return (df[df["played"] == 1].reset_index(drop=True),
            df[df["played"] == 0].reset_index(drop=True))


def summary(df):
    played, future = split_played(df)
    hw = (played["home_goals"] > played["away_goals"]).mean()
    teams = pd.concat([df["home"], df["away"]]).nunique()
    return {
        "partite_giocate": len(played),
        "partite_future":  len(future),
        "squadre":         int(teams),
        "campionati":      int(df["league"].nunique()),
        "dal":             played["date"].min().date().isoformat(),
        "al":              played["date"].max().date().isoformat(),
        "vittorie_casa_%": round(100 * float(hw), 1),
        "con_quote":       int(df["odds_home"].notna().sum()),
    }
