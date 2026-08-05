"""
features.py — Feature engineering strettamente causale per Home Win.

Regola non negoziabile: la riga della partita i contiene SOLO informazione
disponibile prima del fischio d'inizio di i. Tutto viene costruito in una
singola passata cronologica: prima si emettono le feature dallo stato
corrente, poi si aggiorna lo stato con il risultato.

Le fixture future entrano nella stessa passata ma non aggiornano mai lo stato,
quindi non contaminano il passato né si contaminano fra loro.
"""

from collections import defaultdict, deque

import numpy as np
import pandas as pd

# ── Parametri ─────────────────────────────────────────────────────────────────

ELO_START      = 1500.0
ELO_K          = 20.0
ELO_HOME_ADV   = 60.0     # punti Elo di vantaggio campo
ELO_SEASON_REG = 0.75     # regressione verso la media a inizio stagione
ELO_SCALE      = 400.0

WIN_HOME       = 6        # finestra rolling partite in casa
WIN_AWAY       = 6        # finestra rolling partite in trasferta
WIN_ALL        = 5        # finestra rolling forma generale
WIN_H2H        = 6        # ultimi scontri diretti considerati
MIN_HIST       = 3        # sotto questa soglia la feature rolling è NaN
LEAGUE_PRIOR_0 = 0.45     # prior home win rate prima di avere dati di lega
REST_CLIP      = (1.0, 21.0)

FEATURE_COLS = [
    "elo_diff",
    "elo_home",
    "elo_exp_home",        # attesa Elo di vittoria casalinga (con home advantage)
    "home_ppg_home",       # punti/partita in casa, ultime WIN_HOME
    "away_ppg_away",
    "home_ppg_all",
    "away_ppg_all",
    "home_gf_home",
    "home_ga_home",
    "away_gf_away",
    "away_ga_away",
    "gd_edge",             # (GF-GA in casa) - (GF-GA in trasferta)
    "home_win_rate_home",  # % vittorie in casa, ultime WIN_HOME
    "away_loss_rate_away", # % sconfitte in trasferta, ultime WIN_AWAY
    "home_streak_home",    # vittorie consecutive in casa (cap 3)
    "away_streak_away",    # sconfitte consecutive in trasferta (cap 3)
    "h2h_home_rate",
    "h2h_n",
    "rest_home",
    "rest_away",
    "rest_edge",
    "league_home_rate",
    "season_progress",
    "matches_played_home",
    "matches_played_away",
]

META_COLS = [
    "fixture_id", "league_id", "league_name", "season", "date",
    "home_id", "home_name", "away_id", "away_name", "played",
]


# ── Elo ───────────────────────────────────────────────────────────────────────

def _elo_expected(r_home, r_away):
    return 1.0 / (1.0 + 10.0 ** ((r_away - (r_home + ELO_HOME_ADV)) / ELO_SCALE))


class _EloBook:
    """Rating Elo per lega. A inizio stagione regredisce verso la media."""

    def __init__(self):
        self.rating = defaultdict(lambda: ELO_START)   # (league_id, team_id) -> float
        self.season = {}                               # league_id -> stagione corrente

    def _roll_season(self, league_id, season):
        if self.season.get(league_id) == season:
            return
        self.season[league_id] = season
        for key in [k for k in self.rating if k[0] == league_id]:
            self.rating[key] = (ELO_START +
                                ELO_SEASON_REG * (self.rating[key] - ELO_START))

    def get(self, league_id, season, team_id):
        self._roll_season(league_id, season)
        return self.rating[(league_id, team_id)]

    def update(self, league_id, home_id, away_id, outcome_home):
        """outcome_home: 1.0 vittoria casa, 0.5 pari, 0.0 sconfitta."""
        rh = self.rating[(league_id, home_id)]
        ra = self.rating[(league_id, away_id)]
        exp_h = _elo_expected(rh, ra)
        delta = ELO_K * (outcome_home - exp_h)
        self.rating[(league_id, home_id)] = rh + delta
        self.rating[(league_id, away_id)] = ra - delta


# ── Helper rolling ────────────────────────────────────────────────────────────

def _mean_or_nan(seq, min_n=MIN_HIST):
    if len(seq) < min_n:
        return np.nan
    return float(np.mean(seq))


def _streak(seq, cap=3):
    """Quante voci consecutive a 1 partendo dalla più recente (in coda)."""
    n = 0
    for v in reversed(seq):
        if v == 1:
            n += 1
            if n >= cap:
                break
        else:
            break
    return float(n)


def _points(gf, ga):
    if gf > ga:
        return 3.0
    if gf == ga:
        return 1.0
    return 0.0


# ── Costruzione tabella ───────────────────────────────────────────────────────

def _prepare(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]
    for col in ("home_goals", "away_goals"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    for col in ("home_id", "away_id"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["home_id"].notna() & df["away_id"].notna()]
    df["home_id"] = df["home_id"].astype("int64")
    df["away_id"] = df["away_id"].astype("int64")
    if "league_id" not in df.columns:
        df["league_id"] = 0
    df["league_id"] = df["league_id"].fillna(0)
    if "season" not in df.columns:
        df["season"] = 0
    if "league_name" not in df.columns:
        df["league_name"] = ""
    if "fixture_id" not in df.columns:
        df["fixture_id"] = np.arange(len(df))
    return df


def build_feature_table(results, fixtures=None):
    """
    results  : DataFrame partite giocate (all_results.csv)
    fixtures : DataFrame partite future (all_fixtures.csv), opzionale

    Ritorna un DataFrame con META_COLS + FEATURE_COLS + `target`
    (1 = vittoria casa, NaN per le fixture future), ordinato per data.
    """
    res = _prepare(results)
    res = res[(res.get("played", 1) == 1) &
              res["home_goals"].notna() & res["away_goals"].notna()].copy()
    res["played"] = 1

    frames = [res]
    if fixtures is not None and len(fixtures):
        fix = _prepare(fixtures)
        fix["played"] = 0
        fix["home_goals"] = np.nan
        fix["away_goals"] = np.nan
        # una fixture già presente fra i risultati non va duplicata
        known = set(res["fixture_id"].tolist())
        fix = fix[~fix["fixture_id"].isin(known)]
        if len(fix):
            frames.append(fix)

    allm = pd.concat(frames, ignore_index=True)
    # played=1 prima a parità di data: i risultati del giorno aggiornano lo
    # stato prima che si emettano le feature delle fixture dello stesso giorno
    allm = allm.sort_values(["date", "played", "fixture_id"],
                            ascending=[True, False, True]).reset_index(drop=True)

    elo = _EloBook()
    hist_home   = defaultdict(lambda: deque(maxlen=WIN_HOME))   # team -> dict rows
    hist_away   = defaultdict(lambda: deque(maxlen=WIN_AWAY))
    hist_all    = defaultdict(lambda: deque(maxlen=WIN_ALL))
    last_date   = {}
    played_cnt  = defaultdict(int)
    season_cnt  = defaultdict(int)          # (season, team) -> partite giocate
    h2h         = defaultdict(lambda: deque(maxlen=WIN_H2H))
    league_hw   = defaultdict(lambda: [0, 0])   # league_id -> [home wins, totali]

    rows = []
    for r in allm.itertuples(index=False):
        lid, season = r.league_id, r.season
        hid, aid    = r.home_id, r.away_id

        rh = elo.get(lid, season, hid)
        ra = elo.get(lid, season, aid)

        hh = hist_home[hid]
        aa = hist_away[aid]
        ha = hist_all[hid]
        av = hist_all[aid]

        hw, tot = league_hw[lid]
        league_rate = (hw / tot) if tot >= 50 else LEAGUE_PRIOR_0

        h2h_key = (min(hid, aid), max(hid, aid))
        h2h_seq = h2h[h2h_key]
        # normalizza il risultato dal punto di vista della squadra di casa attuale
        h2h_vals = [(v if home_side == hid else 1.0 - v) for home_side, v in h2h_seq]
        h2h_rate = float(np.mean(h2h_vals)) if h2h_vals else np.nan

        rest_h = np.nan
        rest_a = np.nan
        if hid in last_date:
            rest_h = float(np.clip((r.date - last_date[hid]).days, *REST_CLIP))
        if aid in last_date:
            rest_a = float(np.clip((r.date - last_date[aid]).days, *REST_CLIP))

        home_gf = _mean_or_nan([x["gf"] for x in hh])
        home_ga = _mean_or_nan([x["ga"] for x in hh])
        away_gf = _mean_or_nan([x["gf"] for x in aa])
        away_ga = _mean_or_nan([x["ga"] for x in aa])
        gd_edge = np.nan
        if not any(np.isnan(v) for v in (home_gf, home_ga, away_gf, away_ga)):
            gd_edge = (home_gf - home_ga) - (away_gf - away_ga)

        feat = {
            "elo_diff":            rh - ra,
            "elo_home":            rh,
            "elo_exp_home":        _elo_expected(rh, ra),
            "home_ppg_home":       _mean_or_nan([x["pts"] for x in hh]),
            "away_ppg_away":       _mean_or_nan([x["pts"] for x in aa]),
            "home_ppg_all":        _mean_or_nan([x["pts"] for x in ha]),
            "away_ppg_all":        _mean_or_nan([x["pts"] for x in av]),
            "home_gf_home":        home_gf,
            "home_ga_home":        home_ga,
            "away_gf_away":        away_gf,
            "away_ga_away":        away_ga,
            "gd_edge":             gd_edge,
            "home_win_rate_home":  _mean_or_nan([x["win"] for x in hh]),
            "away_loss_rate_away": _mean_or_nan([x["loss"] for x in aa]),
            "home_streak_home":    _streak([x["win"] for x in hh]),
            "away_streak_away":    _streak([x["loss"] for x in aa]),
            "h2h_home_rate":       h2h_rate,
            "h2h_n":               float(len(h2h_vals)),
            "rest_home":           rest_h,
            "rest_away":           rest_a,
            "rest_edge":           (rest_h - rest_a)
                                   if not (np.isnan(rest_h) or np.isnan(rest_a))
                                   else np.nan,
            "league_home_rate":    league_rate,
            "season_progress":     min(season_cnt[(season, hid)], 38) / 38.0,
            "matches_played_home": float(played_cnt[hid]),
            "matches_played_away": float(played_cnt[aid]),
        }

        target = np.nan
        if r.played == 1:
            target = 1.0 if r.home_goals > r.away_goals else 0.0

        rows.append({
            "fixture_id":  r.fixture_id,
            "league_id":   lid,
            "league_name": getattr(r, "league_name", ""),
            "season":      season,
            "date":        r.date,
            "home_id":     hid,
            "home_name":   getattr(r, "home_name", ""),
            "away_id":     aid,
            "away_name":   getattr(r, "away_name", ""),
            "played":      r.played,
            "target":      target,
            **feat,
        })

        # ── aggiornamento stato: solo partite realmente giocate ───────────────
        if r.played != 1:
            continue

        hg, ag = float(r.home_goals), float(r.away_goals)
        outcome_home = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        elo.update(lid, hid, aid, outcome_home)

        hh.append({"gf": hg, "ga": ag, "pts": _points(hg, ag),
                   "win": 1 if hg > ag else 0, "loss": 1 if hg < ag else 0})
        aa.append({"gf": ag, "ga": hg, "pts": _points(ag, hg),
                   "win": 1 if ag > hg else 0, "loss": 1 if ag < hg else 0})
        ha.append({"pts": _points(hg, ag)})
        av.append({"pts": _points(ag, hg)})

        h2h_seq.append((hid, 1.0 if hg > ag else (0.5 if hg == ag else 0.0)))
        last_date[hid] = r.date
        last_date[aid] = r.date
        played_cnt[hid] += 1
        played_cnt[aid] += 1
        season_cnt[(season, hid)] += 1
        season_cnt[(season, aid)] += 1
        league_hw[lid][0] += 1 if hg > ag else 0
        league_hw[lid][1] += 1

    out = pd.DataFrame(rows, columns=META_COLS + ["target"] + FEATURE_COLS)
    # mergesort = stabile: l'ordine dentro la stessa data resta quello della
    # passata, quindi la riga i è identica che il file finisca a i o molto dopo
    return out.sort_values("date", kind="mergesort").reset_index(drop=True)


class Imputer:
    """Mediane calcolate sul solo training, riusate identiche in predizione."""

    def __init__(self):
        self.medians = {}

    def fit(self, X):
        for col in FEATURE_COLS:
            vals = pd.to_numeric(X[col], errors="coerce")
            med = vals.median()
            self.medians[col] = 0.0 if pd.isna(med) else float(med)
        return self

    def transform(self, X):
        out = X[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").copy()
        for col in FEATURE_COLS:
            out[col] = out[col].fillna(self.medians.get(col, 0.0))
        return out.to_numpy(dtype=float)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
