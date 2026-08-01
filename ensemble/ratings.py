"""
ratings.py — rating dinamici calcolati in un solo passaggio cronologico.

Contiene tre sistemi di rating usati come feature dai modelli:

  Elo          forza complessiva, con vantaggio casa per lega stimato online
  pi-ratings   Constantinou & Fenton (2013): rating separati casa/trasferta,
               aggiornati sull'errore di differenza reti
  GAP ratings  attacco/difesa per venue sui gol (o sull'xG quando disponibile),
               base del modello M3

Regola non negoziabile: per ogni partita le feature sono lo stato dei rating
PRIMA che la partita venga giocata. L'aggiornamento avviene subito dopo aver
emesso la riga di feature, e solo se la partita è stata effettivamente giocata.
Le partite future non aggiornano nulla, quindi lo stesso codice serve backtest
walk-forward e produzione.
"""

import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .config import RatingConfig

# Nomi delle feature prodotte, in ordine stabile
RATING_FEATURES = [
    "elo_home", "elo_away", "elo_diff", "elo_exp_home", "elo_hfa",
    "pi_home_h", "pi_home_a", "pi_away_h", "pi_away_a", "pi_exp_gd",
    "gap_att_home", "gap_def_home", "gap_att_away", "gap_def_away",
    "gap_exp_home_goals", "gap_exp_away_goals", "gap_exp_total", "gap_exp_supr",
    "form_ppg_home", "form_ppg_away", "form_ppg_diff",
    "form_gf_home", "form_ga_home", "form_gf_away", "form_ga_away",
    "form_venue_ppg_home", "form_venue_ppg_away",
    "form_o25_home", "form_o25_away", "form_btts_home", "form_btts_away",
    "rest_days_home", "rest_days_away", "rest_diff",
    "matches_recent_home", "matches_recent_away",
    "n_matches_home", "n_matches_away", "is_new_home", "is_new_away",
    "league_avg_goals", "league_home_rate", "league_elo_mean", "tier_num",
    "season_match_home", "season_match_away",
]


def _psi(r, c):
    """Funzione psi dei pi-ratings: converte un rating in differenza reti attesa."""
    return math.copysign(10.0 ** (abs(r) / c) - 1.0, r)


def _elo_goal_multiplier(gd):
    """Moltiplicatore K in funzione della differenza reti (scala Elo per il calcio)."""
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


class _TeamState:
    __slots__ = ("elo", "pi_h", "pi_a",
                 "gap_att_h", "gap_def_h", "gap_att_a", "gap_def_a",
                 "n_matches", "last_date", "recent_dates", "season", "season_matches",
                 "form", "form_home", "form_away", "league_key")

    def __init__(self, elo, gap_home_goals, gap_away_goals, league_key, window):
        self.elo   = elo
        self.pi_h  = 0.0
        self.pi_a  = 0.0
        # attacco = gol attesi segnati nel venue, difesa = gol attesi subiti nel venue
        self.gap_att_h = gap_home_goals
        self.gap_def_h = gap_away_goals
        self.gap_att_a = gap_away_goals
        self.gap_def_a = gap_home_goals
        self.n_matches = 0
        self.last_date = None
        self.recent_dates = deque(maxlen=20)
        self.season = None
        self.season_matches = 0
        self.form      = deque(maxlen=window)
        self.form_home = deque(maxlen=window)
        self.form_away = deque(maxlen=window)
        self.league_key = league_key


class _LeagueState:
    __slots__ = ("n", "sum_goals", "home_pts", "hfa", "elo_mean", "avg_home_goals",
                 "avg_away_goals")

    def __init__(self, hfa, avg_goals, elo_mean):
        self.n = 0
        self.sum_goals = 0.0
        self.home_pts = 0.0
        self.hfa = hfa
        self.elo_mean = elo_mean
        self.avg_home_goals = avg_goals * 0.55
        self.avg_away_goals = avg_goals * 0.45


class RatingEngine:
    """
    Motore di rating online.

    Uso tipico:
        eng   = RatingEngine(cfg.ratings)
        feats = eng.transform(df)          # df ordinato cronologicamente

    `transform` restituisce un DataFrame con lo stesso indice di `df`.
    Dopo `transform` lo stato interno riflette tutte le partite giocate del df,
    quindi il motore può essere riusato per predire fixture successive.
    """

    def __init__(self, cfg: RatingConfig = None):
        self.cfg = cfg or RatingConfig()
        self.teams   = {}
        self.leagues = {}
        # medie globali con shrinkage: partono da valori sensati per il calcio
        self.global_goals = 2.60
        self.global_n     = 0
        self.global_home_pts = 0.5

    # ── accesso allo stato ────────────────────────────────────────────────────
    def _league(self, league_key, tier):
        st = self.leagues.get(league_key)
        if st is None:
            tier = 1.0 if tier is None or (isinstance(tier, float) and np.isnan(tier)) else float(tier)
            elo0 = self.cfg.elo_start - (tier - 1.0) * self.cfg.elo_tier_step
            st = _LeagueState(self.cfg.elo_hfa, self.global_goals, elo0)
            self.leagues[league_key] = st
        return st

    def _team(self, team_id, league_key, tier):
        st = self.teams.get(team_id)
        if st is None:
            lg = self._league(league_key, tier)
            st = _TeamState(lg.elo_mean, lg.avg_home_goals, lg.avg_away_goals,
                            league_key, self.cfg.form_window)
            self.teams[team_id] = st
        return st

    def _league_avg_goals(self, lg):
        """Media gol della lega con shrinkage verso la media globale."""
        prior = self.cfg.league_prior_matches
        return (lg.sum_goals + prior * self.global_goals) / (lg.n + prior)

    def _league_home_rate(self, lg):
        prior = self.cfg.league_prior_matches
        return (lg.home_pts + prior * self.global_home_pts) / (lg.n + prior)

    # ── gestione stagione ─────────────────────────────────────────────────────
    def _season_rollover(self, ts, season, lg):
        """A inizio stagione i rating tornano parzialmente verso la media lega."""
        if season is None or (isinstance(season, float) and np.isnan(season)):
            return
        if ts.season is None:
            ts.season = season
            return
        if season == ts.season:
            return

        r = self.cfg.elo_season_regress
        ts.elo = ts.elo + (lg.elo_mean - ts.elo) * r
        ts.pi_h *= (1.0 - r)
        ts.pi_a *= (1.0 - r)

        g = self.cfg.gap_season_regress
        ts.gap_att_h += (lg.avg_home_goals - ts.gap_att_h) * g
        ts.gap_def_h += (lg.avg_away_goals - ts.gap_def_h) * g
        ts.gap_att_a += (lg.avg_away_goals - ts.gap_att_a) * g
        ts.gap_def_a += (lg.avg_home_goals - ts.gap_def_a) * g

        ts.season = season
        ts.season_matches = 0

    # ── feature ───────────────────────────────────────────────────────────────
    @staticmethod
    def _form_stats(form):
        if not form:
            return dict(ppg=np.nan, gf=np.nan, ga=np.nan, o25=np.nan, btts=np.nan)
        n = len(form)
        return dict(
            ppg=sum(f["pts"] for f in form) / n,
            gf=sum(f["gf"] for f in form) / n,
            ga=sum(f["ga"] for f in form) / n,
            o25=sum(f["o25"] for f in form) / n,
            btts=sum(f["btts"] for f in form) / n,
        )

    def _row_features(self, hs, as_, lg, kickoff):
        cfg = self.cfg

        elo_diff = (hs.elo + lg.hfa) - as_.elo
        elo_exp  = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

        pi_exp_gd = _psi(hs.pi_h, cfg.pi_c) - _psi(as_.pi_a, cfg.pi_c)

        exp_home = max(0.05, (hs.gap_att_h + as_.gap_def_a) / 2.0)
        exp_away = max(0.05, (as_.gap_att_a + hs.gap_def_h) / 2.0)

        fh, fa = self._form_stats(hs.form), self._form_stats(as_.form)
        fhh, faa = self._form_stats(hs.form_home), self._form_stats(as_.form_away)

        def rest(ts):
            if ts.last_date is None:
                return np.nan
            return (kickoff - ts.last_date).total_seconds() / 86400.0

        def recent(ts):
            if not ts.recent_dates:
                return 0
            lim = kickoff - pd.Timedelta(days=cfg.congestion_days)
            return sum(1 for d in ts.recent_dates if d >= lim)

        rest_h, rest_a = rest(hs), rest(as_)

        return {
            "elo_home": hs.elo, "elo_away": as_.elo,
            "elo_diff": elo_diff, "elo_exp_home": elo_exp, "elo_hfa": lg.hfa,
            "pi_home_h": hs.pi_h, "pi_home_a": hs.pi_a,
            "pi_away_h": as_.pi_h, "pi_away_a": as_.pi_a,
            "pi_exp_gd": pi_exp_gd,
            "gap_att_home": hs.gap_att_h, "gap_def_home": hs.gap_def_h,
            "gap_att_away": as_.gap_att_a, "gap_def_away": as_.gap_def_a,
            "gap_exp_home_goals": exp_home, "gap_exp_away_goals": exp_away,
            "gap_exp_total": exp_home + exp_away, "gap_exp_supr": exp_home - exp_away,
            "form_ppg_home": fh["ppg"], "form_ppg_away": fa["ppg"],
            "form_ppg_diff": (fh["ppg"] - fa["ppg"]
                              if not (np.isnan(fh["ppg"]) or np.isnan(fa["ppg"])) else np.nan),
            "form_gf_home": fh["gf"], "form_ga_home": fh["ga"],
            "form_gf_away": fa["gf"], "form_ga_away": fa["ga"],
            "form_venue_ppg_home": fhh["ppg"], "form_venue_ppg_away": faa["ppg"],
            "form_o25_home": fh["o25"], "form_o25_away": fa["o25"],
            "form_btts_home": fh["btts"], "form_btts_away": fa["btts"],
            "rest_days_home": rest_h, "rest_days_away": rest_a,
            "rest_diff": (rest_h - rest_a) if not (np.isnan(rest_h) or np.isnan(rest_a)) else np.nan,
            "matches_recent_home": recent(hs), "matches_recent_away": recent(as_),
            "n_matches_home": hs.n_matches, "n_matches_away": as_.n_matches,
            "is_new_home": 1.0 if hs.n_matches < 5 else 0.0,
            "is_new_away": 1.0 if as_.n_matches < 5 else 0.0,
            "league_avg_goals": self._league_avg_goals(lg),
            "league_home_rate": self._league_home_rate(lg),
            "league_elo_mean": lg.elo_mean,
            "season_match_home": hs.season_matches, "season_match_away": as_.season_matches,
        }

    # ── aggiornamento ─────────────────────────────────────────────────────────
    def _update(self, hs, as_, lg, hg, ag, hxg, axg, kickoff, feats):
        cfg = self.cfg

        # Elo
        gd = hg - ag
        s  = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        k  = cfg.elo_k * _elo_goal_multiplier(gd)
        err = s - feats["elo_exp_home"]
        hs.elo += k * err
        as_.elo -= k * err
        lg.hfa = float(np.clip(lg.hfa + cfg.elo_hfa_lr * k * err, 0.0, 200.0))
        lg.elo_mean += 0.02 * ((hs.elo + as_.elo) / 2.0 - lg.elo_mean)

        # pi-ratings
        e = abs(gd - feats["pi_exp_gd"])
        delta = cfg.pi_lambda * cfg.pi_c * math.log10(1.0 + e)
        sign  = 1.0 if gd > feats["pi_exp_gd"] else -1.0
        d_home = sign * delta
        hs.pi_h += d_home
        hs.pi_a += d_home * cfg.pi_gamma
        as_.pi_a -= d_home
        as_.pi_h -= d_home * cfg.pi_gamma

        # GAP — aggiorna verso i gol osservati (o l'xG, se disponibile)
        w = cfg.gap_xg_weight
        th = hg if (hxg is None or np.isnan(hxg)) else (1 - w) * hg + w * hxg
        ta = ag if (axg is None or np.isnan(axg)) else (1 - w) * ag + w * axg

        lr = cfg.gap_lr
        err_h = th - feats["gap_exp_home_goals"]
        err_a = ta - feats["gap_exp_away_goals"]
        hs.gap_att_h += lr * err_h
        as_.gap_def_a += lr * err_h
        as_.gap_att_a += lr * err_a
        hs.gap_def_h += lr * err_a
        for ts in (hs, as_):
            ts.gap_att_h = max(0.05, ts.gap_att_h)
            ts.gap_def_h = max(0.05, ts.gap_def_h)
            ts.gap_att_a = max(0.05, ts.gap_att_a)
            ts.gap_def_a = max(0.05, ts.gap_def_a)

        # forma / contatori
        tot  = hg + ag
        o25  = 1.0 if tot > 2.5 else 0.0
        btts = 1.0 if (hg > 0 and ag > 0) else 0.0
        pts_h = 3.0 if gd > 0 else (1.0 if gd == 0 else 0.0)
        pts_a = 3.0 if gd < 0 else (1.0 if gd == 0 else 0.0)

        rec_h = {"pts": pts_h, "gf": hg, "ga": ag, "o25": o25, "btts": btts}
        rec_a = {"pts": pts_a, "gf": ag, "ga": hg, "o25": o25, "btts": btts}
        hs.form.append(rec_h);      hs.form_home.append(rec_h)
        as_.form.append(rec_a);     as_.form_away.append(rec_a)

        for ts in (hs, as_):
            ts.n_matches += 1
            ts.season_matches += 1
            ts.last_date = kickoff
            ts.recent_dates.append(kickoff)

        # statistiche di lega e globali
        lg.n += 1
        lg.sum_goals += tot
        lg.home_pts  += s
        lg.avg_home_goals += 0.02 * (hg - lg.avg_home_goals)
        lg.avg_away_goals += 0.02 * (ag - lg.avg_away_goals)

        self.global_n += 1
        self.global_goals += (tot - self.global_goals) / min(self.global_n, 5000)
        self.global_home_pts += (s - self.global_home_pts) / min(self.global_n, 5000)

    # ── API ───────────────────────────────────────────────────────────────────
    def transform(self, df, update=True):
        """
        Calcola le feature di rating per ogni riga di `df` (già ordinato per kickoff).

        update=True aggiorna lo stato con i risultati delle partite giocate.
        Le partite non giocate (goals NaN) producono feature ma non aggiornano.
        """
        if df.empty:
            return pd.DataFrame(columns=RATING_FEATURES, index=df.index)

        required = {"home_id", "away_id", "league_key", "kickoff"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"colonne mancanti per RatingEngine: {sorted(missing)}")

        rows = []
        cols = ["home_id", "away_id", "league_key", "tier", "season", "kickoff",
                "home_goals", "away_goals", "home_xg", "away_xg"]
        for c in ["home_xg", "away_xg"]:
            if c not in df.columns:
                df = df.assign(**{c: np.nan})

        for rec in df[cols].itertuples(index=False, name=None):
            (hid, aid, lkey, tier, season, kickoff, hg, ag, hxg, axg) = rec

            lg = self._league(lkey, tier)
            hs = self._team(hid, lkey, tier)
            as_ = self._team(aid, lkey, tier)

            self._season_rollover(hs, season, lg)
            self._season_rollover(as_, season, lg)
            hs.league_key = as_.league_key = lkey

            feats = self._row_features(hs, as_, lg, kickoff)
            feats["tier_num"] = float(tier) if tier is not None and not (
                isinstance(tier, float) and np.isnan(tier)) else np.nan
            rows.append(feats)

            if update and hg is not None and ag is not None and not (
                    np.isnan(hg) or np.isnan(ag)):
                self._update(hs, as_, lg, float(hg), float(ag), hxg, axg, kickoff, feats)

        out = pd.DataFrame(rows, index=df.index)
        return out[RATING_FEATURES]

    def team_snapshot(self):
        """Stato corrente dei rating, utile per diagnostica e report."""
        return pd.DataFrame([
            {
                "team_id": tid, "league_key": ts.league_key, "n_matches": ts.n_matches,
                "elo": ts.elo, "pi_h": ts.pi_h, "pi_a": ts.pi_a,
                "gap_att_h": ts.gap_att_h, "gap_def_h": ts.gap_def_h,
                "gap_att_a": ts.gap_att_a, "gap_def_a": ts.gap_def_a,
            }
            for tid, ts in self.teams.items()
        ])


def build_ratings(df, cfg: RatingConfig = None):
    """Scorciatoia: rating per l'intero dataset in un passaggio."""
    eng = RatingEngine(cfg)
    return eng.transform(df), eng
