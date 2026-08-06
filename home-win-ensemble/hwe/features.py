"""
features.py — Costruzione delle feature, in una sola passata cronologica.

Regola non negoziabile: la riga della partita i contiene solo informazione
disponibile prima del fischio d'inizio di i. Per ogni partita si emettono
prima le feature dallo stato corrente, poi si aggiorna lo stato con il
risultato. Le partite future entrano nella stessa passata ma non aggiornano
mai lo stato, quindi non contaminano il passato né si contaminano fra loro.
"""

from collections import defaultdict, deque

import numpy as np
import pandas as pd

# ── parametri ─────────────────────────────────────────────────────────────────

ELO_START      = 1500.0
ELO_K          = 20.0
ELO_HOME_ADV   = 60.0
ELO_SEASON_REG = 0.75      # a inizio stagione i rating tornano verso la media
ELO_SCALE      = 400.0

WIN_VENUE      = 6         # finestra rolling sulle partite nello stesso campo
WIN_ALL        = 5         # finestra rolling sulla forma generale
WIN_H2H        = 6         # scontri diretti considerati
MIN_HIST       = 3         # sotto questa soglia la feature resta NaN
PRIOR_HOME_WIN = 0.45      # prior prima di avere abbastanza partite di lega
REST_CLIP      = (1.0, 21.0)

FEATURES = [
    "elo_diff",              # differenza di rating
    "elo_expected",          # attesa Elo di vittoria casalinga, campo incluso
    "home_ppg_venue",        # punti/partita in casa
    "away_ppg_venue",        # punti/partita in trasferta
    "home_ppg_recent",       # punti/partita, ultime partite ovunque
    "away_ppg_recent",
    "home_gf_venue",
    "home_ga_venue",
    "away_gf_venue",
    "away_ga_venue",
    "goal_diff_edge",        # differenza reti casa - differenza reti ospite
    "home_win_rate_venue",
    "away_loss_rate_venue",
    "home_win_streak",       # vittorie interne consecutive (max 3)
    "away_loss_streak",      # sconfitte esterne consecutive (max 3)
    "h2h_home_rate",
    "h2h_count",
    "rest_home",
    "rest_away",
    "rest_edge",
    "league_home_rate",      # frequenza vittorie casalinghe della lega, a oggi
    "season_progress",
    "history_home",          # quanta storia abbiamo su questa squadra (0-1)
    "history_away",
]

# Le feature devono essere limitate. Un conteggio che cresce all'infinito (le
# partite viste finora) è un trend temporale travestito: in training assume
# certi valori, in predizione valori che non ha mai visto, e il modello lo
# estrapola. Misurato: le probabilità medie salivano a 0.51 dove la frequenza
# reale era 0.42. Qui il conteggio è saturato, così dice "ho poca storia su
# questa squadra" senza portarsi dietro la data.
HISTORY_CAP = 40

META = ["match_id", "date", "league", "season", "home", "away", "played",
        "odds_home", "odds_draw", "odds_away"]


# ── Elo ───────────────────────────────────────────────────────────────────────

def elo_expected(rating_home, rating_away):
    return 1.0 / (1.0 + 10.0 ** ((rating_away - (rating_home + ELO_HOME_ADV))
                                 / ELO_SCALE))


class EloBook:
    """Rating per (lega, squadra), con regressione verso la media a ogni stagione."""

    def __init__(self):
        self.rating = defaultdict(lambda: ELO_START)
        self._season = {}

    def _roll(self, league, season):
        if self._season.get(league) == season:
            return
        self._season[league] = season
        for key in [k for k in self.rating if k[0] == league]:
            self.rating[key] = ELO_START + ELO_SEASON_REG * (self.rating[key] - ELO_START)

    def get(self, league, season, team):
        self._roll(league, season)
        return self.rating[(league, team)]

    def update(self, league, home, away, outcome):
        """outcome: 1.0 vince la casa, 0.5 pareggio, 0.0 vince l'ospite."""
        rh, ra = self.rating[(league, home)], self.rating[(league, away)]
        delta = ELO_K * (outcome - elo_expected(rh, ra))
        self.rating[(league, home)] = rh + delta
        self.rating[(league, away)] = ra - delta


# ── helper ────────────────────────────────────────────────────────────────────

def _avg(values, min_n=MIN_HIST):
    return float(np.mean(values)) if len(values) >= min_n else np.nan


def _streak(flags, cap=3):
    n = 0
    for value in reversed(flags):
        if value != 1:
            break
        n += 1
        if n >= cap:
            break
    return float(n)


def _points(scored, conceded):
    return 3.0 if scored > conceded else (1.0 if scored == conceded else 0.0)


# ── tabella ───────────────────────────────────────────────────────────────────

def build(matches):
    """
    matches: DataFrame nello schema di data.load_matches (giocate + future).
    Ritorna META + FEATURES + `target` (1 = vittoria casa, NaN se da giocare).
    """
    df = matches.sort_values(["date", "played", "match_id"],
                             ascending=[True, False, True])

    elo        = EloBook()
    venue_home = defaultdict(lambda: deque(maxlen=WIN_VENUE))
    venue_away = defaultdict(lambda: deque(maxlen=WIN_VENUE))
    recent     = defaultdict(lambda: deque(maxlen=WIN_ALL))
    h2h        = defaultdict(lambda: deque(maxlen=WIN_H2H))
    last_seen  = {}
    seen_count = defaultdict(int)
    season_cnt = defaultdict(int)
    league_tally = defaultdict(lambda: [0, 0])

    rows = []
    for m in df.itertuples(index=False):
        league, season, home, away = m.league, m.season, m.home, m.away

        rating_home = elo.get(league, season, home)
        rating_away = elo.get(league, season, away)

        vh, va = venue_home[home], venue_away[away]
        rh_form, ra_form = recent[home], recent[away]

        wins, total = league_tally[league]
        league_rate = wins / total if total >= 50 else PRIOR_HOME_WIN

        pair = (home, away) if home < away else (away, home)
        # gli scontri diretti vanno riportati al punto di vista di chi è in casa oggi
        h2h_values = [v if side == home else 1.0 - v for side, v in h2h[pair]]

        rest_home = rest_away = np.nan
        if home in last_seen:
            rest_home = float(np.clip((m.date - last_seen[home]).days, *REST_CLIP))
        if away in last_seen:
            rest_away = float(np.clip((m.date - last_seen[away]).days, *REST_CLIP))

        gf_h, ga_h = _avg([x["gf"] for x in vh]), _avg([x["ga"] for x in vh])
        gf_a, ga_a = _avg([x["gf"] for x in va]), _avg([x["ga"] for x in va])
        edge = np.nan
        if not any(np.isnan(v) for v in (gf_h, ga_h, gf_a, ga_a)):
            edge = (gf_h - ga_h) - (gf_a - ga_a)

        rows.append({
            "match_id": m.match_id, "date": m.date, "league": league,
            "season": season, "home": home, "away": away, "played": m.played,
            "odds_home": m.odds_home, "odds_draw": m.odds_draw,
            "odds_away": m.odds_away,
            "target": (np.nan if m.played != 1
                       else float(m.home_goals > m.away_goals)),

            "elo_diff":             rating_home - rating_away,
            "elo_expected":         elo_expected(rating_home, rating_away),
            "home_ppg_venue":       _avg([x["pts"] for x in vh]),
            "away_ppg_venue":       _avg([x["pts"] for x in va]),
            "home_ppg_recent":      _avg([x for x in rh_form]),
            "away_ppg_recent":      _avg([x for x in ra_form]),
            "home_gf_venue":        gf_h,
            "home_ga_venue":        ga_h,
            "away_gf_venue":        gf_a,
            "away_ga_venue":        ga_a,
            "goal_diff_edge":       edge,
            "home_win_rate_venue":  _avg([x["win"] for x in vh]),
            "away_loss_rate_venue": _avg([x["loss"] for x in va]),
            "home_win_streak":      _streak([x["win"] for x in vh]),
            "away_loss_streak":     _streak([x["loss"] for x in va]),
            "h2h_home_rate":        (float(np.mean(h2h_values))
                                     if h2h_values else np.nan),
            "h2h_count":            float(len(h2h_values)),
            "rest_home":            rest_home,
            "rest_away":            rest_away,
            "rest_edge":            (rest_home - rest_away
                                     if not (np.isnan(rest_home) or np.isnan(rest_away))
                                     else np.nan),
            "league_home_rate":     league_rate,
            "season_progress":      min(season_cnt[(season, home)], 38) / 38.0,
            "history_home":         min(seen_count[home], HISTORY_CAP) / HISTORY_CAP,
            "history_away":         min(seen_count[away], HISTORY_CAP) / HISTORY_CAP,
        })

        # ── stato: lo aggiornano solo le partite davvero giocate ──────────────
        if m.played != 1:
            continue

        hg, ag = float(m.home_goals), float(m.away_goals)
        outcome = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        elo.update(league, home, away, outcome)

        vh.append({"gf": hg, "ga": ag, "pts": _points(hg, ag),
                   "win": int(hg > ag), "loss": int(hg < ag)})
        va.append({"gf": ag, "ga": hg, "pts": _points(ag, hg),
                   "win": int(ag > hg), "loss": int(ag < hg)})
        rh_form.append(_points(hg, ag))
        ra_form.append(_points(ag, hg))
        h2h[pair].append((home, outcome))

        last_seen[home] = last_seen[away] = m.date
        seen_count[home] += 1
        seen_count[away] += 1
        season_cnt[(season, home)] += 1
        season_cnt[(season, away)] += 1
        league_tally[league][0] += int(hg > ag)
        league_tally[league][1] += 1

    table = pd.DataFrame(rows, columns=META + ["target"] + FEATURES)
    # mergesort è stabile: l'ordine dentro la stessa data resta quello della
    # passata, quindi la riga i non cambia se il file continua oltre
    return table.sort_values("date", kind="mergesort").reset_index(drop=True)


class Imputer:
    """Mediane stimate sul solo training e riusate identiche in predizione."""

    def __init__(self):
        self.medians = {}

    def fit(self, table):
        for col in FEATURES:
            value = pd.to_numeric(table[col], errors="coerce").median()
            self.medians[col] = 0.0 if pd.isna(value) else float(value)
        return self

    def transform(self, table):
        out = table[FEATURES].apply(pd.to_numeric, errors="coerce")
        return out.fillna(self.medians).to_numpy(dtype=float)

    def fit_transform(self, table):
        return self.fit(table).transform(table)
