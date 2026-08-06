"""
synthetic.py — Generatore di un campionato finto.

Serve a due cose: far girare il progetto quando non si hanno ancora dati, e
dare ai test un terreno in cui si conosce la verità.

Il generatore non produce rumore puro: ogni squadra ha una forza di attacco e
di difesa stabile, più uno stato di forma latente che evolve come una catena
di Markov. È esattamente la struttura che l'HMM dovrebbe recuperare, quindi i
test possono verificare che la recuperi davvero.

Le colonne scritte nel CSV sono quelle di football-data.co.uk (Div, Date,
HomeTeam, AwayTeam, FTHG, FTAG, B365H…), così il file serve anche da prova
del riconoscimento automatico dello schema.
"""

import numpy as np
import pandas as pd

FORM_TRANSITION = np.array([[0.80, 0.18, 0.02],
                            [0.12, 0.76, 0.12],
                            [0.02, 0.18, 0.80]])
FORM_EFFECT  = np.array([-0.30, 0.0, 0.30])
HOME_ADV     = 0.30
BASE_RATE    = 0.15
MARGIN       = 1.06      # margine del bookmaker sintetico


def _poisson_home_win(lambda_home, lambda_away, max_goals=10):
    """Probabilità esatta di vittoria casalinga con due Poisson indipendenti."""
    goals = np.arange(max_goals + 1)
    log_factorial = np.cumsum(np.log(np.maximum(goals, 1)))
    def pmf(lam):
        return np.exp(-lam + goals * np.log(lam) - log_factorial)
    home, away = pmf(lambda_home), pmf(lambda_away)
    return float(np.tril(np.outer(home, away), -1).sum())


def _calendar(teams, rng):
    """
    Calendario di andata e ritorno con il metodo del girone: ogni squadra
    gioca una volta per giornata. Serve che sia un calendario vero e non un
    mucchio di accoppiamenti a caso, altrimenti una squadra giocherebbe più
    partite lo stesso giorno e cose come i giorni di riposo o l'ordine delle
    sequenze perderebbero senso.

    Ritorna una lista di giornate, ciascuna lista di coppie (casa, ospite).
    """
    ruota = list(teams)
    rng.shuffle(ruota)
    if len(ruota) % 2:
        ruota.append(None)          # squadra fittizia: chi la incontra riposa
    meta = len(ruota) // 2

    andata = []
    for giornata in range(len(ruota) - 1):
        partite = []
        for i in range(meta):
            casa, ospite = ruota[i], ruota[-1 - i]
            if casa is None or ospite is None:
                continue
            # alterna il campo, altrimenti le stesse squadre giocherebbero
            # sempre in casa per tutto il girone
            partite.append((casa, ospite) if (giornata + i) % 2 == 0
                           else (ospite, casa))
        andata.append(partite)
        ruota = [ruota[0], ruota[-1]] + ruota[1:-1]

    ritorno = [[(ospite, casa) for casa, ospite in giornata]
               for giornata in andata]
    return andata + ritorno


def generate(n_teams=20, seasons=3, future_matchdays=4, seed=7,
             start="2022-08-06", league="E0"):
    """
    Ritorna un DataFrame in stile football-data.co.uk. Le ultime giornate
    hanno FTHG/FTAG vuoti: sono le partite da prevedere.
    """
    rng = np.random.default_rng(seed)
    teams = [f"Team {i:02d}" for i in range(n_teams)]
    attack  = {t: rng.normal(0, 0.32) for t in teams}
    defence = {t: rng.normal(0, 0.26) for t in teams}
    form    = {t: int(rng.integers(0, 3)) for t in teams}

    rows = []
    day = pd.Timestamp(start)
    first_season = pd.Timestamp(start).year

    for season in range(seasons):
        for team in teams:                       # d'estate la forma si rimescola
            form[team] = int(rng.integers(0, 3))
        for giornata in _calendar(teams, rng):
            for home, away in giornata:
                lam_home = np.exp(BASE_RATE + HOME_ADV + attack[home]
                                  + FORM_EFFECT[form[home]] - defence[away])
                lam_away = np.exp(BASE_RATE + attack[away]
                                  + FORM_EFFECT[form[away]] - defence[home])
                p_home = _poisson_home_win(lam_home, lam_away)
                # il bookmaker conosce la probabilità vera a meno di un errore,
                # e ci aggiunge il margine: battere queste quote deve essere duro
                p_book = float(np.clip(p_home + rng.normal(0, 0.03), 0.05, 0.90))
                rows.append({
                    "Div": league,
                    "Date": day.strftime("%d/%m/%Y"),
                    "Season": f"{first_season + season}/{first_season + season + 1}",
                    "HomeTeam": home,
                    "AwayTeam": away,
                    "FTHG": int(rng.poisson(lam_home)),
                    "FTAG": int(rng.poisson(lam_away)),
                    "B365H": round(1.0 / (p_book * MARGIN), 2),
                    "B365D": round(1.0 / (0.26 * MARGIN), 2),
                    "B365A": round(1.0 / max(0.90 - p_book, 0.05) / MARGIN, 2),
                })
            for team in teams:                   # fra una giornata e l'altra
                form[team] = int(rng.choice(3, p=FORM_TRANSITION[form[team]]))
            day += pd.Timedelta(days=7)
        day += pd.Timedelta(days=45)             # pausa estiva

    df = pd.DataFrame(rows)
    if future_matchdays:
        n_future = future_matchdays * max(1, n_teams // 2)
        df.loc[df.index[-n_future:], ["FTHG", "FTAG"]] = np.nan
    return df


def write_csv(path, **kwargs):
    df = generate(**kwargs)
    df.to_csv(path, index=False)
    return df
