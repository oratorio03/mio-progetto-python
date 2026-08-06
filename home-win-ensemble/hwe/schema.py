"""
schema.py — Riconoscimento delle colonne di un CSV di partite.

Il progetto non impone un formato: prende un CSV qualunque di risultati e
cerca di capire da solo quali colonne sono la data, le squadre e i gol.
Funziona senza configurazione con i nomi più diffusi (football-data.co.uk,
export di API, fogli fatti a mano in italiano o inglese), e quando non basta
si forza a mano con --map.

Colonne canoniche:
    obbligatorie   date, home, away
    risultato      home_goals, away_goals   (assenti/vuote = partita da giocare)
    facoltative    league, season, match_id, odds_home, odds_draw, odds_away
"""

import re

REQUIRED = ("date", "home", "away")
RESULT   = ("home_goals", "away_goals")
OPTIONAL = ("league", "season", "match_id",
            "odds_home", "odds_draw", "odds_away")
CANONICAL = REQUIRED + RESULT + OPTIONAL

# I nomi sono confrontati dopo normalizzazione: minuscole, senza spazi,
# underscore, punti e trattini. "Home Team", "home_team" e "HomeTeam" collidono
# tutti su "hometeam".
ALIASES = {
    "date": ["date", "data", "matchdate", "kickoff", "kickofftime",
             "datetime", "giorno", "datapartita"],
    "home": ["home", "hometeam", "hometeamname", "hometeamnamefull",
             "hometeamn", "homename", "squadracasa", "casa", "team1",
             "hometeamid", "localteam"],
    "away": ["away", "awayteam", "awayteamname", "awayname", "squadratrasferta",
             "trasferta", "ospite", "team2", "awayteamid", "visitorteam"],
    "home_goals": ["homegoals", "fthg", "homescore", "goalcasa", "golcasa",
                   "hg", "scorehome", "homefulltimegoals", "fthomegoals",
                   "homeft", "goalshome"],
    "away_goals": ["awaygoals", "ftag", "awayscore", "goaltrasferta",
                   "goltrasferta", "ag", "scoreaway", "awayfulltimegoals",
                   "ftawaygoals", "awayft", "goalsaway"],
    "league":  ["league", "leaguename", "div", "division", "campionato",
                "competition", "lega", "torneo"],
    "season":  ["season", "stagione", "year", "anno"],
    "match_id": ["matchid", "fixtureid", "id", "gameid", "idpartita"],
    "odds_home": ["oddshome", "b365h", "psh", "pshome", "avgh", "bwh",
                  "quota1", "q1", "quotacasa", "odd1", "homeodds"],
    "odds_draw": ["oddsdraw", "b365d", "psd", "psdraw", "avgd", "bwd",
                  "quotax", "qx", "quotapareggio", "oddx", "drawodds"],
    "odds_away": ["oddsaway", "b365a", "psa", "psaway", "avga", "bwa",
                  "quota2", "q2", "quotatrasferta", "odd2", "awayodds"],
}


def normalise(name):
    return re.sub(r"[\s_\.\-/]+", "", str(name)).strip().lower()


class SchemaError(ValueError):
    """Il CSV non contiene le colonne minime per lavorare."""


def detect(columns, overrides=None):
    """
    columns   : nomi delle colonne del CSV
    overrides : {canonica: nome colonna} per forzare a mano

    Ritorna {canonica: nome colonna reale}. Solleva SchemaError se mancano
    date/home/away, con un messaggio che dice cosa è stato trovato.
    """
    columns = list(columns)
    by_norm = {}
    for col in columns:
        by_norm.setdefault(normalise(col), col)

    mapping = {}
    for canon in CANONICAL:
        # 1. il nome canonico stesso
        if normalise(canon) in by_norm:
            mapping[canon] = by_norm[normalise(canon)]
            continue
        # 2. gli alias noti, in ordine di preferenza
        for alias in ALIASES.get(canon, []):
            if alias in by_norm:
                mapping[canon] = by_norm[alias]
                break

    for canon, col in (overrides or {}).items():
        if canon not in CANONICAL:
            raise SchemaError(
                f"colonna canonica sconosciuta: {canon!r}. "
                f"Ammesse: {', '.join(CANONICAL)}")
        if col not in columns:
            raise SchemaError(
                f"la colonna {col!r} non esiste nel file. "
                f"Colonne disponibili: {', '.join(map(str, columns))}")
        mapping[canon] = col

    missing = [c for c in REQUIRED if c not in mapping]
    if missing:
        raise SchemaError(
            "impossibile riconoscere le colonne " + ", ".join(missing) +
            ".\nColonne trovate nel file: " + ", ".join(map(str, columns)) +
            "\nForza la corrispondenza con --map, per esempio: "
            + " ".join(f"--map {c}=NOME_COLONNA" for c in missing))

    return mapping


def describe(mapping, columns):
    """Righe leggibili su come è stato interpretato il file."""
    lines = []
    for canon in CANONICAL:
        if canon in mapping:
            lines.append(f"  {canon:11s} ← {mapping[canon]}")
    ignored = [c for c in columns if c not in set(mapping.values())]
    if ignored:
        lines.append(f"  (ignorate: {', '.join(map(str, ignored))})")
    return "\n".join(lines)


def parse_overrides(pairs):
    """Da ['home=HomeTeam', 'date=Data'] a {'home': 'HomeTeam', ...}."""
    out = {}
    for item in pairs or []:
        if "=" not in item:
            raise SchemaError(f"--map vuole la forma canonica=colonna, ricevuto {item!r}")
        canon, col = item.split("=", 1)
        out[canon.strip()] = col.strip()
    return out
