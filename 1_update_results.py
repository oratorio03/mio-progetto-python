"""
update.py — Aggiorna risultati stagione corrente per una o più nazioni.
Scarica solo la stagione in corso, aggiorna all_results.csv e all_fixtures.csv.

Uso:
    python update.py england
    python update.py england spain italy france germany
    python update.py --all
"""

import json
import pandas as pd
import sys
import time
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from api_utils import make_api_request

CONFIG_DIR    = "config"
STATUS_PLAYED = {"FT", "AET", "PEN", "AWD", "WO"}

FIELDNAMES = [
    "fixture_id","league_id","league_name","tier","season","round",
    "date","time","status","played",
    "home_id","home_name","away_id","away_name",
    "home_goals","away_goals","home_ht","away_ht","home_2h","away_2h",
    "total_goals","result_1x2","over15","over25","over35","btts"
]

def load_config(nation_code):
    with open(Path(CONFIG_DIR) / f"{nation_code}.json", "r", encoding="utf-8") as f:
        return json.load(f)

def get_paths(nation_code):
    proc = Path("data") / nation_code / "processed"
    return proc / "all_results.csv", proc / "all_fixtures.csv"

def fetch_season(league_id, season):
    url    = "https://v3.football.api-sports.io/fixtures"
    params = {"league": league_id, "season": season}
    
    data = make_api_request(url, params=params)
    
    if data is None:
        return []
        
    if data.get("errors"):
        print(f"    [ERRORE API] {data['errors']}")
        return []
        
    return data.get("response", [])

def parse_fixture(fix, league_id, league_name, tier, season):
    f  = fix.get("fixture", {})
    l  = fix.get("league",  {})
    t  = fix.get("teams",   {})
    g  = fix.get("goals",   {})
    s  = fix.get("score",   {})

    status   = f.get("status", {}).get("short", "")
    played   = status in STATUS_PLAYED
    raw_date = f.get("date", "")
    date_str = raw_date[:10] if raw_date else ""
    time_str = raw_date[11:16] if len(raw_date) > 10 else "00:00"

    home_goals = g.get("home")
    away_goals = g.get("away")
    ht         = s.get("halftime", {})
    home_ht    = ht.get("home")
    away_ht    = ht.get("away")

    total_goals = result_1x2 = over15 = over25 = over35 = btts = None
    home_2h = away_2h = None

    if played and home_goals is not None and away_goals is not None:
        total_goals = home_goals + away_goals
        over15      = 1 if total_goals > 1 else 0
        over25      = 1 if total_goals > 2 else 0
        over35      = 1 if total_goals > 3 else 0
        btts        = 1 if home_goals > 0 and away_goals > 0 else 0
        result_1x2  = "H" if home_goals > away_goals else ("A" if home_goals < away_goals else "D")
        if home_ht is not None and away_ht is not None:
            home_2h = home_goals - home_ht
            away_2h = away_goals - away_ht

    return {
        "fixture_id":  f.get("id"),
        "league_id":   league_id,
        "league_name": league_name,
        "tier":        tier,
        "season":      season,
        "round":       (l.get("round") or "").replace("Regular Season - ", "R"),
        "date":        date_str,
        "time":        time_str,
        "status":      status,
        "played":      1 if played else 0,
        "home_id":     t.get("home", {}).get("id"),
        "home_name":   t.get("home", {}).get("name"),
        "away_id":     t.get("away", {}).get("id"),
        "away_name":   t.get("away", {}).get("name"),
        "home_goals":  home_goals,
        "away_goals":  away_goals,
        "home_ht":     home_ht,
        "away_ht":     away_ht,
        "home_2h":     home_2h,
        "away_2h":     away_2h,
        "total_goals": total_goals,
        "result_1x2":  result_1x2,
        "over15":      over15,
        "over25":      over25,
        "over35":      over35,
        "btts":        btts,
    }

def update_nation(nation_code):
    cfg     = load_config(nation_code)
    nation  = cfg["nation"]
    leagues = cfg["leagues"]
    season  = cfg["current_season"]

    res_path, fix_path = get_paths(nation_code)

    if not res_path.exists() or not fix_path.exists():
        print(f"  [{nation}] CSV non trovati — lancia prima parse_data.py {nation_code}")
        return 0, 0

    # Carica CSV esistenti
    df_res = pd.read_csv(res_path)
    df_fix = pd.read_csv(fix_path)

    # Set fixture_id già in results per deduplicazione
    ids_results = set(df_res["fixture_id"].astype(str))

    nuovi_results  = []
    nuove_fixtures = []
    api_calls      = 0

    for league_id_str, info in leagues.items():
        league_id   = int(league_id_str)
        league_name = info["name"]
        tier        = info["tier"]

        print(f"    {league_name} (season {season})...", end=" ", flush=True)
        
        # Ora passa dal modulo centralizzato
        fixtures = fetch_season(league_id, season)
        
        api_calls += 1
        time.sleep(0.3)

        n_played = n_future = 0
        for fix in fixtures:
            row = parse_fixture(fix, league_id, league_name, tier, season)
            fid = str(row["fixture_id"])

            if row["played"]:
                n_played += 1
                if fid not in ids_results:
                    nuovi_results.append(row)
            else:
                n_future += 1
                nuove_fixtures.append(row)

        print(f"{n_played} giocate | {n_future} future")

    # Aggiorna all_results.csv — aggiungi nuove partite giocate
    n_aggiunti = len(nuovi_results)
    if n_aggiunti > 0:
        df_nuovi  = pd.DataFrame(nuovi_results)[FIELDNAMES]
        df_res    = pd.concat([df_res, df_nuovi], ignore_index=True)
        df_res.drop_duplicates(subset=["fixture_id"], keep="last", inplace=True)
        df_res.to_csv(res_path, index=False)

    # Sostituisci all_fixtures.csv con le partite future aggiornate
    if nuove_fixtures:
        df_fix_new = pd.DataFrame(nuove_fixtures)[FIELDNAMES]
        df_fix_new.to_csv(fix_path, index=False)
        n_future_tot = len(df_fix_new)
    else:
        n_future_tot = 0

    print(f"    → {n_aggiunti} nuovi risultati aggiunti | {n_future_tot} fixture future")
    return api_calls, n_aggiunti

def main():
    args = sys.argv[1:]
    if not args:
        print("Uso: python update.py <nazione> [nazione2 ...]  oppure  python update.py --all")
        sys.exit(0)

    nations = [p.stem for p in Path(CONFIG_DIR).glob("*.json")] if args[0] == "--all" else args

    print(f"\n{'='*60}")
    print(f"  UPDATE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Nazioni: {', '.join(nations)}")
    print(f"{'='*60}")

    total_calls   = 0
    total_nuovi   = 0

    for nation_code in nations:
        print(f"\n── {nation_code.upper()} ──")
        calls, nuovi = update_nation(nation_code)
        total_calls += calls
        total_nuovi += nuovi

    print(f"\n{'='*60}")
    print(f"  Chiamate API usate (in questo script):  {total_calls}")
    print(f"  Nuovi risultati:     {total_nuovi}")
    print(f"{'='*60}")

    if total_nuovi > 0:
        print(f"\n  Prossimi passi consigliati:")
        for n in nations:
            print(f"    python team_stats.py {n}")
            print(f"    python variance.py {n}")
        print(f"\n  Poi: python feedback.py output/<file>.xlsx")

if __name__ == "__main__":
    main()