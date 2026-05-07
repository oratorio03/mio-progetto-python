import json
import time
import sys
import csv
from pathlib import Path
from datetime import datetime, timedelta

# Inseriamo il percorso della root per caricare api_utils
sys.path.append(str(Path(__file__).parent.parent))
from api_utils import make_api_request

CONFIG_DIR = "config"
DAYS_AHEAD = 3 # Coerente con lo script delle quote

def load_config(nation_code):
    with open(Path(CONFIG_DIR) / f"{nation_code}.json", "r", encoding="utf-8") as f:
        return json.load(f)

def get_active_leagues(nation_code):
    """Ritorna un set di league_id che hanno partite nei prossimi X giorni."""
    fixtures_path = Path("data") / nation_code / "processed" / "all_fixtures.csv"
    if not fixtures_path.exists():
        return set()

    active_leagues = set()
    now = datetime.now()
    limit_date = now + timedelta(days=DAYS_AHEAD)

    with open(fixtures_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                f_date = datetime.strptime(row["date"], "%Y-%m-%d")
                if row["played"] == "0" and now <= f_date <= limit_date:
                    active_leagues.add(int(row["league_id"]))
            except:
                continue
    return active_leagues

def fetch_standings(league_id, season):
    url    = "https://v3.football.api-sports.io/standings"
    params = {"league": league_id, "season": season}
    data = make_api_request(url, params=params)
    if data is None or data.get("errors"):
        return None
    return data.get("response", [])

def collect_nation(nation_code):
    cfg     = load_config(nation_code)
    nation  = cfg["nation"]
    season  = cfg["current_season"]
    leagues = cfg["leagues"]
    out_dir = Path("data") / nation_code / "raw" / "standings"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Trova quali leghe servono davvero per questo weekend
    active_ids = get_active_leagues(nation_code)

    print(f"\n── {nation.upper()} ──")
    if not active_ids:
        print("  Nessuna partita in programma nei prossimi 3 giorni. Salto.")
        return 0

    total_calls = 0
    for league_id_str, info in leagues.items():
        league_id   = int(league_id_str)
        league_name = info["name"]
        
        # Filtro: Se la lega non ha partite, non scaricare la classifica
        if league_id not in active_ids:
            # print(f"  {league_name:<30} [IDLE - No Match]")
            continue

        out_path = out_dir / f"standings_{league_id}_s{season}.json"
        
        response = fetch_standings(league_id, season)
        total_calls += 1
        time.sleep(0.4)

        if response:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(response, f)
            print(f"  {league_name:<30} OK (Aggiornata)")

    return total_calls

def main():
    args = sys.argv[1:]
    if not args:
        print("Uso: python 6_collect_standings.py <nazione> oppure --all")
        sys.exit(0)

    nations = [p.stem for p in Path(CONFIG_DIR).glob("*.json")] if args[0] == "--all" else args
    
    print(f"\n{'='*60}\n  COLLECT STANDINGS v5 (Smart Filter)\n{'='*60}")
    
    total = sum(collect_nation(n) for n in nations)
    print(f"\nTOTAL API CALLS: {total}")

if __name__ == "__main__":
    main()