"""
parse_standings.py — Estrae i dati di classifica dai JSON raw.
v5 — Robustezza calcoli e allineamento nomi config.
"""

import json
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

CONFIG_DIR = "config"

def load_config(nation_code):
    with open(Path(CONFIG_DIR) / f"{nation_code}.json", "r", encoding="utf-8") as f:
        return json.load(f)

def parse_nation(nation_code):
    cfg      = load_config(nation_code)
    nation   = cfg["nation"]
    season   = cfg["current_season"]
    leagues  = cfg.get("leagues", {})

    raw_dir  = Path("data") / nation_code / "raw" / "standings"
    out_path = Path("data") / nation_code / "processed" / "standings.csv"

    if not raw_dir.exists():
        print(f"  [{nation}] Cartella raw non trovata. Salto.")
        return

    print(f"\n── {nation.upper()} ──")
    rows = []

    for league_id_str, info in leagues.items():
        league_id   = int(league_id_str)
        league_name = info.get("name", f"League {league_id}")
        json_path   = raw_dir / f"standings_{league_id}_s{season}.json"

        if not json_path.exists():
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            response = json.load(f)

        if not response or "league" not in response[0]:
            continue

        league_data    = response[0].get("league", {})
        standings_list = league_data.get("standings", [[]])

        for group in standings_list:
            total_teams = len(group)
            for entry in group:
                team   = entry.get("team", {})
                all_s  = entry.get("all", {})
                goals  = all_s.get("goals", {})
                played = all_s.get("played", 0)

                # Calcolo partite totali teoriche (doppio turno)
                total_season_games = (total_teams - 1) * 2
                pct_played = round(played / total_season_games, 3) if total_season_games > 0 else 0

                rank   = entry.get("rank", 0)
                points = entry.get("points", 0)
                gf     = goals.get("for", 0)
                ga     = goals.get("against", 0)

                rows.append({
                    "league_id":        league_id,
                    "league_name":      league_name,
                    "team_id":          team.get("id"),
                    "team_name":        team.get("name"),
                    "rank":             rank,
                    "points":           points,
                    "played":           played,
                    "won":              all_s.get("win",  0),
                    "drawn":            all_s.get("draw", 0),
                    "lost":             all_s.get("lose", 0),
                    "goals_for":        gf,
                    "goals_against":    ga,
                    "goal_diff":        gf - ga,
                    "form_last5":       entry.get("form", ""),
                    "total_teams":      total_teams,
                    "rank_from_bottom": total_teams - rank + 1,
                    "pct_played":       pct_played,
                    "pts_per_game":     round(points / played, 3) if played > 0 else 0,
                    "updated_at":       datetime.now().strftime("%Y-%m-%d"),
                })

    if rows:
        df = pd.DataFrame(rows)
        # Assicura che la cartella processed esista
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"  [OK] Classifiche salvate: {len(df)} righe.")
    else:
        print(f"  [!] Nessun dato utile estratto per {nation}.")

def main():
    args = sys.argv[1:]
    nations = ([p.stem for p in Path(CONFIG_DIR).glob("*.json")]
               if not args or args[0] == "--all" else args)

    print(f"\n{'='*60}")
    print(f"  PARSE STANDINGS v5")
    print(f"{'='*60}")

    for nation_code in nations:
        parse_nation(nation_code)

if __name__ == "__main__":
    main()