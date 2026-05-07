"""
parse_odds.py — Estrae quote dai JSON globali e le distribuisce per nazione.

Legge:  data/raw_odds/odds_{date}_p{page}.json  (globale)
Incrocia: fixture_id con all_fixtures.csv di ogni nazione
Salva:  data/{nation}/processed/odds.csv

Mercati estratti:
  - 1X2          (bet_id=1)  → q1, qx, q2
  - Over/Under   (bet_id=5)  → odd_o25, odd_o15
  - BTTS         (bet_id=8)  → odd_btts
  - Double Chance (bet_id=12) → odd_1x

Colonne output: fixture_id, q1, qx, q2, imp_h, imp_d, imp_a,
                overround, odd_o25, odd_o15, odd_btts, odd_1x, bookmaker

Uso:
    python parse_odds.py england
    python parse_odds.py --all
"""

import json
import pandas as pd
import sys
from pathlib import Path
from collections import defaultdict

CONFIG_DIR = "config"
RAW_DIR    = Path("data/raw_odds")


def load_config(nation_code):
    with open(Path(CONFIG_DIR) / f"{nation_code}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_raw_odds():
    """
    Carica tutti i file raw_odds in un dizionario fixture_id → dati odds.
    Fatto una volta sola, condiviso tra tutte le nazioni.
    """
    if not RAW_DIR.exists():
        print(f"[ERRORE] {RAW_DIR} non trovata — lancia collect_odds.py prima")
        return {}

    files = sorted(RAW_DIR.glob("odds_*.json"))
    if not files:
        print(f"[ERRORE] Nessun file in {RAW_DIR} — lancia collect_odds.py prima")
        return {}

    print(f"\n  Caricamento odds globali — {len(files)} file...")
    odds_map = {}   # fixture_id → item raw

    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data.get("response", []):
            fid = item.get("fixture", {}).get("id")
            if fid and fid not in odds_map:
                odds_map[fid] = item

    print(f"  Fixture con odds nel pool: {len(odds_map):,}")
    return odds_map


def extract_markets(item):
    """
    Estrae tutti i mercati da un item odds.
    Ritorna dizionario con q1/qx/q2/odd_o25/odd_o15/odd_btts/odd_1x.
    """
    bookmakers = item.get("bookmakers", [])
    if not bookmakers:
        return None

    bm      = bookmakers[0]
    bm_name = bm.get("name", "Unknown")
    bets    = bm.get("bets", [])

    result  = {
        "q1": None, "qx": None, "q2": None,
        "odd_o25": None, "odd_o15": None,
        "odd_btts": None, "odd_1x": None,
        "bookmaker": bm_name,
    }

    for b in bets:
        bid    = b.get("id")
        values = {v["value"]: v["odd"] for v in b.get("values", [])}

        if bid == 1:   # 1X2
            try:
                q1 = float(values.get("Home", 0))
                qx = float(values.get("Draw", 0))
                q2 = float(values.get("Away", 0))
                if q1 > 1 and qx > 1 and q2 > 1:
                    result["q1"] = q1
                    result["qx"] = qx
                    result["q2"] = q2
            except (ValueError, TypeError):
                pass

        elif bid == 5:   # Goals Over/Under
            try:
                v25 = values.get("Over 2.5")
                v15 = values.get("Over 1.5")
                if v25: result["odd_o25"]  = float(v25)
                if v15: result["odd_o15"]  = float(v15)
            except (ValueError, TypeError):
                pass

        elif bid == 8:   # BTTS
            try:
                v = values.get("Yes")
                if v: result["odd_btts"] = float(v)
            except (ValueError, TypeError):
                pass

        elif bid == 12:  # Double Chance
            try:
                v = values.get("Home/Draw")
                if v: result["odd_1x"] = float(v)
            except (ValueError, TypeError):
                pass

    # Validazione minima: serve almeno 1X2
    if result["q1"] is None:
        return None

    return result


def implied_probs(q1, qx, q2):
    try:
        overround = 1/q1 + 1/qx + 1/q2
        return (
            round((1/q1) / overround * 100, 1),
            round((1/qx) / overround * 100, 1),
            round((1/q2) / overround * 100, 1),
            round(overround, 4),
        )
    except Exception:
        return None, None, None, None


def parse_nation(nation_code, odds_map):
    cfg      = load_config(nation_code)
    nation   = cfg["nation"]
    out_path = Path("data") / nation_code / "processed" / "odds.csv"
    fix_path = Path("data") / nation_code / "processed" / "all_fixtures.csv"

    if not fix_path.exists():
        print(f"  [{nation}] all_fixtures.csv non trovato — skip")
        return

    fixtures_df = pd.read_csv(fix_path)
    if "fixture_id" not in fixtures_df.columns:
        print(f"  [{nation}] colonna fixture_id mancante — skip")
        return

    fixture_ids = set(fixtures_df["fixture_id"].dropna().astype(int))
    rows        = []
    matched     = 0
    no_markets  = 0

    for fid in fixture_ids:
        item = odds_map.get(fid)
        if item is None:
            continue
        markets = extract_markets(item)
        if markets is None:
            no_markets += 1
            continue

        imp_h, imp_d, imp_a, overround = implied_probs(
            markets["q1"], markets["qx"], markets["q2"]
        )
        rows.append({
            "fixture_id": fid,
            "q1":         markets["q1"],
            "qx":         markets["qx"],
            "q2":         markets["q2"],
            "imp_h":      imp_h,
            "imp_d":      imp_d,
            "imp_a":      imp_a,
            "overround":  overround,
            "odd_o25":    markets["odd_o25"],
            "odd_o15":    markets["odd_o15"],
            "odd_btts":   markets["odd_btts"],
            "odd_1x":     markets["odd_1x"],
            "bookmaker":  markets["bookmaker"],
        })
        matched += 1

    if rows:
        pd.DataFrame(rows).to_csv(out_path, index=False)
        pct = round(matched / len(fixture_ids) * 100, 1)
        print(f"  {nation:<14}  {len(fixture_ids):>4} fixture  "
              f"{matched:>4} con odds ({pct}%)  "
              f"[O25:{sum(1 for r in rows if r['odd_o25'])}  "
              f"BTTS:{sum(1 for r in rows if r['odd_btts'])}  "
              f"1X:{sum(1 for r in rows if r['odd_1x'])}]")
    else:
        print(f"  {nation:<14}  nessuna quota trovata nel pool")


def main():
    args = sys.argv[1:]
    if not args:
        print("Uso: python parse_odds.py <nazione>  oppure  --all")
        sys.exit(0)

    nations = ([p.stem for p in Path(CONFIG_DIR).glob("*.json")]
               if args[0] == "--all" else args)

    print(f"\n{'='*65}")
    print(f"  PARSE ODDS — distribuzione per nazione")
    print(f"{'='*65}")

    # Carica il pool globale UNA VOLTA SOLA
    odds_map = load_raw_odds()
    if not odds_map:
        sys.exit(1)

    print(f"\n  {'Nazione':<14}  {'Fix tot':>8}  {'Con odds':>9}  Mercati extra")
    print(f"  {'─'*55}")
    for nation_code in nations:
        parse_nation(nation_code, odds_map)

    print(f"\n{'='*65}")
    print(f"  Prossimo: python predict.py --all")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()