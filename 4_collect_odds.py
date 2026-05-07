"""
collect_odds.py — Scarica quote da API-Football con approccio globale per data.

Strategia RISPARMIO CHIAMATE:
  - Una chiamata per data (non per lega) → /odds?date=X&bookmaker=8&page=N
  - Il server filtra per bookmaker lato API → payload minimo
  - 7 giorni × ~10 pagine = ~70 chiamate totali (vs centinaia con approccio per lega)
  - Mercati catturati in un colpo solo: 1X2, Over/Under, BTTS, Doppia Chance

Salva:  data/raw_odds/odds_{date}_p{page}.json  (cartella globale, non per nazione)
Parse: parse_odds.py incrocia fixture_id con i fixture di ogni nazione

Uso:
    python collect_odds.py
    python collect_odds.py --days=10
    python collect_odds.py --from=2026-03-09 --to=2026-03-15
    python collect_odds.py --clean
"""

import requests
import json
import time
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

API_KEY  = "8d1986998b5a73bfedddd7e2ae493b74"
BASE_URL = "https://v3.football.api-sports.io"
HEADERS  = {
    "x-rapidapi-key":  API_KEY,
    "x-rapidapi-host": "v3.football.api-sports.io"
}

RAW_DIR      = Path("data/raw_odds")   # cartella globale condivisa tra nazioni
SLEEP_SEC    = 0.35
DEFAULT_DAYS = 7
MAX_RETRY    = 5

# Bet ID → nome mercato (da API-Football)
BET_IDS = {
    1:  "1X2",
    5:  "Goals Over/Under",
    8:  "Both Teams Score",
    12: "Double Chance",
}


# ── FETCH ────────────────────────────────────────────────────────────────────
def fetch_odds_page(target_date: date, page: int):
    """
    GET /odds?date=YYYY-MM-DD&bookmaker=8&page=N
    Bookmaker=8 (Bet365) filtrato server-side → payload minimo.
    5 retry con backoff progressivo.
    """
    params = {
        "date":       str(target_date),
        "bookmaker":  8,   # Bet365 — filtro server-side
        "page":       page,
    }
    for attempt in range(MAX_RETRY):
        try:
            r = requests.get(
                f"{BASE_URL}/odds",
                headers=HEADERS,
                params=params,
                timeout=20
            )
            if r.status_code != 200:
                print(f"\n    [HTTP {r.status_code}] retry {attempt+1}/{MAX_RETRY}")
                time.sleep(2 + attempt)
                continue
            data = r.json()
            if data.get("errors"):
                err = data["errors"]
                if "rateLimit" in str(err):
                    print(f"\n    [RATELIMIT] sleep 60s")
                    time.sleep(60)
                    continue
                print(f"\n    [API ERROR] {err}")
                return None, 0
            total_pages = data.get("paging", {}).get("total", 1)
            results     = data.get("response", [])
            return results, total_pages
        except Exception as e:
            print(f"\n    [EXCEPTION] {e} retry {attempt+1}/{MAX_RETRY}")
            time.sleep(2 + attempt)
    print("\n    [FAILED AFTER RETRIES]")
    return None, 0


# ── COLLECT PER DATA ──────────────────────────────────────────────────────────
def collect_date(target_date: date, force: bool = False) -> int:
    """
    Scarica tutte le pagine di odds per una data.
    Ritorna il numero di fixture trovati.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    page        = 1
    total_pages = 1
    total_fix   = 0
    calls       = 0

    while page <= total_pages:
        out_path = RAW_DIR / f"odds_{target_date}_p{page}.json"

        if out_path.exists() and not force:
            with open(out_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            results     = saved.get("response", [])
            total_pages = saved.get("total_pages", 1)
            print(f"    [cache] {target_date} p{page}/{total_pages} — {len(results)} fixture")
        else:
            results, total_pages = fetch_odds_page(target_date, page)
            calls += 1
            time.sleep(SLEEP_SEC)

            if results is None:
                break

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "response":    results,
                    "total_pages": total_pages,
                    "fetched_at":  datetime.now().isoformat(),
                }, f)
            print(f"    {target_date} p{page}/{total_pages} — {len(results)} fixture  [{calls} call]")

        total_fix += len(results)

        if page >= total_pages or not results:
            break
        page += 1

    return total_fix, calls


# ── CLEAN ────────────────────────────────────────────────────────────────────
def clean_raw():
    if not RAW_DIR.exists():
        return
    files = list(RAW_DIR.glob("odds_*.json"))
    for f in files:
        f.unlink()
    print(f"  Pulizia: {len(files)} file eliminati da {RAW_DIR}")


# ── PARSE ARGS ────────────────────────────────────────────────────────────────
def parse_args(args_raw):
    params = {"days": DEFAULT_DAYS, "from": None, "to": None,
              "clean": False, "force": False}
    for a in args_raw:
        if a == "--clean":
            params["clean"] = True
        elif a == "--force":
            params["force"] = True
        elif a.startswith("--days="):
            params["days"] = int(a.split("=")[1])
        elif a.startswith("--from="):
            params["from"] = date.fromisoformat(a.split("=")[1])
        elif a.startswith("--to="):
            params["to"] = date.fromisoformat(a.split("=")[1])
    return params


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    args_raw = sys.argv[1:]
    if "--help" in args_raw:
        print(__doc__)
        sys.exit(0)

    params = parse_args(args_raw)

    if params["clean"]:
        clean_raw()
        if not any(a for a in args_raw if a not in ("--clean",)):
            sys.exit(0)

    if params["from"] and params["to"]:
        start_date = params["from"]
        end_date   = params["to"]
    else:
        start_date = date.today()
        end_date   = start_date + timedelta(days=params["days"] - 1)

    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)

    print(f"\n{'='*65}")
    print(f"  COLLECT ODDS — {start_date} → {end_date}  ({len(dates)} giorni)")
    print(f"  Bookmaker: Bet365 filtrato server-side (bookmaker=8)")
    print(f"  Stima chiamate: {len(dates)} × ~10 pagine = ~{len(dates)*10} call")
    print(f"  Raw dir: {RAW_DIR}")
    print(f"{'='*65}\n")

    total_fixtures = 0
    total_calls    = 0

    for d in dates:
        fix, calls    = collect_date(d, force=params["force"])
        total_fixtures += fix
        total_calls    += calls

    print(f"\n{'='*65}")
    print(f"  Fixture con odds: {total_fixtures}")
    print(f"  Chiamate API:     {total_calls}  (budget rimanente: ~{7500-total_calls}/giorno)")
    print(f"  Prossimo: python parse_odds.py --all")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()