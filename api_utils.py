"""
api_utils.py — Centralizza le chiamate HTTP verso API-Football.
Gestisce retry, rate limit, e header autenticazione.
"""

import requests
import time
import os
from pathlib import Path

_KEY_FILE = Path(__file__).resolve().parent.parent / ".api_key"

def _get_api_key():
    key = os.environ.get("APIFOOTBALL_KEY", "")
    if not key and _KEY_FILE.exists():
        key = _KEY_FILE.read_text().strip()
    return key

BASE_URL = "https://v3.football.api-sports.io"
MAX_RETRY = 3
SLEEP_SEC = 0.35


def make_api_request(url, params=None):
    """
    Esegue una GET verso API-Football con retry e gestione rate limit.
    Ritorna il dict JSON o None in caso di errore definitivo.
    """
    api_key = _get_api_key()
    headers = {
        "x-rapidapi-key":  api_key,
        "x-rapidapi-host": "v3.football.api-sports.io",
    }

    for attempt in range(MAX_RETRY):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            if r.status_code == 429:
                print(f"    [RATELIMIT] sleep 60s")
                time.sleep(60)
                continue
            if r.status_code != 200:
                print(f"    [HTTP {r.status_code}] retry {attempt+1}/{MAX_RETRY}")
                time.sleep(2 ** attempt)
                continue
            return r.json()
        except requests.RequestException as e:
            print(f"    [NETWORK] {e} retry {attempt+1}/{MAX_RETRY}")
            time.sleep(2 ** attempt)

    return None
