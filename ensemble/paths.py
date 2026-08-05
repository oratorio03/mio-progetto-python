"""
paths.py — Risoluzione percorsi progetto.

Gli script della pipeline vivono in 0_CORE_PIPELINE/ mentre data/ e config/
stanno nella root, ma in sviluppo capita di avere tutto piatto nella stessa
cartella. Qui cerchiamo la root risalendo finché non troviamo data/ o config/.
"""

from pathlib import Path
import json
import os

_SCRIPT_DIR = Path(__file__).resolve().parent


def find_root(start=None):
    """Prima directory risalendo da `start` che contiene data/ o config/."""
    env = os.environ.get("BETPRO_ROOT")
    if env:
        return Path(env).resolve()

    candidates = []
    base = Path(start).resolve() if start else _SCRIPT_DIR.parent
    candidates.append(base)
    candidates.extend(base.parents)
    candidates.append(Path.cwd())
    candidates.extend(Path.cwd().parents)

    for cand in candidates:
        if (cand / "data").is_dir() or (cand / "config").is_dir():
            return cand
    return base


ROOT = find_root()


def config_dir():
    return ROOT / "config"


def data_dir(nation_code):
    return ROOT / "data" / nation_code / "processed"


def results_path(nation_code):
    return data_dir(nation_code) / "all_results.csv"


def fixtures_path(nation_code):
    return data_dir(nation_code) / "all_fixtures.csv"


def odds_path(nation_code):
    return data_dir(nation_code) / "odds.csv"


def models_dir():
    d = ROOT / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def output_dir():
    d = ROOT / "output"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_nations():
    """Nazioni disponibili: config/*.json, con fallback su data/*/processed."""
    cfg = config_dir()
    if cfg.is_dir():
        nations = sorted(p.stem for p in cfg.glob("*.json"))
        if nations:
            return nations
    root_data = ROOT / "data"
    if root_data.is_dir():
        return sorted(p.name for p in root_data.iterdir()
                      if (p / "processed" / "all_results.csv").exists())
    return []


def load_config(nation_code):
    path = config_dir() / f"{nation_code}.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
