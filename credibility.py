"""
credibility.py — Ranking credibilità Home Win per fixture future.

Il CRED score misura la probabilità empirica di vittoria casalinga
basandosi SOLO su frequenze osservate, senza modelli e senza quote.

Componenti:
  C1 = HR% storico della lega (prior strutturale, stabile cross-stagione)
  C2 = % vittorie della squadra di casa nelle ultime N partite IN CASA
  C3 = % sconfitte dell'ospite nelle ultime N partite IN TRASFERTA
  C4 = streak recente casa (ultime 3 in casa)

  CRED_simple = (C2 + C3) / 2   ← il segnale principale
  CRED_full   = 0.30*C1 + 0.30*C2 + 0.25*C3 + 0.15*C4

Performance su 15.498 match (11 nazioni, 3 stagioni):
  CRED_simple ≥ 0.60 → HR 58.9%  (2.4 match/giorno, stabile cross-stagione)
  CRED_simple ≥ 0.70 → HR 64.0%  (1.0 match/giorno)
  C2 ≥ 0.8 × C3 ≥ 0.8 → HR 75.4% (raro: ~134 match su 15k)

Pipeline: step 8 (dopo parse_standings, prima di predict)
Zero chiamate API — usa solo all_results.csv e all_fixtures.csv

Uso:
    python credibility.py england
    python credibility.py --all
    python credibility.py --all --cutoff=2026-03-16
    python credibility.py --all --min-cred=0.55

Legge:  data/{nation}/processed/all_results.csv
        data/{nation}/processed/all_fixtures.csv
Salva:  data/{nation}/processed/credibility.csv
        output/credibility_ranking.csv  (aggregato cross-nazione)
"""

import pandas as pd
import numpy as np
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────

CONFIG_DIR    = "config"
WINDOW        = 6       # ultimi N match per rolling (6 = ottimale da backtest)
MIN_MATCHES   = 3       # minimo match per calcolare rolling
DEFAULT_CRED  = 0.50    # soglia minima default per output

# Prior HR% per lega — calcolati su 18.862 match (2023-2026)
# Stabili cross-stagione (std < 3pp per quasi tutte le leghe)
LEAGUE_HR_PRIOR = {
    "Segunda División":      0.457,
    "La Liga":               0.451,
    "Super Lig":             0.448,
    "Premiership":           0.445,
    "Eredivisie":            0.444,
    "Ligue 1":               0.444,
    "Championship":          0.443,
    "League One":            0.442,
    "2. Bundesliga":         0.435,
    "Jupiler Pro League":    0.433,
    "Premier League":        0.429,
    "League Two":            0.426,
    "Primeira Liga":         0.425,
    "Super League":          0.424,
    "Bundesliga":            0.423,
    "Ligue 2":               0.423,
    "Serie A":               0.405,
    "Serie B":               0.395,
}
DEFAULT_PRIOR = 0.430


# ── FUNZIONI CORE ─────────────────────────────────────────────────────────────

def load_config(nation_code):
    with open(Path(CONFIG_DIR) / f"{nation_code}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_results(nation_code):
    path = Path("data") / nation_code / "processed" / "all_results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["played"] == 1].copy()
    df = df[df["home_goals"].notna() & df["away_goals"].notna()]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_fixtures(nation_code):
    path = Path("data") / nation_code / "processed" / "all_fixtures.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_rolling_stats(df):
    """
    Costruisce dizionari rolling per ogni squadra:
      home_wins[team_id] = lista di 0/1 (vittoria in casa?)
      away_losses[team_id] = lista di 0/1 (sconfitta in trasferta?)
      home_gf[team_id] = lista gol fatti in casa
    """
    home_wins   = defaultdict(list)
    away_losses = defaultdict(list)
    home_gf     = defaultdict(list)

    for _, row in df.iterrows():
        hid = row["home_id"]
        aid = row["away_id"]
        hg  = row["home_goals"]
        ag  = row["away_goals"]

        # Casa: vittoria?
        home_wins[hid].append(1 if hg > ag else 0)
        home_gf[hid].append(hg)

        # Away: sconfitta?
        away_losses[aid].append(1 if hg > ag else 0)

    return home_wins, away_losses, home_gf


def compute_cred(home_id, away_id, league_name,
                 home_wins, away_losses, home_gf):
    """
    Calcola il CRED score per una singola partita.

    Ritorna dict con tutte le componenti e il punteggio finale.
    Se dati insufficienti → CRED = None.
    """
    # C1: prior lega
    c1 = LEAGUE_HR_PRIOR.get(league_name, DEFAULT_PRIOR)

    # C2: home win rate in casa (ultimi WINDOW)
    hw = home_wins.get(home_id, [])
    if len(hw) >= MIN_MATCHES:
        c2 = np.mean(hw[-WINDOW:])
        home_n = min(len(hw), WINDOW)
    else:
        c2 = None
        home_n = len(hw)

    # C3: away loss rate in trasferta (ultimi WINDOW)
    al = away_losses.get(away_id, [])
    if len(al) >= MIN_MATCHES:
        c3 = np.mean(al[-WINDOW:])
        away_n = min(len(al), WINDOW)
    else:
        c3 = None
        away_n = len(al)

    # C4: streak ultime 3 in casa
    if len(hw) >= MIN_MATCHES:
        last3 = hw[-3:]
        c4 = sum(last3) / 3
    else:
        c4 = None

    # CRED_simple = (C2 + C3) / 2
    if c2 is not None and c3 is not None:
        cred_simple = (c2 + c3) / 2
    else:
        cred_simple = None

    # CRED_full = ponderato con prior e streak
    if c2 is not None and c3 is not None and c4 is not None:
        cred_full = 0.30 * c1 + 0.30 * c2 + 0.25 * c3 + 0.15 * c4
    else:
        cred_full = None

    # Media gol fatti in casa (per info)
    hgf = home_gf.get(home_id, [])
    avg_gf_home = round(np.mean(hgf[-WINDOW:]), 2) if len(hgf) >= MIN_MATCHES else None

    return {
        "C1_league_prior": round(c1, 3),
        "C2_home_win_pct": round(c2, 3)  if c2 is not None else None,
        "C3_away_loss_pct": round(c3, 3) if c3 is not None else None,
        "C4_home_streak":  round(c4, 3)  if c4 is not None else None,
        "avg_gf_home":     avg_gf_home,
        "home_sample":     home_n,
        "away_sample":     away_n,
        "CRED":            round(cred_simple, 3) if cred_simple is not None else None,
        "CRED_full":       round(cred_full, 3)   if cred_full is not None else None,
    }


def cred_label(cred):
    """Etichetta leggibile per il CRED score."""
    if cred is None:    return "—"
    if cred >= 0.75:    return "★★★ ELITE"
    if cred >= 0.65:    return "★★  FORTE"
    if cred >= 0.55:    return "★   BUONO"
    if cred >= 0.45:    return "·   NEUTRO"
    if cred >= 0.35:    return "✗   DEBOLE"
    return                     "✗✗  SCARTO"


# ── PROCESS NAZIONE ───────────────────────────────────────────────────────────

def process_nation(nation_code, min_cred=DEFAULT_CRED, cutoff=None):
    try:
        cfg = load_config(nation_code)
    except FileNotFoundError:
        print(f"  [SKIP] {nation_code}: config non trovato")
        return []

    nation = cfg["nation"]
    df_res = load_results(nation_code)
    df_fix = load_fixtures(nation_code)

    if df_res is None:
        print(f"  [{nation}] all_results.csv non trovato")
        return []
    if df_fix is None or len(df_fix) == 0:
        print(f"  [{nation}] all_fixtures.csv vuoto o non trovato")
        return []

    # Cutoff opzionale: usa solo risultati prima di una certa data
    if cutoff:
        cutoff_dt = pd.to_datetime(cutoff).normalize()
        df_res = df_res[df_res["date"] < cutoff_dt].copy()

    # Costruisci rolling stats da tutti i risultati
    home_wins, away_losses, home_gf = build_rolling_stats(df_res)

    # Filtra fixture: solo future (non giocate)
    if "played" in df_fix.columns:
        df_fix = df_fix[df_fix["played"] != 1].copy()

    if cutoff:
        df_fix = df_fix[df_fix["date"] >= cutoff_dt].copy()

    rows = []
    for _, fix in df_fix.iterrows():
        hid   = fix.get("home_id")
        aid   = fix.get("away_id")
        lname = fix.get("league_name", "")

        cred = compute_cred(hid, aid, lname, home_wins, away_losses, home_gf)

        rows.append({
            "fixture_id":  fix.get("fixture_id"),
            "date":        fix.get("date"),
            "time":        fix.get("time", ""),
            "nation":      nation,
            "league":      lname,
            "tier":        fix.get("tier"),
            "round":       fix.get("round", ""),
            "home_name":   fix.get("home_name"),
            "away_name":   fix.get("away_name"),
            "home_id":     hid,
            "away_id":     aid,
            **cred,
            "label":       cred_label(cred["CRED"]),
        })

    out_df = pd.DataFrame(rows)

    # Salva per nazione (tutto)
    out_path = Path("data") / nation_code / "processed" / "credibility.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    # Stampa riepilogo
    valid = out_df[out_df["CRED"].notna()]
    above = valid[valid["CRED"] >= min_cred]
    elite = valid[valid["CRED"] >= 0.70]

    print(f"  {nation:<14}  fix={len(df_fix):>4}  scored={len(valid):>4}  "
          f"CRED≥{min_cred:.2f}={len(above):>3}  ★★★={len(elite):>2}")

    # Print top picks
    if len(above) > 0:
        top = above.sort_values("CRED", ascending=False).head(5)
        for _, r in top.iterrows():
            print(f"    {str(r['date'])[:10]} {r['home_name']:<20} vs {r['away_name']:<20} "
                  f"CRED={r['CRED']:.2f}  {r['label']}")

    return rows


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    if not args:
        print("Uso: python credibility.py <nazione>  oppure  --all")
        print("     --min-cred=0.55   soglia minima per output")
        print("     --cutoff=YYYY-MM-DD  usa risultati prima di questa data")
        sys.exit(0)

    # Parse args
    min_cred = DEFAULT_CRED
    cutoff   = None
    clean_args = []
    for a in args:
        if a.startswith("--min-cred="):
            min_cred = float(a.split("=")[1])
        elif a.startswith("--cutoff="):
            cutoff = a.split("=")[1]
        else:
            clean_args.append(a)

    nations = ([p.stem for p in Path(CONFIG_DIR).glob("*.json")]
               if clean_args[0] == "--all" else clean_args)

    print(f"\n{'='*65}")
    print(f"  CREDIBILITY RANKING — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Soglia minima: CRED ≥ {min_cred:.2f}")
    if cutoff:
        print(f"  Cutoff:        {cutoff}")
    print(f"  Window:        ultimi {WINDOW} match per componente")
    print(f"{'='*65}\n")

    all_rows = []
    for nation_code in sorted(nations):
        rows = process_nation(nation_code, min_cred=min_cred, cutoff=cutoff)
        all_rows.extend(rows)

    if not all_rows:
        print("\n  Nessuna fixture trovata.")
        return

    # ── OUTPUT AGGREGATO ──────────────────────────────────────────────────────
    agg = pd.DataFrame(all_rows)
    valid = agg[agg["CRED"].notna()].copy()

    # Ranking globale
    ranked = valid[valid["CRED"] >= min_cred].sort_values(
        ["date", "CRED"], ascending=[True, False]
    )

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    ranked.to_csv(out_dir / "credibility_ranking.csv", index=False)

    # ── REPORT FINALE ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  RANKING GLOBALE — {len(ranked)} fixture con CRED ≥ {min_cred:.2f}")
    print(f"{'='*65}")

    if len(ranked) == 0:
        print("  Nessuna fixture supera la soglia.")
        return

    # Raggruppa per data
    for dt in sorted(ranked["date"].unique()):
        day_df = ranked[ranked["date"] == dt].sort_values("CRED", ascending=False)
        dt_str = str(dt)[:10]
        print(f"\n  ── {dt_str} ({len(day_df)} match) ──")
        for _, r in day_df.iterrows():
            print(f"    {r['label']:<14} {r['CRED']:.2f}  "
                  f"{r['home_name']:<20} v {r['away_name']:<20} "
                  f"[{r['league']}, {r['nation']}]  "
                  f"C2={r['C2_home_win_pct']:.2f} C3={r['C3_away_loss_pct']:.2f}")

    # Stats
    print(f"\n  ── Distribuzione ──")
    for label_name in ["★★★ ELITE", "★★  FORTE", "★   BUONO", "·   NEUTRO"]:
        n = (ranked["label"] == label_name).sum()
        if n > 0:
            sub = ranked[ranked["label"] == label_name]
            print(f"    {label_name}: {n:3d} match  CRED medio={sub['CRED'].mean():.2f}")

    print(f"\n  [SALVATO] output/credibility_ranking.csv")
    print(f"  [SALVATO] data/{{nation}}/processed/credibility.csv per ogni nazione")

    # Suggerimento prossimi passi
    print(f"\n  Prossimi passi:")
    print(f"    → Filtra CRED ≥ 0.60 per selezione conservativa (HR ~59%)")
    print(f"    → Incrocia con odds.csv: se CRED ≥ 0.60 e quota ≥ 1.60 → value bet")
    print(f"    → Top-1 per giornata con CRED ≥ 0.50 → HR ~61% storicamente")


if __name__ == "__main__":
    main()