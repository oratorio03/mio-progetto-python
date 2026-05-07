"""
team_stats.py — Profili statistici squadre + forma recente pesata.

Uso:
    python team_stats.py england
    python team_stats.py --all

Novità rispetto alla versione precedente:
  - Salva gli ultimi 6 match per squadra (gol_fatti, gol_subiti, venue, data)
  - Calcola moltiplicatori forma F_att e F_def con pesi decrescenti
    pesi = [0.30, 0.22, 0.17, 0.13, 0.10, 0.08]  (più recente = peso maggiore)
  - F_att = media_ponderata_gf / media_storica_gf
  - F_def = media_ponderata_gs / media_storica_gs
  Questi moltiplicatori vengono usati da predict.py per correggere i lambda.
"""

import pandas as pd
import numpy as np
import sys
import json
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

CONFIG_DIR    = "config"
SEASONS_LONG  = 5
SEASONS_SHORT = 2
FORMA_N       = 6   # ultimi N match per forma recente
FORMA_WEIGHTS = [0.30, 0.22, 0.17, 0.13, 0.10, 0.08]  # somma = 1.0

def load_config(nation_code):
    path = Path(CONFIG_DIR) / f"{nation_code}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_paths(nation_code):
    proc = Path("data") / nation_code / "processed"
    return proc / "all_results.csv", proc / "team_stats.csv"

def load_results(results_path):
    df = pd.read_csv(results_path)
    df = df[df["played"] == 1].copy()
    df = df[df["home_goals"].notna() & df["away_goals"].notna()]
    for col in ["home_goals","away_goals","home_ht","away_ht","home_2h","away_2h"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df

def build_team_rows(df):
    """Esplode ogni partita in due righe: una per la squadra di casa, una per la trasferta."""
    rows = []
    for _, r in df.iterrows():
        ht_ok  = pd.notna(r["home_ht"]) and pd.notna(r["away_ht"])
        tot    = r["home_goals"] + r["away_goals"]
        tot_1t = tot_2t = t2_more = None

        if ht_ok:
            tot_1t  = r["home_ht"] + r["away_ht"]
            tot_2t  = r["home_2h"] + r["away_2h"]
            t2_more = 1 if tot_2t > tot_1t else 0

        base = {
            "fixture_id":  r["fixture_id"],
            "date":        r["date"],
            "season":      r["season"],
            "league_name": r["league_name"],
            "tier":        r["tier"],
            "total_gol":   tot,
            "over15":      1 if tot > 1 else 0,
            "over25":      1 if tot > 2 else 0,
            "over35":      1 if tot > 3 else 0,
            "tot_1t":      tot_1t,
            "tot_2t":      tot_2t,
            "t2_more":     t2_more,
        }

        btts = 1 if r["home_goals"] > 0 and r["away_goals"] > 0 else 0

        rows.append({**base,
            "team_id":     r["home_id"],
            "team_name":   r["home_name"],
            "venue":       "H",
            "gol_fatti":   r["home_goals"],
            "gol_subiti":  r["away_goals"],
            "win":         1 if r["home_goals"] > r["away_goals"] else 0,
            "draw":        1 if r["home_goals"] == r["away_goals"] else 0,
            "loss":        1 if r["home_goals"] < r["away_goals"] else 0,
            "scored":      1 if r["home_goals"] > 0 else 0,
            "clean_sheet": 1 if r["away_goals"] == 0 else 0,
            "btts":        btts,
        })

        rows.append({**base,
            "team_id":     r["away_id"],
            "team_name":   r["away_name"],
            "venue":       "A",
            "gol_fatti":   r["away_goals"],
            "gol_subiti":  r["home_goals"],
            "win":         1 if r["away_goals"] > r["home_goals"] else 0,
            "draw":        1 if r["away_goals"] == r["home_goals"] else 0,
            "loss":        1 if r["away_goals"] < r["home_goals"] else 0,
            "scored":      1 if r["away_goals"] > 0 else 0,
            "clean_sheet": 1 if r["home_goals"] == 0 else 0,
            "btts":        btts,
        })

    return pd.DataFrame(rows)

def compute_profile(tm, label):
    """Calcola il profilo aggregato per una finestra temporale."""
    n = len(tm)
    if n == 0:
        return {}

    h    = tm[tm["venue"] == "H"]
    a    = tm[tm["venue"] == "A"]
    tm_ht = tm[tm["t2_more"].notna()]
    n_ht  = len(tm_ht)

    def pct(col, sub=None):
        s = sub if sub is not None else tm
        return round(s[col].mean() * 100, 1) if len(s) > 0 else None

    def avg(col, sub=None):
        s = sub if sub is not None else tm
        return round(s[col].mean(), 3) if len(s) > 0 else None

    return {
        f"{label}_n":            n,
        f"{label}_win%":         pct("win"),
        f"{label}_draw%":        pct("draw"),
        f"{label}_loss%":        pct("loss"),
        f"{label}_gol_fatti":    avg("gol_fatti"),
        f"{label}_gol_subiti":   avg("gol_subiti"),
        f"{label}_scored%":      pct("scored"),
        f"{label}_cs%":          pct("clean_sheet"),
        f"{label}_btts%":        pct("btts"),
        f"{label}_over15%":      pct("over15"),
        f"{label}_over25%":      pct("over25"),
        f"{label}_over35%":      pct("over35"),
        f"{label}_H_win%":       pct("win",  h),
        f"{label}_H_gol_fatti":  avg("gol_fatti",  h),
        f"{label}_H_gol_subiti": avg("gol_subiti", h),
        f"{label}_H_btts%":      pct("btts",   h),
        f"{label}_H_over25%":    pct("over25", h),
        f"{label}_A_win%":       pct("win",  a),
        f"{label}_A_gol_fatti":  avg("gol_fatti",  a),
        f"{label}_A_gol_subiti": avg("gol_subiti", a),
        f"{label}_A_btts%":      pct("btts",   a),
        f"{label}_A_over25%":    pct("over25", a),
        f"{label}_t2more%":      round(tm_ht["t2_more"].mean()*100, 1) if n_ht > 5 else None,
        f"{label}_avg_1t":       round(tm_ht["tot_1t"].mean(), 3)      if n_ht > 5 else None,
        f"{label}_avg_2t":       round(tm_ht["tot_2t"].mean(), 3)      if n_ht > 5 else None,
    }

def compute_forma(tm, profile):
    """
    Calcola moltiplicatori forma sugli ultimi FORMA_N match.

    Logica:
      - Prende gli ultimi FORMA_N match ordinati dal più recente
      - Applica pesi decrescenti FORMA_WEIGHTS
      - F_att = media_ponderata(gol_fatti) / media_storica(gol_fatti)
      - F_def = media_ponderata(gol_subiti) / media_storica(gol_subiti)
      - F_win = media_ponderata(win)  → tendenza vittoria recente

    Valori > 1.0 = forma migliore della media storica
    Valori < 1.0 = forma peggiore della media storica

    Se meno di 3 match recenti → None (non abbastanza dati per forma)
    """
    recent = tm.sort_values("date", ascending=False).head(FORMA_N)
    n      = len(recent)

    if n < 3:
        return {
            "F_n":       n,
            "F_att":     None,
            "F_def":     None,
            "F_win":     None,
            "F_gf_avg":  None,
            "F_gs_avg":  None,
        }

    # Pesi: prende solo i pesi per i match disponibili e li rinormalizza
    weights = FORMA_WEIGHTS[:n]
    w_sum   = sum(weights)
    weights = [w / w_sum for w in weights]

    gf_vals  = recent["gol_fatti"].values
    gs_vals  = recent["gol_subiti"].values
    win_vals = recent["win"].values

    wgf  = sum(w * gf for w, gf in zip(weights, gf_vals))
    wgs  = sum(w * gs for w, gs in zip(weights, gs_vals))
    wwin = sum(w * wn for w, wn in zip(weights, win_vals))

    # Riferimento storico: usa C se disponibile, altrimenti L2, altrimenti L5
    ref_gf = (profile.get("C_gol_fatti") or
              profile.get("L2_gol_fatti") or
              profile.get("L5_gol_fatti"))
    ref_gs = (profile.get("C_gol_subiti") or
              profile.get("L2_gol_subiti") or
              profile.get("L5_gol_subiti"))

    # Moltiplicatori — cappati tra 0.5 e 2.0 per evitare esplosioni
    def safe_mult(num, den):
        if den is None or den == 0 or np.isnan(den):
            return 1.0
        m = num / den
        return round(max(0.5, min(2.0, m)), 3)

    f_att = safe_mult(wgf, ref_gf)
    f_def = safe_mult(wgs, ref_gs)

    # Salva anche i singoli match (gf, gs, venue) per trasparenza nel foglio Forma
    slot_data = {}
    for i, (_, row) in enumerate(recent.iterrows(), start=1):
        slot_data[f"F_gf{i}"]  = row["gol_fatti"]
        slot_data[f"F_gs{i}"]  = row["gol_subiti"]
        slot_data[f"F_v{i}"]   = row["venue"]
        slot_data[f"F_win{i}"] = row["win"]

    return {
        "F_n":      n,
        "F_att":    f_att,
        "F_def":    f_def,
        "F_win":    round(wwin, 3),
        "F_gf_avg": round(wgf,  3),
        "F_gs_avg": round(wgs,  3),
        **slot_data,
    }

def process_nation(nation_code, cutoff=None):
    cfg    = load_config(nation_code)
    nation = cfg["nation"]
    results_path, out_path = get_paths(nation_code)

    if not results_path.exists():
        print(f"[SKIP] {nation}: all_results.csv non trovato")
        return

    print(f"\n{'='*60}")
    print(f"  Team stats: {nation}" + (f"  [cutoff: {cutoff}]" if cutoff else ""))
    print(f"{'='*60}")

    df = load_results(results_path)

    # Applica cutoff: esclude partite >= cutoff date
    if cutoff:
        cutoff_dt = pd.to_datetime(cutoff).normalize()
        before = len(df)
        df = df[df["date"] < cutoff_dt].copy()
        print(f"Partite caricate: {len(df):,}  (escluse {before - len(df):,} dopo {cutoff})")
    else:
        print(f"Partite caricate: {len(df):,}")

    tm_df      = build_team_rows(df)
    max_season = tm_df["season"].max()
    cutoff_l   = max_season - SEASONS_LONG
    cutoff_s   = max_season - SEASONS_SHORT

    # Anagrafica squadre: prende la lega più recente per ogni team_id
    teams = (tm_df.sort_values("season", ascending=False)
                  .drop_duplicates("team_id")
                  [["team_id","team_name","league_name","tier"]])

    print(f"Squadre uniche: {len(teams):,}")

    all_profiles = []
    for _, t in teams.iterrows():
        tid    = t["team_id"]
        tm_all = tm_df[tm_df["team_id"] == tid]

        profile = {
            "team_id":       tid,
            "team_name":     t["team_name"],
            "nation":        nation,
            "last_league":   t["league_name"],
            "tier":          t["tier"],
            "last_season":   tm_all["season"].max(),
            "total_matches": len(tm_all),
        }

        # Finestre storiche
        profile.update(compute_profile(tm_all[tm_all["season"] >  cutoff_l],  "L5"))
        profile.update(compute_profile(tm_all[tm_all["season"] >  cutoff_s],  "L2"))
        profile.update(compute_profile(tm_all[tm_all["season"] == max_season],"C"))

        # Forma recente — usa tutti i match disponibili, ordinati per data
        profile.update(compute_forma(tm_all, profile))

        all_profiles.append(profile)

    out_df = pd.DataFrame(all_profiles)
    # Se cutoff attivo salva in file separato per non sovrascrivere la produzione
    save_path = out_path.parent / f"team_stats_cutoff_{cutoff}.csv" if cutoff else out_path

    out_df.to_csv(save_path, index=False)
    print(f"[SALVATO] {save_path}  ({len(out_df)} squadre, {len(out_df.columns)} colonne)")

    # Mini-report forma
    f_att_vals = out_df["F_att"].dropna()
    f_def_vals = out_df["F_def"].dropna()
    if len(f_att_vals) > 0:
        print(f"\n  Forma recente — distribuzione moltiplicatori:")
        print(f"  F_att  media={f_att_vals.mean():.2f}  min={f_att_vals.min():.2f}  max={f_att_vals.max():.2f}")
        print(f"  F_def  media={f_def_vals.mean():.2f}  min={f_def_vals.min():.2f}  max={f_def_vals.max():.2f}")
        hot     = out_df[out_df["F_att"] > 1.3]["team_name"].tolist()[:5]
        cold    = out_df[out_df["F_att"] < 0.7]["team_name"].tolist()[:5]
        if hot:  print(f"  Top forma offensiva:  {', '.join(hot)}")
        if cold: print(f"  Crisi offensiva:      {', '.join(cold)}")

def main():
    args = sys.argv[1:]
    if not args:
        print("Uso: python team_stats.py <nazione> [--cutoff=YYYY-MM-DD]")
        sys.exit(0)

    cutoff = None
    clean_args = []
    for a in args:
        if a.startswith("--cutoff="):
            cutoff = a.split("=")[1]
        else:
            clean_args.append(a)
    args = clean_args

    nations = ([p.stem for p in Path(CONFIG_DIR).glob("*.json")]
               if args and args[0] == "--all" else args)
    for n in nations:
        process_nation(n, cutoff=cutoff)

if __name__ == "__main__":
    main()