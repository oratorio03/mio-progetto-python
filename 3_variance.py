"""
variance.py — Analisi varianza annuale per lega di una nazione.

Uso:
    python variance.py england
    python variance.py --all
"""

import pandas as pd
import sys
import json
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

CONFIG_DIR = "config"
THRESHOLDS = {"stable": 2.5, "moderate": 5.0}

def load_config(nation_code):
    with open(Path(CONFIG_DIR) / f"{nation_code}.json", "r", encoding="utf-8") as f:
        return json.load(f)

def load_data(nation_code):
    path = Path("data") / nation_code / "processed" / "all_results.csv"
    if not path.exists():
        print(f"[ERRORE] {path} non trovato")
        return None
    df = pd.read_csv(path)
    df = df[df["played"] == 1].copy()
    df = df[df["home_goals"].notna() & df["away_goals"].notna()]
    for col in ["home_goals","away_goals","home_ht","away_ht","home_2h","away_2h"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Calcola colonne derivate se non presenti — update.py le calcola ma
    # versioni vecchie del DB potrebbero non averle
    if "total_goals" not in df.columns:
        df["total_goals"] = df["home_goals"] + df["away_goals"]
    if "over25" not in df.columns:
        df["over25"] = (df["total_goals"] > 2).astype(float)
    if "over15" not in df.columns:
        df["over15"] = (df["total_goals"] > 1).astype(float)
    if "btts" not in df.columns:
        df["btts"] = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(float)
    if "result_1x2" not in df.columns:
        df["result_1x2"] = df.apply(
            lambda r: "H" if r["home_goals"] > r["away_goals"]
                      else ("A" if r["home_goals"] < r["away_goals"] else "D"), axis=1)
    return df

def stability_label(std):
    if std is None: return "N/A"
    if std < THRESHOLDS["stable"]:   return "✓ stabile"
    if std < THRESHOLDS["moderate"]: return "~ moderata"
    return "✗ volatile"

def season_stats(df_s):
    n = len(df_s)
    if n < 30: return None
    hw = (df_s["result_1x2"] == "H").sum()
    dr = (df_s["result_1x2"] == "D").sum()
    df_ht = df_s[df_s["home_ht"].notna() & df_s["away_ht"].notna()].copy()
    for col in ["home_ht","away_ht","home_2h","away_2h"]:
        df_ht[col] = pd.to_numeric(df_ht[col], errors="coerce")
    df_ht = df_ht.dropna(subset=["home_ht","away_ht","home_2h","away_2h"])
    df_ht["tot_1t"] = df_ht["home_ht"] + df_ht["away_ht"]
    df_ht["tot_2t"] = df_ht["home_2h"] + df_ht["away_2h"]
    n_ht = len(df_ht)
    return {
        "n":       n,
        "H%":      round(hw/n*100, 1),
        "D%":      round(dr/n*100, 1),
        "A%":      round((n-hw-dr)/n*100, 1),
        "avg_gol": round(df_s["total_goals"].mean(), 2),
        "O2.5%":   round(df_s["over25"].mean()*100, 1),
        "BTTS%":   round(df_s["btts"].mean()*100, 1),
        "O1.5%":   round(df_s["over15"].mean()*100, 1),
        "2T_più%": round((df_ht["tot_2t"]>df_ht["tot_1t"]).mean()*100, 1) if n_ht > 20 else None,
    }

def analyze_variance(df, league_name):
    df_l    = df[df["league_name"] == league_name]
    seasons = sorted(df_l["season"].unique())
    rows    = []
    for s in seasons:
        stats = season_stats(df_l[df_l["season"] == s])
        if stats:
            stats["season"] = s
            rows.append(stats)
    if len(rows) < 3:
        return None, None
    df_var  = pd.DataFrame(rows).set_index("season")
    metrics = ["H%","D%","A%","avg_gol","O2.5%","BTTS%","O1.5%","2T_più%"]
    summary = {}
    for m in metrics:
        if m not in df_var.columns:
            continue
        col = df_var[m].dropna()
        if len(col) >= 3:
            summary[m] = {
                "mean": round(col.mean(), 2),
                "std":  round(col.std(),  2),
                "min":  round(col.min(),  1),
                "max":  round(col.max(),  1),
            }
    return df_var, summary

def print_league_detail(league_name, df_var, summary):
    metrics = [m for m in ["H%","D%","A%","avg_gol","O2.5%","BTTS%","O1.5%","2T_più%"]
               if m in df_var.columns]
    print(f"\n{'═'*80}")
    print(f"  {league_name}")
    print(f"{'═'*80}")
    print(f"{'Season':>8}" + "".join(f"{m:>10}" for m in metrics))
    print("-" * 80)
    for season, row in df_var.iterrows():
        print(f"{season:>8}" + "".join(f"{str(row.get(m,'N/A')):>10}" for m in metrics))
    print(f"\n  {'':>8}" + "".join(f"{m:>10}" for m in metrics))
    for lbl, key in [("Media","mean"),("Std","std"),("Min","min"),("Max","max")]:
        print(f"  {lbl:>6}" + "".join(
            f"{str(summary.get(m,{}).get(key,'N/A')):>10}" for m in metrics))
    print(f"\n  Stabilità:")
    for m in metrics:
        std = summary.get(m,{}).get("std")
        print(f"    {m:<12} std={str(std):<6}  {stability_label(std)}")

def print_stability_matrix(all_summaries):
    metrics = ["H%","D%","A%","avg_gol","O2.5%","BTTS%","O1.5%","2T_più%"]
    print(f"\n\n{'═'*90}")
    print("  MATRICE STABILITÀ")
    print(f"{'═'*90}")
    print(f"{'Lega':<26}" + "".join(f"{m:>10}" for m in metrics))
    print("-" * 90)
    for league_name, summary in all_summaries.items():
        line = f"{league_name:<26}"
        for m in metrics:
            std = summary.get(m,{}).get("std")
            if std is None:
                line += f"{'N/A':>10}"
            else:
                sym = "✓" if std < THRESHOLDS["stable"] else ("~" if std < THRESHOLDS["moderate"] else "✗")
                line += f"{sym+' '+f'{std:.1f}':>10}"
        print(line)
    print(f"\n  AFFIDABILITÀ COME FEATURE:")
    for m in metrics:
        stds = [s.get(m,{}).get("std") for s in all_summaries.values() if s.get(m)]
        stds = [s for s in stds if s is not None]
        if not stds: continue
        avg_std  = sum(stds)/len(stds)
        n_stable = sum(1 for s in stds if s < THRESHOLDS["stable"])
        print(f"    {m:<12} std medio={avg_std:.2f}  leghe stabili={n_stable}/{len(stds)}  → {stability_label(avg_std)}")

def process_nation(nation_code):
    try:
        cfg = load_config(nation_code)
    except FileNotFoundError:
        print(f"  [SKIP] {nation_code}: config non trovato")
        return

    print(f"\n{'='*60}")
    print(f"  VARIANCE — {cfg['nation']}")
    print(f"{'='*60}")

    df = load_data(nation_code)
    if df is None:
        return

    leagues_order = [v["name"] for v in cfg["leagues"].values()]
    all_summaries = {}

    for league_name in leagues_order:
        df_var, summary = analyze_variance(df, league_name)
        if df_var is not None:
            print_league_detail(league_name, df_var, summary)
            all_summaries[league_name] = summary

    if not all_summaries:
        print(f"  [SKIP] {cfg['nation']}: dati insufficienti")
        return

    print_stability_matrix(all_summaries)

    rows_out = []
    for league_name, summary in all_summaries.items():
        row = {"lega": league_name}
        for m, stats in summary.items():
            row[f"{m}_mean"] = stats["mean"]
            row[f"{m}_std"]  = stats["std"]
            row[f"{m}_min"]  = stats["min"]
            row[f"{m}_max"]  = stats["max"]
        rows_out.append(row)

    out_path = Path("data") / nation_code / "processed" / "variance_stats.csv"
    pd.DataFrame(rows_out).to_csv(out_path, index=False)
    print(f"\n[SALVATO] {out_path}")

def main():
    args = sys.argv[1:]
    if not args:
        print("Uso: python variance.py <nazione>  oppure  --all")
        sys.exit(0)
    nations = ([p.stem for p in Path(CONFIG_DIR).glob("*.json")]
               if args[0] == "--all" else args)
    for nation_code in nations:
        process_nation(nation_code)

if __name__ == "__main__":
    main()