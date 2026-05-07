"""
backtest.py — Walk-forward backtest definitivo.

Logica walk-forward:
  Per ogni settimana nel range:
    1. Ricalcola profili squadre con cutoff (zero data-leak)
    2. Genera previsioni come se fossero LIVE a quella data
    3. Confronta con risultati reali

Mercati valutati:
    HOME WIN   — casa vince
    OVER 2.5   — gol totali > 2.5
    BTTS       — entrambe le squadre segnano
    PARACADUTE — DC (1X) AND Over 1.5 entrambi veri

Regola best-signal:
    Se una partita ha N segnali attivi, solo il segnale con quota piu alta
    viene marcato is_best=True. Gli altri sono dimmed (is_best=False).
    Le metriche vengono calcolate sia su tutti i segnali che solo sui best.

Uso:
    python backtest.py --start=2025-10-01 --end=2026-03-09
    python backtest.py --start=2025-10-01 --end=2026-03-09 --nations=england,italy
    python backtest.py --start=2025-10-01 --end=2026-03-09 --step=7 --verbose

Output:
    output/backtest_results_<start>_<end>.csv   -- una riga per segnale
    output/backtest_summary_<start>_<end>.csv   -- riepilogo settimanale
"""

import sys
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, datetime, timedelta

warnings.filterwarnings("ignore")

# Ancora tutte le path alla directory del file — funziona da qualsiasi working dir
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT_DIR   = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_ROOT_DIR))

from team_stats import process_nation as ts_process_nation
from predict   import build_nation, load_config

CONFIG_DIR = _ROOT_DIR / "config"
OUTPUT_DIR = _ROOT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

MARKETS = ["HOME WIN", "OVER 2.5", "BTTS", "PARACADUTE"]

# -- UTILITA TEMPO ------------------------------------------------------------
def week_range(start: date, end: date, step: int = 7):
    current = start
    while current <= end:
        yield current, current + timedelta(days=step - 1)
        current += timedelta(days=step)

# -- CARICAMENTO RISULTATI ----------------------------------------------------
def load_results_nation(nation_code):
    path = _ROOT_DIR / "data" / nation_code / "processed" / "all_results.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["played"] == 1].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for col in ["home_goals", "away_goals"]:
        df[col] = pd.to_numeric(df.get(col, np.nan), errors="coerce")
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["result_1x2"]  = df.get("result_1x2", pd.Series(dtype=str))
    df["over25"]      = (df["total_goals"] > 2.5).astype(float)
    df["over15"]      = (df["total_goals"] > 1.5).astype(float)
    df["btts"]        = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(float)
    df["dc"]          = (df["result_1x2"].isin(["H", "D"])).astype(float)
    df["paracadute"]  = ((df["dc"] == 1.0) & (df["over15"] == 1.0)).astype(float)
    return df

def trova_risultato(fixture_id, home, away, d, results_df):
    if results_df.empty:
        return None
    if fixture_id and "fixture_id" in results_df.columns:
        m = results_df[results_df["fixture_id"] == fixture_id]
        if not m.empty:
            return m.iloc[0]
    mask = (
        (results_df["date"] == d) &
        (results_df["home_name"].str.strip().str.lower() == str(home).strip().lower()) &
        (results_df["away_name"].str.strip().str.lower() == str(away).strip().lower())
    )
    m = results_df[mask]
    return m.iloc[0] if not m.empty else None

# -- VALUTAZIONE MERCATO ------------------------------------------------------
def valuta_mercato(market, ris):
    if ris is None:
        return None
    market = market.replace("★", "").strip()
    if market == "HOME WIN":
        return ris.get("result_1x2") == "H"
    if market == "OVER 2.5":
        v = ris.get("over25")
        return bool(v == 1.0) if pd.notna(v) else None
    if market == "BTTS":
        v = ris.get("btts")
        return bool(v == 1.0) if pd.notna(v) else None
    if market == "PARACADUTE":
        # Vince se DC=True AND Over1.5=True
        v = ris.get("paracadute")
        return bool(v == 1.0) if pd.notna(v) else None
    return None

# -- ACCUMULATOR --------------------------------------------------------------
class Accumulator:
    def __init__(self):
        self.data = {m: {
            "seg": 0, "val": 0, "vinti": 0,
            "star_seg": 0, "star_val": 0, "star_vinti": 0,
            "best_seg": 0, "best_val": 0, "best_vinti": 0,
            "roi_num": 0.0, "roi_den": 0,
            "best_roi_num": 0.0, "best_roi_den": 0,
        } for m in MARKETS}

    def add(self, market, esito, is_star, is_best, quota=None):
        if market not in self.data:
            return
        d = self.data[market]
        d["seg"] += 1
        if is_star:  d["star_seg"] += 1
        if is_best:  d["best_seg"] += 1
        if esito is None:
            return
        d["val"] += 1
        if is_star:  d["star_val"] += 1
        if is_best:  d["best_val"] += 1
        if esito:
            d["vinti"] += 1
            if is_star:  d["star_vinti"] += 1
            if is_best:  d["best_vinti"] += 1
            if quota:
                d["roi_num"] += quota - 1;      d["roi_den"] += 1
                if is_best:
                    d["best_roi_num"] += quota - 1; d["best_roi_den"] += 1
        else:
            if quota:
                d["roi_num"] -= 1;              d["roi_den"] += 1
                if is_best:
                    d["best_roi_num"] -= 1;     d["best_roi_den"] += 1

    def hit(self, m):
        d = self.data[m]
        return round(d["vinti"] / d["val"] * 100, 1) if d["val"] > 0 else None

    def star_hit(self, m):
        d = self.data[m]
        return round(d["star_vinti"] / d["star_val"] * 100, 1) if d["star_val"] > 0 else None

    def best_hit(self, m):
        d = self.data[m]
        return round(d["best_vinti"] / d["best_val"] * 100, 1) if d["best_val"] > 0 else None

    def roi(self, m, best=False):
        d = self.data[m]
        if best:
            return round(d["best_roi_num"] / d["best_roi_den"] * 100, 1) if d["best_roi_den"] > 0 else None
        return round(d["roi_num"] / d["roi_den"] * 100, 1) if d["roi_den"] > 0 else None

# -- CORE SETTIMANA -----------------------------------------------------------
def run_week(nations, week_start, week_end, results_by_nation, verbose=False):
    cutoff_str = str(week_start)
    rows_out   = []

    for nation_code in nations:
        try:
            ts_process_nation(nation_code, cutoff=cutoff_str)
        except Exception as e:
            if verbose: print(f"  [{nation_code}] team_stats ERR: {e}")
            continue

        try:
            cfg     = load_config(nation_code)
            df_p, _ = build_nation(
                nation_code, cfg,
                start_date=week_start, end_date=week_end,
                backtest=True, cutoff=cutoff_str
            )
        except Exception as e:
            if verbose: print(f"  [{nation_code}] predict ERR: {e}")
            continue

        if df_p is None or df_p.empty:
            continue

        results_df = results_by_nation.get(nation_code, pd.DataFrame())

        for _, pred in df_p.iterrows():
            try:
                d   = pd.to_datetime(pred["Data"]).date()
                ris = trova_risultato(
                    pred.get("fixture_id"),
                    pred["Casa"], pred["Trasferta"], d, results_df
                )

                def get_quota(m):
                    try:
                        q = pred.get("Q1") if m == "HOME WIN" else pred.get("Q_Para") if m == "PARACADUTE" else None
                        return float(q) if q and not pd.isna(q) else None
                    except: return None

                # Segnali attivi
                attivi = {}
                for m in MARKETS:
                    sig = pred.get(m)
                    if sig and str(sig).strip() not in ("", "None", "nan"):
                        attivi[m] = get_quota(m)

                if not attivi:
                    continue

                # Best = quota piu alta; se pareggio o assenza quote, priority order
                def sort_q(m):
                    q = attivi[m]
                    return float(q) if q else 0.0

                best_market = max(attivi, key=sort_q)

                for m, quota in attivi.items():
                    sig     = str(pred.get(m, ""))
                    esito   = valuta_mercato(sig, ris)
                    rows_out.append({
                        "week":               str(week_start),
                        "data":               str(d),
                        "nazione":            pred.get("Nazione", ""),
                        "lega":               pred.get("Lega", ""),
                        "tier":               pred.get("Tier"),
                        "casa":               pred["Casa"],
                        "trasferta":          pred["Trasferta"],
                        "mercato":            m,
                        "is_star":            "★" in sig,
                        "is_best":            m == best_market,
                        "n_segnali_partita":  len(attivi),
                        "h_pct":              pred.get("H%"),
                        "d_pct":              pred.get("D%"),
                        "o25_pct":            pred.get("O2.5%"),
                        "o15_pct":            pred.get("O1.5%"),
                        "btts_pct":           pred.get("BTTS%"),
                        "dc_pct":             pred.get("DC%"),
                        "gol_attesi":         pred.get("Gol Attesi"),
                        "quota":              quota,
                        "q1":                 pred.get("Q1"),
                        "q_para":             pred.get("Q_Para"),
                        "confidenza":         pred.get("Confidenza"),
                        "b365_conf":          pred.get("B365_Conf"),
                        "hw_grigia":          bool(pred.get("HW_Grigia", False)),
                        "hw_arancione":       bool(pred.get("HW_Arancione", False)),
                        "giocata":            ris is not None,
                        "esito":              esito,
                        "ris_1x2":            ris.get("result_1x2") if ris is not None else None,
                        "ris_gol":            ris.get("total_goals") if ris is not None else None,
                    })
            except Exception:
                continue

    return rows_out

# -- STAMPA -------------------------------------------------------------------
def stampa_summary(acc, label="TOTALE"):
    print(f"\n{'='*76}")
    print(f"  {label}")
    print(f"{'='*76}")
    print(f"  {'Mercato':<14} {'Seg':>5} {'Hit%':>7} {'*Hit%':>7} "
          f"{'Best':>6} {'BestHit%':>9} {'BestROI%':>9}")
    print(f"  {'-'*70}")
    for m in MARKETS:
        d  = acc.data[m]
        h  = f"{acc.hit(m):.1f}%"    if acc.hit(m)      is not None else "    --"
        sh = f"{acc.star_hit(m):.1f}%" if acc.star_hit(m) is not None else "    --"
        bh = f"{acc.best_hit(m):.1f}%" if acc.best_hit(m) is not None else "    --"
        br = f"{acc.roi(m,best=True):.1f}%" if acc.roi(m,best=True) is not None else "    --"
        print(f"  {m:<14} {d['seg']:>5} {h:>7} {sh:>7} "
              f"{d['best_seg']:>6} {bh:>9} {br:>9}")

def stampa_breakdown_lega(df_res, mercato="HOME WIN", min_n=10):
    sub = df_res[(df_res["mercato"]==mercato) & df_res["giocata"] & df_res["is_best"]].copy()
    if sub.empty: return
    print(f"\n  Per LEGA -- {mercato} best-only (min {min_n}):")
    print(f"  {'Lega':<28} {'Naz':<10} T {'N':>5} {'Hit%':>7} {'ROI%':>7}")
    print(f"  {'-'*62}")
    rows = []
    for lega, g in sub.groupby("lega"):
        n   = g["esito"].notna().sum()
        if n < min_n: continue
        win = (g["esito"]==True).sum()
        naz = g["nazione"].iloc[0]
        tier= g["tier"].iloc[0]
        qg  = g[g["quota"].notna()]
        roi = round(((qg["esito"]*qg["quota"]).sum()-len(qg))/len(qg)*100,1) if len(qg)>=5 else None
        rows.append((lega, naz, tier, n, round(win/n*100,1), roi))
    for lega, naz, tier, n, hit, roi in sorted(rows, key=lambda x:-x[4]):
        mk  = " *" if hit >= 60 else ""
        rs  = f"{roi:.1f}%" if roi is not None else "   --"
        print(f"  {str(lega):<28} {str(naz):<10} {int(tier):>1} {n:>5} {hit:>6}% {rs:>7}{mk}")

def stampa_breakdown_quota(df_res, mercato="HOME WIN"):
    sub = df_res[(df_res["mercato"]==mercato) & df_res["giocata"] & df_res["quota"].notna()].copy()
    if sub.empty: return
    print(f"\n  Per FASCIA QUOTA -- {mercato}:")
    bins   = [1.0,1.50,1.70,1.90,2.10,2.50,99]
    labels = ["1.00-1.50","1.50-1.70","1.70-1.90","1.90-2.10","2.10-2.50",">2.50"]
    sub["qbin"] = pd.cut(sub["quota"], bins=bins, labels=labels)
    print(f"  {'Fascia':<12} {'N':>5} {'Hit%':>7} {'ROI%':>8}  best: {'N':>4} {'Hit%':>7}")
    print(f"  {'-'*55}")
    for lbl in labels:
        g = sub[sub["qbin"]==lbl]
        n = g["esito"].notna().sum()
        if n < 5: continue
        win = (g["esito"]==True).sum()
        hit = round(win/n*100,1)
        roi = round(((g["esito"]*g["quota"]).sum()-n)/n*100,1)
        bg  = g[g["is_best"]]
        bn  = bg["esito"].notna().sum()
        bw  = (bg["esito"]==True).sum()
        bh  = f"{round(bw/bn*100,1):.1f}%" if bn>0 else "   --"
        print(f"  {lbl:<12} {n:>5} {hit:>6}% {roi:>7}%   {bn:>5} {bh:>7}")

def stampa_n_segnali(df_res):
    hw = df_res[(df_res["mercato"]=="HOME WIN") & df_res["giocata"]].copy()
    if hw.empty: return
    print(f"\n  Hit rate HOME WIN per N segnali sulla stessa partita:")
    for n in sorted(hw["n_segnali_partita"].dropna().unique()):
        g   = hw[hw["n_segnali_partita"]==n]
        vl  = g["esito"].notna().sum()
        win = (g["esito"]==True).sum()
        if vl < 5: continue
        print(f"    N={int(n):>2}  {vl:>4} casi  {win:>3} vinti  {round(win/vl*100,1):>5}%")

# -- MAIN ---------------------------------------------------------------------
def main():
    args_raw = sys.argv[1:]
    if not args_raw or "--help" in args_raw:
        print(__doc__)
        sys.exit(0)

    params = {}
    for a in args_raw:
        if "=" in a:
            k, v = a.lstrip("-").split("=", 1)
            params[k] = v

    start_str = params.get("start")
    end_str   = params.get("end")
    step      = int(params.get("step", 7))
    verbose   = "--verbose" in args_raw

    if not start_str or not end_str:
        print("[ERRORE] --start e --end obbligatori")
        sys.exit(1)

    start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
    end_date   = datetime.strptime(end_str,   "%Y-%m-%d").date()

    nations = (
        [n.strip() for n in params["nations"].split(",")]
        if "nations" in params
        else [p.stem for p in Path(CONFIG_DIR).glob("*.json")]
    )

    print(f"\n{'='*76}")
    print(f"  WALK-FORWARD BACKTEST -- DEFINITIVO")
    print(f"  Periodo : {start_date} -> {end_date}  (step={step}gg)")
    print(f"  Nazioni : {', '.join(nations)}")
    print(f"  Mercati : {', '.join(MARKETS)}")
    print(f"  Note    : momentum disabilitato (standings storiche non disponibili)")
    print(f"{'='*76}")

    results_by_nation = {n: load_results_nation(n) for n in nations}
    print(f"\n  Risultati nel DB : {sum(len(v) for v in results_by_nation.values()):,} partite")

    weeks = list(week_range(start_date, end_date, step))
    print(f"  Settimane        : {len(weeks)}")

    acc_total   = Accumulator()
    weekly_rows = []
    all_rows    = []

    for i, (ws, we) in enumerate(weeks, 1):
        print(f"  [{i:>2}/{len(weeks)}] {ws} -> {we}", end="  ", flush=True)
        rows = run_week(nations, ws, we, results_by_nation, verbose=verbose)
        all_rows.extend(rows)

        acc_w = Accumulator()
        for r in rows:
            if r["giocata"]:
                acc_w.add(r["mercato"], r["esito"], r["is_star"], r["is_best"], r.get("quota"))
                acc_total.add(r["mercato"], r["esito"], r["is_star"], r["is_best"], r.get("quota"))

        seg  = len(rows)
        gioc = sum(1 for r in rows if r["giocata"])
        parts = [f"{seg} seg", f"{gioc} gioc"]
        for m, tag in [("HOME WIN","HW"),("OVER 2.5","O25"),("BTTS","BT"),("PARACADUTE","PARA")]:
            h = acc_w.hit(m)
            if h is not None: parts.append(f"{tag}={h:.0f}%")
        print("  ".join(parts))

        for m in MARKETS:
            d = acc_w.data[m]
            if d["seg"] > 0:
                weekly_rows.append({
                    "week": str(ws), "mercato": m,
                    "segnali": d["seg"], "valutabili": d["val"], "vinti": d["vinti"],
                    "hit_pct": acc_w.hit(m),
                    "star_seg": d["star_seg"], "star_hit_pct": acc_w.star_hit(m),
                    "best_seg": d["best_seg"], "best_val": d["best_val"],
                    "best_vinti": d["best_vinti"], "best_hit_pct": acc_w.best_hit(m),
                    "best_roi_pct": acc_w.roi(m, best=True),
                })

    # Riepilogo
    stampa_summary(acc_total, "RIEPILOGO TOTALE")

    df_res = pd.DataFrame(all_rows)
    if not df_res.empty:
        df_res["quota"]   = pd.to_numeric(df_res["quota"],   errors="coerce")
        df_res["ris_gol"] = pd.to_numeric(df_res["ris_gol"], errors="coerce")
        df_res["tier"]    = pd.to_numeric(df_res["tier"],    errors="coerce")

        for m in MARKETS:
            stampa_breakdown_lega(df_res, m)

        stampa_breakdown_quota(df_res, "HOME WIN")
        stampa_breakdown_quota(df_res, "PARACADUTE")
        stampa_n_segnali(df_res)

        # Confidenza
        print(f"\n  HOME WIN best-only per CONFIDENZA:")
        hw = df_res[(df_res["mercato"]=="HOME WIN") & df_res["giocata"] & df_res["is_best"]]
        for cv in ["Alta","Media","Bassa"]:
            g = hw[hw["confidenza"]==cv]
            n = g["esito"].notna().sum()
            w = (g["esito"]==True).sum()
            if n>0: print(f"    {cv:<8}  {n:>4} segnali  {w:>3} vinti  {round(w/n*100,1):>5}%")

        # B365
        print(f"\n  HOME WIN best-only per B365_CONF:")
        for cv in ["✓ CONF","△ DIV","✗ SKIP","—"]:
            g = hw[hw["b365_conf"]==cv]
            n = g["esito"].notna().sum()
            w = (g["esito"]==True).sum()
            if n>0: print(f"    {cv:<10}  {n:>4} segnali  {w:>3} vinti  {round(w/n*100,1):>5}%")

        # Zone colore
        print(f"\n  HOME WIN -- Grigia / Arancione / Normale (best):")
        for flag, label in [("hw_grigia","Grigia"),("hw_arancione","Arancione"),(None,"Normale")]:
            g = hw[hw[flag]==True] if flag else hw[(hw["hw_grigia"]==False)&(hw["hw_arancione"]==False)]
            n = g["esito"].notna().sum()
            w = (g["esito"]==True).sum()
            if n>0: print(f"    {label:<12}  {n:>4} segnali  {w:>3} vinti  {round(w/n*100,1):>5}%")

    # Salva
    tag = f"{start_str}_to_{end_str}"
    if all_rows:
        p = OUTPUT_DIR / f"backtest_results_{tag}.csv"
        df_res.to_csv(p, index=False)
        print(f"\n[SALVATO] {p}")
    if weekly_rows:
        p = OUTPUT_DIR / f"backtest_summary_{tag}.csv"
        pd.DataFrame(weekly_rows).to_csv(p, index=False)
        print(f"[SALVATO] {p}")

    print(f"\n{'='*76}\n")

if __name__ == "__main__":
    main()