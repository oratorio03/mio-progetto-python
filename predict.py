"""
predict.py — Genera previsioni betting per una o più nazioni.
v5 — Momentum Score + fix gate fisico + Poisson range esteso + confidenza robusta.

Pipeline per ogni partita:
  Step 1: lambda base via Dixon-Coles
  Step 2: correttore forma recente (F_att, F_def)
  Step 3: Poisson → probabilità 1X2, over, BTTS  [range 12 per leghe alto-scoring]
  Step 3.5: blend H2H post-Poisson
  Step 3.6: blend Momentum post-H2H
  Step 4: gate fisico gol attesi → z-score → score → BET
  Step 5: confronto probabilità implicite Bet365

Uso:
    python predict.py england
    python predict.py --all
    python predict.py --all --date=2026-02-27
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from math import exp, factorial
import pytz
import sys
import json
import warnings
warnings.filterwarnings("ignore")

# La cartella root del progetto è il parent di 0_CORE_PIPELINE
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT_DIR   = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROOT_DIR))    # per features/, team_stats, ecc.
sys.path.insert(0, str(_SCRIPT_DIR))

try:
    from features.momentum import (calc_momentum_delta, apply_momentum,
                                    get_standing, get_last_home_result,
                                    empty_momentum)
    MOMENTUM_AVAILABLE = True
except ImportError:
    MOMENTUM_AVAILABLE = False
    def apply_momentum(ph, pd_p, pa, mom): return ph, pd_p, pa, False
    def calc_momentum_delta(*a, **k):      return {}
    def get_standing(*a, **k):             return {}
    def get_last_home_result(*a, **k):     return None
    def empty_momentum():                  return {}

CONFIG_DIR = _ROOT_DIR / "config"
OUTPUT_DIR = _ROOT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

TZ_IT = pytz.timezone("Europe/Rome")
TZ_UK = pytz.timezone("Europe/London")

SCORE_BET      = 55
H2H_MAX_W      = 0.20
H2H_MIN_N      = 4      # era 3 — con 3 osservazioni il peso è già 30%, troppo rumoroso
H2H_SCALE      = 10
ODDS_THRESHOLD = 15.0
ODDS_SKIP_THR  = 25.0

# Gate fisico gol attesi — soglie coerenti con ogni mercato
GATE_OVER25    = 2.80   # alzato da 2.55: sotto 2.8 hit rate 42-43% (backtest 22wk)
GATE_BTTS      = 1.80   # sotto: una delle due quasi certamente non segna
# PARACADUTE = DC (1X) + OVER 1.5 — sostituisce 2T_più
# Gate: H% >= 65 (già filtrato da soglia lega) + O1.5 >= 72%
# Backtest: H%>75 → combo 65.1%, H%>65 → combo 59.4%
GATE_O15_PARA  = 72.0   # soglia minima O1.5% per attivare paracadute
GATE_DC_PARA   = 70.0   # soglia minima DC% (H%+D%) per attivare paracadute
# Soglia score dedicata — PARACADUTE ha base rate 65%+, non serve il bar di HOME WIN
# Il gate fisico fa già la selezione, lo score serve solo come ranking
SCORE_PARA     = 15     # gate è il filtro vero — score serve solo per ★ vs normale

# HOME WIN — soglia minima H% per segnalare (backtest: <65% → hit 45-49%)
HW_MIN_PCT     = 65.0

# Nazioni in zona grigia per HOME WIN — segnale presente ma colorato grigio
# Backtest: France 47.2%, Poland 38.1% → segnale inaffidabile, non bloccare ma avvisare
HW_GRIGIA_NATIONS = {"France", "Poland"}

# Moltiplicatore HOME WIN per nazione — calibrato sul backtest 22 settimane
# >1.0 = lega con forte vantaggio casa strutturale (hit rate confermata)
# <1.0 = lega dove il vantaggio casa è debole o rumoroso
# 1.0  = neutro (non abbastanza dati o hit rate nella media)
HW_NATION_MULT = {
    "Norway":   1.30,   # 73.2% hit — vantaggio casa fortissimo (campi, trasferte)
    "Turkey":   1.20,   # 64.5% hit — Super Lig strutturalmente favorevole alla casa
    "Croatia":  1.15,   # 62.3% hit
    "Romania":  1.12,   # 61.2% hit
    "Austria":  1.05,   # 54.8% — lieve boost
    "Serbia":   1.05,   # 54.3%
    "Belgium":  1.05,   # 54.1%
    "Scotland": 1.03,   # 53.0%
    "Denmark":  1.03,   # 52.8%
    "Portugal": 1.03,   # 52.7%
    "Sweden":   1.00,   # 50.8% — neutro
    "Germany":  1.00,   # 50.7%
    "England":  0.97,   # 50.0% — mercato efficiente, leggera penalizzazione
    "Italy":    0.95,   # 49.5%
    "Spain":    0.95,   # 49.1%
    "France":   0.90,   # 47.2% — già zona grigia, score ridotto
    "Poland":   0.80,   # 38.1% — già zona grigia, score ridotto fortemente
}

# ── FILTRO PER LEGA — HOME WIN ───────────────────────────────────────────────
# Fonte: backtest 22 settimane, 79 leghe analizzate.
# Valore = soglia minima H% per segnalare HOME WIN in quella lega.
# None  = HOME WIN silenziato (lega irrecuperabile: H% non discriminante
#         o base rate home win strutturalmente basso)
# 80    = zona ARANCIONE (dati insufficienti o anomalia — osservazione,
#         non giocare ancora, ma segnale visibile per raccolta dati)
#
# Leghe non presenti nel dizionario → comportamento default (HW_MIN_PCT=65)
HW_LEGA_SOGLIA = {
    # ── LEGHE BUONE — segnalano con soglia calibrata ──────────────────────────
    "Eliteserien":          65,   # Norway T1   72.0% glob
    "2. Lig":               65,   # Turkey T3   67.5% glob
    "HNL":                  65,   # Croatia T1  66.7% glob
    "First NL":             65,   # Croatia T2  65.2% glob
    "Liga I":               65,   # Romania T1  65.0% glob
    "Primeira Liga":        65,   # Portugal T1 64.3% glob
    "1. Lig":               65,   # Turkey T2   62.7% glob
    "2. Division":          65,   # Denmark T3  62.5% glob
    "Bundesliga":           65,   # Austria T1  62.2% glob
    "Premiership":          65,   # Scotland T1 61.7% glob
    "NonLeague Southern":   65,   # England T7  60.2% glob
    "National 2 GA":        65,   # France T4   60.0% glob
    "Prva Liga":            65,   # Serbia T2   60.0% glob
    "Regionalliga SudWest": 55,   # Germany T4  59.5% con 55
    "Superliga":            55,   # Denmark T1  59.1% con 55
    "Ligue 1":              70,   # France T1   61.1% con 70
    "Super Lig":            55,   # Turkey T1   64.3% con 55
    "Segunda RFEF G4":      55,   # Spain T4    62.5% con 55
    "Serie D GF":           55,   # Italy T4    63.8% con 55
    "Serie D GC":           65,   # Italy T4    63.6% con 65
    "National 3 GE":        55,   # France T5   63.6% con 55
    "Second NL":            55,   # Croatia T3  62.5% con 55
    "La Liga":              55,   # Spain T1    62.1% con 55
    "Regionalliga Nordost": 55,   # Germany T4  61.5% con 55
    "2. Bundesliga":        70,   # Germany T2  61.5% con 70
    "National 3 GG":        55,   # France T5   61.3% con 55
    "Serie D GH":           70,   # Italy T4    60.9% con 70
    "National 3 GH":        55,   # France T5   60.9% con 55
    "Premier League":       75,   # England T1  60.7% con 75
    "Liga 3":               55,   # Portugal T3 60.7% con 55
    "Serie B":              55,   # Italy T2    60.6% con 55
    "National League":      55,   # England T5  60.2% con 55
    "Liga II":              55,   # Romania T2  60.0% con 55
    "Serie D GD":           55,   # Italy T4    60.0% con 55
    "Serie D GA":           75,   # Italy T4    60.0% con 75
    "National 3 GF":        70,   # France T5   60.0% con 70
    "Serie D GB":           60,   # Italy T4    60.0% con 60
    "National 1":           75,   # France T3   60.0% con 75
    "Challenger Pro League":55,   # Belgium T2  70.0% con 55
    "Serie C GC":           65,   # Italy T3    69.2% con 65
    "Jupiler Pro League":   60,   # Belgium T1  69.2% con 60
    "1. Division":          60,   # Denmark T2  66.7% con 60
    "Segunda RFEF G2":      80,   # Spain T4    66.7% con 80 (n=9, zona arancione)
    "Serie D GG":           75,   # Italy T4    65.0% con 75
    "Segunda RFEF G3":      80,   # Spain T4    66.7% con 80
    "National 3 GB":        80,   # France T5   zona arancione — anomalia, dati scarsi
    "3. Liga":              80,   # Germany T3  71.4% con 80 — zona arancione
    "NonLeague Northern":   80,   # England T7  80.0% con 80 — zona arancione
    "Ettan Norra":          80,   # Sweden T3   66.7% con 80 — pochi dati stagione solare
    "National 3 GD":        80,   # France T5   dati insufficienti
    "National 2 GB":        80,   # France T4   anomalia — osservazione
    "National 3 GA":        80,   # France T5   anomalia — osservazione
    "National 3 GC":        80,   # France T5   anomalia — osservazione
    # ── LEGHE IRRECUPERABILI — HOME WIN silenziato ────────────────────────────
    "2. Liga":              None,  # Austria T2   H% non discriminante
    "Regionalliga Bayern":  None,  # Germany T4   H% non discriminante
    "Primera RFEF G1":      None,  # Spain T3     H% non discriminante
    "League One":           None,  # England T3   H% non discriminante (159 seg)
    "Segunda División":     None,  # Spain T2     H% non discriminante
    "Segunda RFEF G5":      None,  # Spain T4     H% non discriminante
    "Championship":         None,  # Scotland T2  H% non discriminante
    "League Two":           None,  # England T4   H% non discriminante
    "NL North":             None,  # England T6   H% non discriminante
    "NL South":             None,  # England T6   strutturale
    "Ekstraklasa":          None,  # Poland T1    H% non discriminante
    "Regionalliga West":    None,  # Germany T4   H% non discriminante
    "Regionalliga Nord":    None,  # Germany T4   H% non discriminante
    "I Liga":               None,  # Poland T2    strutturale
    "Segunda Liga":         None,  # Portugal T2  H% non discriminante
    "Serie D GI":           None,  # Italy T4     H% non discriminante
    "Primera RFEF G2":      None,  # Spain T3     H% non discriminante
    "Ligue 2":              None,  # France T2    strutturale
    "National 2 GC":        None,  # France T4    strutturale
    "Super Liga":           None,  # Serbia T1    strutturale
    "NonLeague Isthmian":   None,  # England T7   strutturale
    "Serie C GA":           None,  # Italy T3     strutturale
    "Serie A":              None,  # Italy T1     strutturale (base rate 44%)
    "Serie C GB":           None,  # Italy T3     strutturale
    "Serie D GE":           None,  # Italy T4     strutturale
    "Segunda RFEF G1":      None,  # Spain T4     anomalia severa (31% hit)
}

# Leghe in zona ARANCIONE — segnale visibile ma non affidabile (dati scarsi/anomalia)
HW_ARANCIONE_SOGLIA = 80   # soglia unica per tutte le leghe arancione

# ── CONFIG ────────────────────────────────────────────────────────────────────
def load_config(nation_code):
    with open(Path(CONFIG_DIR) / f"{nation_code}.json", "r", encoding="utf-8") as f:
        return json.load(f)

def get_paths(nation_code, cutoff=None):
    proc      = _ROOT_DIR / "data" / nation_code / "processed"
    team_file = f"team_stats_cutoff_{cutoff}.csv" if cutoff else "team_stats.csv"
    return {
        "fixtures":  proc / "all_fixtures.csv",
        "results":   proc / "all_results.csv",
        "teams":     proc / team_file,
        "leagues":   proc / "variance_stats.csv",
        "odds":      proc / "odds.csv",
        "standings": proc / "standings.csv",
    }

# ── DATE WINDOW ───────────────────────────────────────────────────────────────
def get_date_window(from_date=None):
    if from_date:
        today = datetime.strptime(from_date, "%Y-%m-%d").date()
    else:
        today = datetime.now(TZ_IT).date()
    weekday  = today.weekday()
    days_to_sunday = (6 - weekday) % 7
    # Fix: se oggi è domenica vogliamo includere oggi, non fermarci a ieri
    if days_to_sunday == 0:
        end_date = today
    else:
        end_date = today + timedelta(days=days_to_sunday)
    return today, end_date

# ── UTILS ─────────────────────────────────────────────────────────────────────
def uk_to_it(date_str, time_str):
    try:
        dt_uk = TZ_UK.localize(datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M"))
        return dt_uk.astimezone(TZ_IT).strftime("%H:%M")
    except:
        return time_str

def get_val(prof, *cols):
    for col in cols:
        v = prof.get(col)
        try:
            fv = float(v)
            if not np.isnan(fv):
                return fv
        except:
            pass
    return None

# ── POISSON ───────────────────────────────────────────────────────────────────
# range 12 invece di 8/9 — necessario per leghe nordiche/est con lambda 4-6
MAX_GOALS = 12

def poisson_p(lam, k):
    if lam <= 0:
        return 0.0
    return (lam**k * exp(-lam)) / factorial(k)

def prob_over(lh, la, thr):
    p = sum(poisson_p(lh, h) * poisson_p(la, a)
            for h in range(MAX_GOALS) for a in range(MAX_GOALS) if h + a <= thr)
    return round((1 - p) * 100, 1)

def prob_btts(lh, la):
    return round((1 - poisson_p(lh, 0)) * (1 - poisson_p(la, 0)) * 100, 1)

def prob_1x2(lh, la):
    ph = pd = pa = 0.0
    for h in range(MAX_GOALS):
        for a in range(MAX_GOALS):
            p = poisson_p(lh, h) * poisson_p(la, a)
            if   h > a:  ph += p
            elif h == a: pd += p
            else:        pa += p
    tot = ph + pd + pa or 1
    return round(ph/tot*100, 1), round(pd/tot*100, 1), round(pa/tot*100, 1)

# ── LAMBDA BASE ───────────────────────────────────────────────────────────────
def calc_lambda_base(hp, ap, avg_gol):
    lm = (avg_gol / 2) or 1.3
    ha = get_val(hp, "C_H_gol_fatti",  "L2_H_gol_fatti",  "L5_H_gol_fatti")  or lm
    hd = get_val(hp, "C_H_gol_subiti", "L2_H_gol_subiti", "L5_H_gol_subiti") or lm
    aa = get_val(ap, "C_A_gol_fatti",  "L2_A_gol_fatti",  "L5_A_gol_fatti")  or lm
    ad = get_val(ap, "C_A_gol_subiti", "L2_A_gol_subiti", "L5_A_gol_subiti") or lm
    return round((ha * ad) / lm, 3), round((aa * hd) / lm, 3)

# ── FORMA ─────────────────────────────────────────────────────────────────────
def get_forma_mult(prof, key):
    v = get_val(prof, key)
    return v if v is not None else 1.0

def apply_forma(lh_base, la_base, hp, ap):
    lh = max(0.3, min(8.0, lh_base * get_forma_mult(hp, "F_att") * get_forma_mult(ap, "F_def")))
    la = max(0.3, min(8.0, la_base * get_forma_mult(ap, "F_att") * get_forma_mult(hp, "F_def")))
    return round(lh, 3), round(la, 3)

# ── H2H ───────────────────────────────────────────────────────────────────────
def calc_h2h(home_id, away_id, results_df, cutoff_date=None):
    if results_df is None or results_df.empty:
        return _empty_h2h()
    mask = (
        ((results_df["home_id"] == home_id) & (results_df["away_id"] == away_id)) |
        ((results_df["home_id"] == away_id) & (results_df["away_id"] == home_id))
    )
    h2h = results_df[mask].copy()
    if cutoff_date:
        h2h = h2h[h2h["date"] < pd.to_datetime(cutoff_date)]
    h2h = h2h.sort_values("date", ascending=False).head(10)
    n   = len(h2h)
    if n < H2H_MIN_N:
        return _empty_h2h()

    # Decay temporale: partite più recenti pesano di più
    # Fattore 0.75 per posizione → la più vecchia (pos 9) pesa ~0.75^9 ≈ 0.075
    decay = [0.75 ** i for i in range(n)]
    w_sum = sum(decay)
    decay = [w / w_sum for w in decay]

    hw = dw = aw = 0.0
    total_gol = 0.0
    for i, (_, row) in enumerate(h2h.iterrows()):
        hg = row.get("home_goals")
        ag = row.get("away_goals")
        if pd.isna(hg) or pd.isna(ag):
            continue
        hg, ag  = int(hg), int(ag)
        w       = decay[i]
        total_gol += (hg + ag) * w
        if row["home_id"] == home_id:
            if hg > ag:    hw += w
            elif hg == ag: dw += w
            else:          aw += w
        else:
            if ag > hg:    hw += w
            elif ag == hg: dw += w
            else:          aw += w

    tot = hw + dw + aw or 1
    return {
        "h2h_n":       n,
        "h2h_hw":      round(hw / tot * 100, 1),
        "h2h_dw":      round(dw / tot * 100, 1),
        "h2h_aw":      round(aw / tot * 100, 1),
        "h2h_avg_gol": round(total_gol, 2),
        "h2h_w":       round(min(n / H2H_SCALE, H2H_MAX_W), 3),
    }

def _empty_h2h():
    return {"h2h_n": 0, "h2h_hw": None, "h2h_dw": None, "h2h_aw": None,
            "h2h_avg_gol": None, "h2h_w": 0.0}

def apply_h2h(ph, pd_prob, pa, h2h):
    w = h2h.get("h2h_w", 0.0)
    if w == 0 or h2h["h2h_hw"] is None:
        return ph, pd_prob, pa, False
    ph_new = (1-w)*ph      + w*h2h["h2h_hw"]
    pd_new = (1-w)*pd_prob + w*h2h["h2h_dw"]
    pa_new = (1-w)*pa      + w*h2h["h2h_aw"]
    tot    = ph_new + pd_new + pa_new or 1
    return round(ph_new/tot*100, 1), round(pd_new/tot*100, 1), round(pa_new/tot*100, 1), True

# ── ODDS ─────────────────────────────────────────────────────────────────────
def get_odds_row(fixture_id, odds_map):
    if odds_map is None or fixture_id not in odds_map:
        return _empty_odds()
    return odds_map[fixture_id]

def _empty_odds():
    return {"q1": None, "qx": None, "q2": None,
            "imp_h": None, "imp_d": None, "imp_a": None}

def odds_conferma(ph, bet1, odds_row):
    """
    Conferma Bet365 contestualizzata al mercato segnalato.
    Se il segnale è HOME WIN confronta H% vs imp_h.
    Se il segnale è OVER/BTTS confronta solo che Bet365 abbia la quota (presenza = conferma base).
    """
    imp_h = odds_row.get("imp_h")
    if imp_h is None:
        return "—", None

    # Per HOME WIN: confronto diretto probabilità
    if bet1 and "HOME WIN" in str(bet1):
        delta = abs(ph - imp_h)
        if delta <= ODDS_THRESHOLD:  return "✓ CONF", round(delta, 1)
        elif delta <= ODDS_SKIP_THR: return "△ DIV",  round(delta, 1)
        else:                        return "✗ SKIP",  round(delta, 1)

    # Per altri mercati: Bet365 ha la quota = dato disponibile, nessun conflitto 1X2
    delta = abs(ph - imp_h)
    if delta <= ODDS_THRESHOLD:  return "✓ CONF", round(delta, 1)
    elif delta <= ODDS_SKIP_THR: return "△ DIV",  round(delta, 1)
    else:                        return "△ DIV",  round(delta, 1)  # non SKIP per mercati diversi

def _to_float(v):
    """Converte quota a float, None se non disponibile."""
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None

# ── Z-SCORE E SCORE ───────────────────────────────────────────────────────────
def zscore(v, mean, std):
    if not std or std == 0 or np.isnan(std):
        return 0.0
    return (v - mean) / std

def calc_score(z, conf):
    base = min(max((z / 2.0) * 100, 0), 100)
    mult = {"Alta": 1.0, "Media": 0.85, "Bassa": 0.65}.get(conf, 0.75)
    return round(base * mult, 1)

def get_confidenza(hp, ap):
    """
    Confidenza robusta: usa stagione corrente se disponibile,
    altrimenti fallback su dati storici L2/L5.
    Evita che tutte le nuove leghe a inizio stagione siano 'Bassa'.
    """
    def best_n(prof):
        # Stagione corrente
        cn = prof.get("C_n", 0) or 0
        if cn >= 8:
            return cn
        # Fallback storico
        l2n = prof.get("L2_n", 0) or 0
        l5n = prof.get("L5_n", 0) or 0
        return max(cn, l2n // 3, l5n // 5)  # normalizza su scala simile

    mn = min(best_n(hp), best_n(ap))
    return "Alta" if mn >= 15 else ("Media" if mn >= 8 else "Bassa")

def get_t2(prof, avg_2t):
    v = get_val(prof, "C_t2more%", "L2_t2more%", "L5_t2more%")
    return v if v is not None else avg_2t

# ── SEGNALE MONOGRAFICO ──────────────────────────────────────────────────────
def _sig(scores, market, mom_delta=0.0, ph=0.0, lega=""):
    """
    Ritorna il segnale per un mercato specifico oppure None.
    ★ se score >= 70, testo normale se >= SCORE_BET, None altrimenti.
    HOME WIN:
      - soppressa se mom_delta < -0.15
      - soppressa se lega in HW_LEGA_SOGLIA con soglia None (irrecuperabile)
      - soppressa se ph < soglia lega (o HW_MIN_PCT se lega non in dizionario)
    """
    if market == "HOME WIN":
        if mom_delta < -0.15:
            return None
        # Soglia per lega dal dizionario backtest
        if lega in HW_LEGA_SOGLIA:
            soglia = HW_LEGA_SOGLIA[lega]
            if soglia is None:
                return None          # lega irrecuperabile — silenzio totale
            if ph < soglia:
                return None
        else:
            if ph < HW_MIN_PCT:      # fallback globale per leghe non censite
                return None
    score     = scores.get(market, 0)
    threshold = SCORE_PARA if market == "PARACADUTE" else SCORE_BET
    if score <= 0:
        return None
    if score >= 70:
        return f"★ {market}"
    if score >= threshold:
        return market
    return None

# ── PREDICT MATCH ─────────────────────────────────────────────────────────────
def predict_match(hp, ap, lg, h2h, odds_row, momentum=None, nation="", lega=""):
    def lv(key, default):
        return float(lg.get(key, default) or default)

    avg_gol = lv("avg_gol_mean", 2.6)
    avg_o25 = lv("O2.5%_mean",  48.0);  std_o25 = lv("O2.5%_std",  3.5)
    avg_bt  = lv("BTTS%_mean",  52.0);  std_bt  = lv("BTTS%_std",  3.0)
    avg_o15 = lv("O1.5%_mean",  74.0);  std_o15 = lv("O1.5%_std",  3.0)
    avg_hw  = lv("H%_mean",     43.0);  std_hw  = lv("H%_std",     2.5)

    lh_base, la_base = calc_lambda_base(hp, ap, avg_gol)
    lh, la           = apply_forma(lh_base, la_base, hp, ap)
    ph, pd_p, pa     = prob_1x2(lh, la)
    ph, pd_p, pa, _  = apply_h2h(ph, pd_p, pa, h2h)

    # Step 3.6 — Momentum post-H2H
    mom = momentum or {}
    ph, pd_p, pa, _  = apply_momentum(ph, pd_p, pa, mom)

    # Clamp probabilità in [0.1, 99.9] — momentum/H2H possono spingere fuori range
    ph   = max(0.1, min(99.9, ph))
    pd_p = max(0.1, min(99.9, pd_p))
    pa   = max(0.1, min(99.9, pa))
    # Rinormalizza a 100
    _tot = ph + pd_p + pa
    ph, pd_p, pa = round(ph/_tot*100,1), round(pd_p/_tot*100,1), round(pa/_tot*100,1)

    # Cap lambda a 4.0 per Over/BTTS/gate — 1X2 già calcolato con lambda raw
    lh = min(lh, 4.0)
    la = min(la, 4.0)

    o25 = prob_over(lh, la, 2)
    o35 = prob_over(lh, la, 3)
    o15 = prob_over(lh, la, 1)
    bt  = prob_btts(lh, la)
    dc  = round(ph + pd_p, 1)   # DC = P(casa) + P(pareggio)

    conf   = get_confidenza(hp, ap)
    # Score PARACADUTE: quanto O1.5% e DC% superano i gate (non z-score globale)
    # Il gate fisico è il filtro vero — lo score serve solo come ranking interno
    # Usiamo eccesso sopra gate, normalizzato: O1.5 va da 72 a ~95, DC da 70 a ~99
    exc_o15 = max(0, o15  - GATE_O15_PARA) / (95  - GATE_O15_PARA)   # 0-1
    exc_dc  = max(0, dc   - GATE_DC_PARA)  / (99  - GATE_DC_PARA)    # 0-1
    score_para = round(min((exc_o15 + exc_dc) / 2 * 100, 100), 1)    # 0-100
    scores = {
        "OVER 2.5":   calc_score(zscore(o25, avg_o25, std_o25), conf),
        "BTTS":       calc_score(zscore(bt,  avg_bt,  std_bt),  conf),
        "PARACADUTE": score_para,
        "HOME WIN":   calc_score(zscore(ph,  avg_hw,  std_hw),  conf),
    }

    # Moltiplicatore lega per HOME WIN — boost/penalizza basato su backtest
    hw_mult = HW_NATION_MULT.get(nation, 1.0)
    if hw_mult != 1.0:
        scores["HOME WIN"] = round(min(scores["HOME WIN"] * hw_mult, 100), 1)

    # Gate fisico — soglie coerenti con ogni mercato
    gol_tot = lh + la
    if gol_tot < GATE_OVER25: scores["OVER 2.5"]   = 0
    if gol_tot < GATE_BTTS:   scores["BTTS"]        = 0
    # PARACADUTE: gate su O1.5% e DC%
    if o15 < GATE_O15_PARA or dc < GATE_DC_PARA:
        scores["PARACADUTE"] = 0
    # Quota combo paracadute: odd_1x * odd_o15
    odd_1x  = odds_row.get("odd_1x")
    odd_o15 = odds_row.get("odd_o15")
    q_para  = round(float(odd_1x) * float(odd_o15), 2) if odd_1x and odd_o15 else None

    mom_delta = mom.get("momentum_delta", 0.0)
    # note sintetica per debug
    active = [m for m in ["HOME WIN","BTTS","OVER 2.5","PARACADUTE"]
              if _sig(scores, m, mom_delta, ph=(ph if m=="HOME WIN" else 0.0), lega=(lega if m=="HOME WIN" else "")) is not None]
    note = " | ".join(f"{m}={scores[m]:.0f}" for m in active) if active else f"max={max(scores.values()):.0f}"
    pred_1x2 = "H" if ph >= pd_p and ph >= pa else ("A" if pa >= ph and pa >= pd_p else "D")

    first_sig = next((s for s in [_sig(scores,"HOME WIN",mom_delta,ph=ph,lega=lega),
                                      _sig(scores,"BTTS",mom_delta),
                                      _sig(scores,"OVER 2.5",mom_delta),
                                      _sig(scores,"PARACADUTE",mom_delta)] if s), None)
    conf_label, conf_delta = odds_conferma(ph, first_sig, odds_row)

    return {
        "H%": ph, "D%": pd_p, "A%": pa, "pred_1x2": pred_1x2,
        "gol_tot":     round(gol_tot, 2),
        "O2.5%": o25,  "O3.5%": o35,  "O1.5%": o15,
        "BTTS%": bt,   "DC%": dc,
        "h2h_n":       h2h.get("h2h_n", 0),
        "h2h_hw":      h2h.get("h2h_hw"),
        "h2h_avg_gol": h2h.get("h2h_avg_gol"),
        "h2h_w":       h2h.get("h2h_w", 0),
        "q1":          odds_row.get("q1"),
        "qx":          odds_row.get("qx"),
        "q2":          odds_row.get("q2"),
        "imp_h":       odds_row.get("imp_h"),
        "imp_d":       odds_row.get("imp_d"),
        "imp_a":       odds_row.get("imp_a"),
        "b365_conf":   conf_label,
        "b365_delta":  conf_delta,
        "f_att_h":     get_forma_mult(hp, "F_att"),
        "f_def_h":     get_forma_mult(hp, "F_def"),
        "f_att_a":     get_forma_mult(ap, "F_att"),
        "f_def_a":     get_forma_mult(ap, "F_def"),
        "confidenza":  conf,
        "sig_hw":    _sig(scores, "HOME WIN",  mom_delta, ph=ph, lega=lega),
        "sig_btts":  _sig(scores, "BTTS",      mom_delta),
        "sig_o25":   _sig(scores, "OVER 2.5",  mom_delta),
        "sig_para":  _sig(scores, "PARACADUTE",mom_delta),
        "q_para":    q_para,
        "note": note,
        "mom_delta":    mom_delta,
        "mom_home_sit": mom.get("home_situation", "—"),
        "mom_away_sit": mom.get("away_situation", "—"),
        "urgenza_h":    mom.get("urgenza_home", 0.0),
        "urgenza_a":    mom.get("urgenza_away", 0.0),
        "zona_morta":   mom.get("zona_morta_flag", False),
    }

# ── BUILD NATION ──────────────────────────────────────────────────────────────
def build_nation(nation_code, cfg, start_date, end_date, backtest=False, cutoff=None):
    paths  = get_paths(nation_code, cutoff=cutoff)
    nation = cfg["nation"]

    src_key    = "results" if backtest else "fixtures"
    check_keys = [src_key, "results", "teams", "leagues"]
    for k in check_keys:
        if not paths[k].exists():
            print(f"  [SKIP] {nation}: {paths[k].name} non trovato")
            return pd.DataFrame(), pd.DataFrame()

    fixtures   = pd.read_csv(paths[src_key])
    fixtures["date"] = pd.to_datetime(fixtures["date"]).dt.date
    teams      = pd.read_csv(paths["teams"])
    leagues    = pd.read_csv(paths["leagues"])
    results_df = pd.read_csv(paths["results"])
    results_df["date"] = pd.to_datetime(results_df["date"])
    results_df = results_df[results_df["played"] == 1]
    for col in ["home_goals", "away_goals"]:
        results_df[col] = pd.to_numeric(results_df[col], errors="coerce")

    # Odds
    odds_map = None
    if paths["odds"].exists():
        odds_df  = pd.read_csv(paths["odds"])
        odds_map = {row["fixture_id"]: row.to_dict() for _, row in odds_df.iterrows()}
        print(f"  [{nation}] Quote Bet365: {len(odds_map)} fixture")
    else:
        print(f"  [{nation}] Quote Bet365: non disponibili")

    # Standings / Momentum
    standings_df = None
    if MOMENTUM_AVAILABLE and paths["standings"].exists():
        standings_df = pd.read_csv(paths["standings"])
        print(f"  [{nation}] Standings: {len(standings_df)} squadre")
    else:
        print(f"  [{nation}] Standings: non disponibili — momentum disabilitato")

    total_season_games_map = {}
    for league_id_str, info in cfg["leagues"].items():
        n_teams = info.get("n_teams", 20)
        total_season_games_map[int(league_id_str)] = (n_teams - 1) * 2

    lg_map     = {row["lega"]: row for _, row in leagues.iterrows()}
    tm_map     = {row["team_id"]: row for _, row in teams.iterrows()}
    id_to_name = {int(k): v["name"] for k, v in cfg["leagues"].items()}

    mask = (fixtures["date"] >= start_date) & (fixtures["date"] <= end_date)
    fx   = fixtures[mask].copy()
    mode = "BACKTEST" if backtest else "LIVE"
    print(f"  {nation}: {len(fx)} partite [{mode}]")

    pred_rows  = []
    forma_rows = []

    for _, fix in fx.iterrows():
        lg_key = str(fix.get("league_name", ""))
        lg     = lg_map.get(lg_key)
        hp     = tm_map.get(fix["home_id"])
        ap     = tm_map.get(fix["away_id"])
        if lg is None or hp is None or ap is None:
            continue

        lid     = int(fix.get("league_id", 0))
        lg_name = id_to_name.get(lid, lg_key)
        ora_it  = uk_to_it(str(fix["date"]), str(fix.get("time", "00:00")))
        h2h     = calc_h2h(fix["home_id"], fix["away_id"], results_df, cutoff_date=cutoff)
        odds_row= get_odds_row(fix.get("fixture_id"), odds_map)

        momentum = None
        if MOMENTUM_AVAILABLE and standings_df is not None:
            tsg        = total_season_games_map.get(lid, 38)
            h_standing = get_standing(fix["home_id"], standings_df, tsg)
            a_standing = get_standing(fix["away_id"], standings_df, tsg)
            last_home  = get_last_home_result(fix["home_id"], results_df, cutoff_date=cutoff)
            momentum   = calc_momentum_delta(h_standing, a_standing, dict(hp), dict(ap), last_home)

        pred = predict_match(hp, ap, lg, h2h, odds_row, momentum=momentum, nation=nation, lega=lg_name)

        # ── REGOLA UN SEGNALE PER PARTITA ────────────────────────────────────
        # Se più mercati sono sopra soglia, evidenzia solo quello con quota più
        # alta (valore atteso maggiore). Gli altri restano visibili ma dimmed.
        #
        # Mappa mercato → (segnale, quota di riferimento)
        sig_map = {
            "HOME WIN":   (pred["sig_hw"],   _to_float(pred.get("q1"))),
            "BTTS":       (pred["sig_btts"], _to_float(odds_row.get("odd_btts"))),
            "OVER 2.5":   (pred["sig_o25"],  _to_float(odds_row.get("odd_o25"))),
            "PARACADUTE": (pred["sig_para"], _to_float(pred.get("q_para"))),
        }
        active_sigs = [(mkt, sig, q) for mkt, (sig, q) in sig_map.items() if sig]
        if len(active_sigs) <= 1:
            best_mkt   = active_sigs[0][0] if active_sigs else ""
            dimmed_mkts = []
        else:
            # Ordina per quota desc — quota None vale 0 (va in fondo)
            active_sigs.sort(key=lambda x: x[2] if x[2] else 0, reverse=True)
            best_mkt    = active_sigs[0][0]
            dimmed_mkts = [x[0] for x in active_sigs[1:]]

        pred_rows.append({
            "Data":       str(fix["date"]),
            "Ora (IT)":   ora_it,
            "Nazione":    nation,
            "Lega":       lg_name,
            "Tier":       int(fix.get("tier", 0)),
            "Casa":       fix["home_name"],
            "Trasferta":  fix["away_name"],
            "HOME WIN":   pred["sig_hw"],
            "BTTS":       pred["sig_btts"],
            "OVER 2.5":   pred["sig_o25"],
            "PARACADUTE": pred["sig_para"],
            "Q_Para":     pred["q_para"],
            "BestSig":    best_mkt,
            "DimmedSigs": "|".join(dimmed_mkts),
            "HW_Grigia":    pred["sig_hw"] is not None and nation in HW_GRIGIA_NATIONS,
            "HW_Arancione": pred["sig_hw"] is not None and HW_LEGA_SOGLIA.get(lg_name) == 80,
            "HW_Mult":    HW_NATION_MULT.get(nation, 1.0),
            "H%":         pred["H%"],
            "D%":         pred["D%"],
            "A%":         pred["A%"],
            "Pred 1X2":   pred["pred_1x2"],
            "Q1":         pred["q1"],
            "QX":         pred["qx"],
            "Q2":         pred["q2"],
            "B365_H%":    pred["imp_h"],
            "B365_D%":    pred["imp_d"],
            "B365_A%":    pred["imp_a"],
            "Gol Attesi": pred["gol_tot"],
            "O2.5%":      pred["O2.5%"],
            "O3.5%":      pred["O3.5%"],
            "O1.5%":      pred["O1.5%"],
            "BTTS%":      pred["BTTS%"],
            "DC%":        pred["DC%"],
            "odd_btts":   odds_row.get("odd_btts"),
            "odd_o25":    odds_row.get("odd_o25"),
            "B365_Conf":  pred["b365_conf"],
            "b365_delta": pred["b365_delta"],
            "Confidenza": pred["confidenza"],
            "Note":       pred["note"],
            "Mom_Delta":  pred["mom_delta"],
            "Mom_Casa":   pred["mom_home_sit"],
            "Mom_Ospite": pred["mom_away_sit"],
            "Zona_Morta": "⚠" if pred["zona_morta"] else "",
            "H2H_n":      pred["h2h_n"],
            "H2H_hw%":    pred["h2h_hw"],
        })

        for ruolo, prof, f_att, f_def in [
            ("Casa",      hp, pred["f_att_h"], pred["f_def_h"]),
            ("Trasferta", ap, pred["f_att_a"], pred["f_def_a"]),
        ]:
            forma_str_parts = []
            for i in range(1, 7):
                gf  = prof.get(f"F_gf{i}")
                gs  = prof.get(f"F_gs{i}")
                v   = prof.get(f"F_v{i}")
                win = prof.get(f"F_win{i}")
                if gf is not None and not (isinstance(gf, float) and np.isnan(gf)):
                    esito = "W" if win == 1 else ("D" if win == 0 and gs == gf else "L")
                    forma_str_parts.append(f"{int(gf)}-{int(gs)}({v},{esito})")

            forma_rows.append({
                "Data":       str(fix["date"]),
                "Nazione":    nation,
                "Casa":       fix["home_name"],
                "Trasferta":  fix["away_name"],
                "Lega":       lg_name,
                "Ruolo":      ruolo,
                "Squadra":    prof["team_name"],
                "N partite":  prof.get("C_n", 0),
                "Win%":       get_val(prof, "C_win%",       "L2_win%",       "L5_win%"),
                "Gol Fatti":  get_val(prof, "C_gol_fatti",  "L2_gol_fatti",  "L5_gol_fatti"),
                "Gol Subiti": get_val(prof, "C_gol_subiti", "L2_gol_subiti", "L5_gol_subiti"),
                "CS%":        get_val(prof, "C_cs%",        "L2_cs%",        "L5_cs%"),
                "BTTS%":      get_val(prof, "C_btts%",      "L2_btts%",      "L5_btts%"),
                "O2.5%":      get_val(prof, "C_over25%",    "L2_over25%",    "L5_over25%"),
                "2T_più%":    get_val(prof, "C_t2more%",    "L2_t2more%",    "L5_t2more%"),
                "H Win%":     get_val(prof, "C_H_win%",     "L2_H_win%",     "L5_H_win%"),
                "A Win%":     get_val(prof, "C_A_win%",     "L2_A_win%",     "L5_A_win%"),
                "F_n":        prof.get("F_n"),
                "F_att":      f_att,
                "F_def":      f_def,
                "F_win":      prof.get("F_win"),
                "Ultimi 6":   " | ".join(forma_str_parts),
                "H2H_n":      pred["h2h_n"],
                "H2H_hw%":    pred["h2h_hw"],
                "H2H_gol":    pred["h2h_avg_gol"],
                "H2H_peso":   pred["h2h_w"],
            })

    return pd.DataFrame(pred_rows), pd.DataFrame(forma_rows)

# ── SAVE EXCEL ────────────────────────────────────────────────────────────────
def save_excel(df_pred, df_forma, start_date, end_date):
    from openpyxl import load_workbook
    from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    fname = OUTPUT_DIR / f"predictions_{start_date}_to_{end_date}.xlsx"

    # ── Costruisci df_main (tab Previsioni) e df_det (tab Dettagli) ──────────
    # Colonne principali — ordinate come richiesto
    COLS_MAIN = [
        "Data", "Ora (IT)", "Nazione", "Lega", "Casa", "Trasferta",
        # scommesse
        "Scommessa 1", "Scommessa 2",
        # probabilità nostre
        "H%", "X%", "A%", "BTTS%", "O2.5%", "DC%", "O1.5%",
        # quote bookmaker
        "Q1", "QX", "Q2", "Q_BTTS", "Q_O25", "Q_DC+O15",
        # flag interni per coloring (nascosti)
        "_BestMkt", "_DimmedMkts", "_HW_Grigia", "_HW_Arancione",
    ]
    # Colonne dettagli — tutto il resto
    COLS_DET = [
        "Data", "Ora (IT)", "Nazione", "Lega", "Casa", "Trasferta",
        "Gol Attesi", "B365_Conf", "Δ Mercato",
        "Momentum", "Situazione Casa", "Situazione Ospite", "Zona Morta",
        "H2H Partite", "H2H Casa%",
        "HOME WIN", "BTTS", "OVER 2.5", "PARACADUTE",   # segnali raw per backtest
    ]

    def _label(sig):
        """Traduce il segnale interno in etichetta italiana leggibile."""
        if not sig or str(sig) in ("", "None", "nan"): return ""
        s = str(sig)
        star = "★ " if "★" in s else ""
        mkt  = (s.replace("★","").strip())
        MAP  = {
            "HOME WIN":   "Vittoria Casa",
            "BTTS":       "Goal Goal",
            "OVER 2.5":   "Over 2.5",
            "PARACADUTE": "DC + Over 1.5",
        }
        return star + MAP.get(mkt, mkt)

    rows_main = []
    rows_det  = []
    for _, r in df_pred.iterrows():
        best_raw    = str(r.get("BestSig","") or "")
        dimmed_raw  = str(r.get("DimmedSigs","") or "")
        # Prima scommessa = best, Seconda = prima dei dimmed (se esiste)
        dimmed_list = [d.strip() for d in dimmed_raw.split("|") if d.strip()]
        sec_raw     = dimmed_list[0] if dimmed_list else ""
        # ricostruisci segnale completo per best (con ★ se presente)
        best_sig_full = ""
        for col in ["HOME WIN","BTTS","OVER 2.5","PARACADUTE"]:
            sv = str(r.get(col,"") or "")
            if sv and sv not in ("","None","nan"):
                mkt = sv.replace("★","").strip()
                if mkt == best_raw:
                    best_sig_full = sv
                    break
        sec_sig_full = ""
        for col in ["HOME WIN","BTTS","OVER 2.5","PARACADUTE"]:
            sv = str(r.get(col,"") or "")
            if sv and sv not in ("","None","nan"):
                mkt = sv.replace("★","").strip()
                if mkt == sec_raw:
                    sec_sig_full = sv
                    break

        rows_main.append({
            "Data":         r.get("Data",""),
            "Ora (IT)":     r.get("Ora (IT)",""),
            "Nazione":      r.get("Nazione",""),
            "Lega":         r.get("Lega",""),
            "Casa":         r.get("Casa",""),
            "Trasferta":    r.get("Trasferta",""),
            "Scommessa 1":  _label(best_sig_full),
            "Scommessa 2":  _label(sec_sig_full),
            "H%":           r.get("H%"),
            "X%":           r.get("D%"),
            "A%":           r.get("A%"),
            "BTTS%":        r.get("BTTS%"),
            "O2.5%":        r.get("O2.5%"),
            "DC%":          r.get("DC%"),
            "O1.5%":        r.get("O1.5%"),
            "Q1":           r.get("Q1"),
            "QX":           r.get("QX"),
            "Q2":           r.get("Q2"),
            "Q_BTTS":       r.get("odd_btts"),
            "Q_O25":        r.get("odd_o25"),
            "Q_DC+O15":     r.get("Q_Para"),
            # flag interni per coloring — nascosti nell'Excel
            "_BestMkt":     best_raw,
            "_DimmedMkts":  dimmed_raw,
            "_HW_Grigia":   r.get("HW_Grigia", False),
            "_HW_Arancione":r.get("HW_Arancione", False),
        })
        rows_det.append({
            "Data":             r.get("Data",""),
            "Ora (IT)":         r.get("Ora (IT)",""),
            "Nazione":          r.get("Nazione",""),
            "Lega":             r.get("Lega",""),
            "Casa":             r.get("Casa",""),
            "Trasferta":        r.get("Trasferta",""),
            "Gol Attesi":       r.get("Gol Attesi"),
            "B365_Conf":        r.get("B365_Conf",""),
            "Δ Mercato":        r.get("b365_delta"),
            "Momentum":         r.get("Mom_Delta"),
            "Situazione Casa":  r.get("Mom_Casa",""),
            "Situazione Ospite":r.get("Mom_Ospite",""),
            "Zona Morta":       r.get("Zona_Morta",""),
            "H2H Partite":      r.get("H2H_n"),
            "H2H Casa%":        r.get("H2H_hw%"),
            "HOME WIN":         r.get("HOME WIN",""),
            "BTTS":             r.get("BTTS",""),
            "OVER 2.5":         r.get("OVER 2.5",""),
            "PARACADUTE":       r.get("PARACADUTE",""),
        })

    df_main = pd.DataFrame(rows_main)
    df_det  = pd.DataFrame(rows_det)

    with pd.ExcelWriter(fname, engine="openpyxl") as writer:
        df_main.to_excel(writer,  sheet_name="Previsioni", index=False)
        df_det.to_excel(writer,   sheet_name="Dettagli",   index=False)
        df_forma.to_excel(writer, sheet_name="Forma",      index=False)

    wb   = load_workbook(fname)
    thin = Border(left=Side(style="thin"), right=Side(style="thin"),
                  top=Side(style="thin"),  bottom=Side(style="thin"))

    # Palette colori
    HDR    = PatternFill("solid", fgColor="1F3864")
    BET    = PatternFill("solid", fgColor="00B050")
    STAR   = PatternFill("solid", fgColor="FFD700")
    ALT    = PatternFill("solid", fgColor="F2F2F2")
    CONF   = PatternFill("solid", fgColor="C6EFCE")
    DIV    = PatternFill("solid", fgColor="FFEB9C")
    NCONF  = PatternFill("solid", fgColor="FFC7CE")
    MOM_P  = PatternFill("solid", fgColor="DEEAF1")
    MOM_N  = PatternFill("solid", fgColor="FCE4D6")
    GRIGIA = PatternFill("solid", fgColor="D9D9D9")
    ARANCIO= PatternFill("solid", fgColor="F4B942")
    DIMMED = PatternFill("solid", fgColor="E8E8E8")
    H2H_C  = PatternFill("solid", fgColor="E2EFDA")

    # ── TAB PREVISIONI ────────────────────────────────────────────────────────
    ws = wb["Previsioni"]

    col_widths = {
        "Data": 12, "Ora (IT)": 8, "Nazione": 12, "Lega": 22,
        "Casa": 22, "Trasferta": 22,
        "Scommessa 1": 18, "Scommessa 2": 18,
        "H%": 7, "X%": 7, "A%": 7,
        "BTTS%": 8, "O2.5%": 8, "DC%": 8, "O1.5%": 8,
        "Q1": 7, "QX": 7, "Q2": 7, "Q_BTTS": 8, "Q_O25": 8, "Q_DC+O15": 10,
        "_BestMkt": 0, "_DimmedMkts": 0, "_HW_Grigia": 0, "_HW_Arancione": 0,
    }
    for col_cells in ws.iter_cols(min_row=1, max_row=1):
        h = str(col_cells[0].value or "")
        cl = get_column_letter(col_cells[0].column)
        w = col_widths.get(h, 12)
        if w == 0:
            ws.column_dimensions[cl].hidden = True
        else:
            ws.column_dimensions[cl].width = w

    for cell in ws[1]:
        cell.fill = HDR; cell.font = Font(bold=True, color="FFFFFF", size=10)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin
    ws.row_dimensions[1].height = 30
    ws.freeze_panes = "A2"

    cols = {cell.value: cell.column for cell in ws[1]}
    c_s1   = cols.get("Scommessa 1")
    c_s2   = cols.get("Scommessa 2")
    c_best = cols.get("_BestMkt")
    c_dim  = cols.get("_DimmedMkts")
    c_gr   = cols.get("_HW_Grigia")
    c_ar   = cols.get("_HW_Arancione")

    MKT_IT = {
        "Vittoria Casa": "HOME WIN",
        "Goal Goal":     "BTTS",
        "Over 2.5":      "OVER 2.5",
        "DC + Over 1.5": "PARACADUTE",
    }

    for i, row in enumerate(ws.iter_rows(min_row=2, max_row=ws.max_row), start=2):
        fill_base = ALT if i % 2 == 0 else None
        for cell in row:
            cell.border = thin
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.font = Font(size=9)
            if fill_base: cell.fill = fill_base

        is_grigia    = bool(row[c_gr  - 1].value) if c_gr  else False
        is_arancione = bool(row[c_ar  - 1].value) if c_ar  else False
        best_mkt     = str(row[c_best - 1].value or "") if c_best else ""
        dimmed_str   = str(row[c_dim  - 1].value or "") if c_dim  else ""

        for c_col, is_best_col in [(c_s1, True), (c_s2, False)]:
            if not c_col: continue
            cell = row[c_col - 1]
            label = str(cell.value or "")
            if not label: continue
            # Risali al mercato raw per sapere se è HW grigia/arancione
            mkt_raw = MKT_IT.get(label.replace("★ ","").strip(), "")
            is_hw   = (mkt_raw == "HOME WIN")
            is_star = "★" in label

            if is_hw and is_arancione:
                cell.fill = ARANCIO; cell.font = Font(bold=True, size=9, color="7B3F00")
            elif is_hw and is_grigia:
                cell.fill = GRIGIA;  cell.font = Font(bold=True, size=9, color="595959")
            elif not is_best_col:
                # Seconda scommessa — dimmed
                cell.fill = DIMMED;  cell.font = Font(size=9, color="888888", italic=True)
            elif is_star:
                cell.fill = STAR;    cell.font = Font(bold=True, size=10, color="000000")
            else:
                cell.fill = BET;     cell.font = Font(bold=True, size=9, color="FFFFFF")
            cell.alignment = Alignment(horizontal="center", vertical="center")

    ws.auto_filter.ref = ws.dimensions

    # ── TAB DETTAGLI ──────────────────────────────────────────────────────────
    ws_det = wb["Dettagli"]
    HDR2   = PatternFill("solid", fgColor="2E4057")
    for cell in ws_det[1]:
        cell.fill = HDR2; cell.font = Font(bold=True, color="FFFFFF", size=9)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin
    ws_det.row_dimensions[1].height = 28
    ws_det.freeze_panes = "A2"

    col_widths_det = {
        "Data": 12, "Ora (IT)": 8, "Nazione": 12, "Lega": 22,
        "Casa": 22, "Trasferta": 22,
        "Gol Attesi": 11, "B365_Conf": 11, "Δ Mercato": 10,
        "Momentum": 10, "Situazione Casa": 20, "Situazione Ospite": 20, "Zona Morta": 10,
        "H2H Partite": 10, "H2H Casa%": 10,
        "HOME WIN": 12, "BTTS": 10, "OVER 2.5": 10, "PARACADUTE": 14,
    }
    cols_det = {cell.value: cell.column for cell in ws_det[1]}
    for col_cells in ws_det.iter_cols(min_row=1, max_row=1):
        h = str(col_cells[0].value or "")
        cl = get_column_letter(col_cells[0].column)
        ws_det.column_dimensions[cl].width = col_widths_det.get(h, 12)

    bc_det  = cols_det.get("B365_Conf")
    mdc_det = cols_det.get("Momentum")
    zmc_det = cols_det.get("Zona Morta")
    h2hc    = cols_det.get("H2H Partite")

    for i, row in enumerate(ws_det.iter_rows(min_row=2, max_row=ws_det.max_row), start=2):
        fill_base = ALT if i % 2 == 0 else None
        for cell in row:
            cell.border = thin
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.font = Font(size=9)
            if fill_base: cell.fill = fill_base
        if bc_det:
            v = str(row[bc_det - 1].value or "")
            if   "CONF" in v: row[bc_det-1].fill = CONF;  row[bc_det-1].font = Font(bold=True, size=9, color="006100")
            elif "DIV"  in v: row[bc_det-1].fill = DIV;   row[bc_det-1].font = Font(bold=True, size=9, color="9C5700")
            elif "SKIP" in v: row[bc_det-1].fill = NCONF; row[bc_det-1].font = Font(bold=True, size=9, color="9C0006")
        if mdc_det:
            try:
                mv = float(row[mdc_det - 1].value or 0)
                if mv > 0.1:    row[mdc_det-1].fill = MOM_P; row[mdc_det-1].font = Font(bold=True, size=9)
                elif mv < -0.1: row[mdc_det-1].fill = MOM_N; row[mdc_det-1].font = Font(bold=True, size=9)
            except: pass
        if zmc_det:
            if str(row[zmc_det - 1].value or "") == "⚠":
                row[zmc_det-1].font = Font(bold=True, size=10, color="FF0000")
        if h2hc:
            try:
                if int(row[h2hc - 1].value or 0) >= H2H_MIN_N:
                    row[h2hc - 1].fill = H2H_C; row[h2hc - 1].font = Font(bold=True, size=9)
            except: pass

    ws_det.auto_filter.ref = ws_det.dimensions

    # ── FOGLIO FORMA ─────────────────────────────────────────────────────────
    ws2  = wb["Forma"]
    HDR2 = PatternFill("solid", fgColor="2E4057")
    CASA = PatternFill("solid", fgColor="DEEAF1")
    AWAY = PatternFill("solid", fgColor="FFF2CC")
    HOT  = PatternFill("solid", fgColor="FF6B6B")
    COLD = PatternFill("solid", fgColor="74B9FF")

    for cell in ws2[1]:
        cell.fill      = HDR2
        cell.font      = Font(bold=True, color="FFFFFF", size=9)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border    = thin
    ws2.row_dimensions[1].height = 28
    ws2.freeze_panes = "A2"

    cols2   = {cell.value: cell.column for cell in ws2[1]}
    rc      = cols2.get("Ruolo")
    f_att_c = cols2.get("F_att")

    for row in ws2.iter_rows(min_row=2, max_row=ws2.max_row):
        fill = CASA if rc and str(row[rc - 1].value) == "Casa" else AWAY
        for cell in row:
            cell.border    = thin
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.font      = Font(size=9)
            cell.fill      = fill
        if f_att_c:
            try:
                v = float(row[f_att_c - 1].value)
                if v >= 1.25:   row[f_att_c-1].fill = HOT;  row[f_att_c-1].font = Font(bold=True, size=9)
                elif v <= 0.75: row[f_att_c-1].fill = COLD; row[f_att_c-1].font = Font(bold=True, size=9)
            except: pass

    for col in ws2.columns:
        header = str(col[0].value or "")
        w = 55 if header == "Ultimi 6" else min(max((len(str(c.value)) for c in col if c.value), default=8) + 2, 22)
        ws2.column_dimensions[get_column_letter(col[0].column)].width = w

    ws2.auto_filter.ref = ws2.dimensions
    wb.save(fname)
    print(f"\n[SALVATO] {fname}")
    return fname

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    args = sys.argv[1:]
    if not args:
        args = ["--all"]   # default: tutte le nazioni

    from_date  = None
    clean_args = []
    for a in args:
        if a.startswith("--date="):
            from_date = a.split("=")[1]
        elif not a.startswith("--"):
            clean_args.append(a)
        else:
            clean_args.append(a)

    nations = ([p.stem for p in Path(CONFIG_DIR).glob("*.json")]
               if clean_args and clean_args[0] == "--all" else
               [a for a in clean_args if not a.startswith("--")])

    start_date, end_date = get_date_window(from_date)
    today_real = datetime.now(TZ_IT).date()
    backtest   = (start_date < today_real)
    mode_label = f"BACKTEST ({start_date})" if backtest else "LIVE"

    print(f"\n{'='*60}")
    print(f"  BETTING PREDICTIONS v5 — {start_date} → {end_date}  [{mode_label}]")
    print(f"  Nazioni: {', '.join(nations)}")
    print(f"{'='*60}")

    all_pred  = []
    all_forma = []

    for nation_code in nations:
        cfg = load_config(nation_code)
        df_p, df_f = build_nation(
            nation_code, cfg, start_date, end_date,
            backtest=backtest, cutoff=from_date
        )
        if not df_p.empty:  all_pred.append(df_p)
        if not df_f.empty:  all_forma.append(df_f)

    if not all_pred:
        print("\nNessuna partita trovata.")
        return

    df_pred  = pd.concat(all_pred,  ignore_index=True).sort_values(["Data", "Ora (IT)"])
    df_forma = pd.concat(all_forma, ignore_index=True)

    print(f"\n{'─'*50}")
    h2h_cov = (df_pred["H2H_n"] >= H2H_MIN_N).sum() if "H2H_n" in df_pred.columns else 0
    print(f"  H2H disponibile:   {h2h_cov}/{len(df_pred)} partite")
    if "B365_Conf" in df_pred.columns:
        for label in ["✓ CONF", "△ DIV", "✗ SKIP", "—"]:
            n = (df_pred["B365_Conf"] == label).sum()
            if n > 0:
                print(f"  Bet365 {label}:  {n} partite")

    print()
    for market in ["HOME WIN", "BTTS", "OVER 2.5", "PARACADUTE"]:
        col = df_pred.get(market, pd.Series(dtype=str))
        n   = col.notna().sum() - (col == "").sum()
        if n > 0:
            stars = (col.str.contains("★", na=False)).sum()
            print(f"  {market:<22} {n:>4} segnali  (★ {stars})")

    save_excel(df_pred, df_forma, start_date, end_date)
    print(f"\n  Partite totali:   {len(df_pred)}")
    con_bet = df_pred[["HOME WIN","BTTS","OVER 2.5","PARACADUTE"]].notna().any(axis=1).sum()
    print(f"  Con almeno 1 segnale: {con_bet}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()