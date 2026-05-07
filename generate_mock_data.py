"""
generate_mock_data.py — Genera dati sintetici realistici per la Premier League.

Crea:
  /home/user/data/england/processed/all_results.csv   (stagioni 2021-2024)
  /home/user/data/england/processed/all_fixtures.csv  (fixture future 2025)

Usa un modello Poisson con rating offesa/difesa per squadra e vantaggio casa.
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path
import itertools

np.random.seed(42)

# ── SQUADRE PREMIER LEAGUE ──────────────────────────────────────────────────
# (id, nome, rating_att, rating_def)
# rating_def = gol concessi in media: basso = difesa forte
TEAMS = [
    (50,   "Man City",         2.10, 0.72),
    (42,   "Arsenal",          1.90, 0.80),
    (40,   "Liverpool",        1.95, 0.88),
    (49,   "Chelsea",          1.60, 1.05),
    (47,   "Tottenham",        1.65, 1.10),
    (33,   "Man United",       1.55, 1.20),
    (34,   "Newcastle",        1.45, 1.00),
    (66,   "Aston Villa",      1.50, 1.05),
    (51,   "Brighton",         1.40, 0.98),
    (48,   "West Ham",         1.35, 1.18),
    (55,   "Brentford",        1.25, 1.12),
    (36,   "Fulham",           1.20, 1.15),
    (52,   "Crystal Palace",   1.15, 1.18),
    (39,   "Wolves",           1.18, 1.22),
    (65,   "Nott'm Forest",    1.12, 1.20),
    (45,   "Everton",          1.05, 1.35),
    (35,   "Bournemouth",      1.15, 1.30),
    (44,   "Burnley",          0.95, 1.48),
    (62,   "Sheffield Utd",    0.90, 1.52),
    (1359, "Luton",            0.88, 1.55),
]

HOME_ADV  = 0.28   # moltiplicatore vantaggio casa
LEAGUE_ID = 39
LEAGUE    = "Premier League"
TIER      = 1

OUT_DIR = Path("/home/user/data/england/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def simulate_match(h_att, h_def, a_att, a_def):
    avg = 2.70
    lm  = avg / 2.0
    lh  = (h_att * a_def / lm) * (1 + HOME_ADV)
    la  = (a_att * h_def / lm)
    lh  = max(0.3, lh)
    la  = max(0.3, la)

    hg = int(np.random.poisson(lh))
    ag = int(np.random.poisson(la))

    # Primo tempo: ~42% dei gol
    hht = int(np.random.binomial(hg, 0.42)) if hg > 0 else 0
    aht = int(np.random.binomial(ag, 0.42)) if ag > 0 else 0
    h2h = hg - hht
    a2h = ag - aht
    return hg, ag, hht, aht, h2h, a2h


def build_season_fixtures(season, season_start, season_end):
    """Genera le 380 partite di una stagione in ordine temporale."""
    # Round-robin: ogni coppia (home, away) gioca una volta
    pairs = [(h, a) for h, a in itertools.permutations(TEAMS, 2)]
    np.random.shuffle(pairs)

    # Distribuisce uniformemente nell'arco della stagione
    total_days = (season_end - season_start).days
    step = total_days / len(pairs)

    rows = []
    for i, (home, away) in enumerate(pairs):
        match_date = season_start + timedelta(days=int(i * step))
        # Giornate di solito Sabato/Domenica
        match_date += timedelta(days=(5 - match_date.weekday()) % 7)
        if match_date > season_end:
            match_date = season_end - timedelta(days=1)

        rows.append((match_date, home, away))

    return sorted(rows, key=lambda x: x[0])


def create_season_rows(season, fixtures, fid_start, played_until=None):
    rows = []
    fid  = fid_start
    for match_date, (hid, hname, h_att, h_def), (aid, aname, a_att, a_def) in fixtures:
        played = played_until is None or match_date <= played_until

        if played:
            hg, ag, hht, aht, h2h, a2h = simulate_match(h_att, h_def, a_att, a_def)
            total  = hg + ag
            result = "H" if hg > ag else ("A" if hg < ag else "D")
            status = "FT"
        else:
            hg = ag = hht = aht = h2h = a2h = total = result = None
            status = "NS"

        rows.append({
            "fixture_id":  fid,
            "league_id":   LEAGUE_ID,
            "league_name": LEAGUE,
            "tier":        TIER,
            "season":      season,
            "round":       f"R{(fid - fid_start) // 10 + 1}",
            "date":        str(match_date),
            "time":        "15:00",
            "status":      status,
            "played":      1 if played else 0,
            "home_id":     hid,
            "home_name":   hname,
            "away_id":     aid,
            "away_name":   aname,
            "home_goals":  hg,
            "away_goals":  ag,
            "home_ht":     hht,
            "away_ht":     aht,
            "home_2h":     h2h,
            "away_2h":     a2h,
            "total_goals": total,
            "result_1x2":  result,
            "over15":  (1 if total > 1 else 0) if played else None,
            "over25":  (1 if total > 2 else 0) if played else None,
            "over35":  (1 if total > 3 else 0) if played else None,
            "btts":    (1 if (hg or 0) > 0 and (ag or 0) > 0 else 0) if played else None,
        })
        fid += 1

    return rows, fid


# ── GENERA STAGIONI ──────────────────────────────────────────────────────────
all_results  = []
all_fixtures = []
fid = 1_000_000

SEASONS = [
    # (season, start, end, played_until)
    (2021, date(2021, 8, 14), date(2022, 5, 22), None),          # completa
    (2022, date(2022, 8,  6), date(2023, 5, 28), None),          # completa
    (2023, date(2023, 8, 12), date(2024, 5, 19), None),          # completa
    (2024, date(2024, 8, 17), date(2025, 5, 18), date(2025, 3, 30)),  # parziale
]

for season, start, end, played_until in SEASONS:
    fixtures = build_season_fixtures(season, start, end)
    rows, fid = create_season_rows(season, fixtures, fid, played_until)

    played = [r for r in rows if r["played"] == 1]
    future = [r for r in rows if r["played"] == 0]

    all_results.extend(played)
    if future:
        all_fixtures.extend(future)

# ── SALVA ────────────────────────────────────────────────────────────────────
df_res = pd.DataFrame(all_results)
df_fix = pd.DataFrame(all_fixtures)

df_res.to_csv(OUT_DIR / "all_results.csv",  index=False)
df_fix.to_csv(OUT_DIR / "all_fixtures.csv", index=False)

# ── STAMPA STATISTICHE ───────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Dati generati per England / Premier League")
print(f"{'='*55}")
print(f"  Risultati storici : {len(df_res):,} partite")
print(f"  Fixture future    : {len(df_fix):,} partite")
print(f"  Stagioni          : {df_res['season'].unique().tolist()}")
print()

total = len(df_res)
hw = (df_res["result_1x2"] == "H").sum()
dr = (df_res["result_1x2"] == "D").sum()
aw = (df_res["result_1x2"] == "A").sum()
o25 = df_res["over25"].mean()
btts = df_res["btts"].mean()
avg_gol = df_res["total_goals"].mean()

print(f"  H%={hw/total*100:.1f}%  D%={dr/total*100:.1f}%  A%={aw/total*100:.1f}%")
print(f"  Avg gol={avg_gol:.2f}  O2.5%={o25*100:.1f}%  BTTS%={btts*100:.1f}%")
print(f"{'='*55}")
print(f"  [SALVATO] {OUT_DIR}/all_results.csv")
print(f"  [SALVATO] {OUT_DIR}/all_fixtures.csv")
