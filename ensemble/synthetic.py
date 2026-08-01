"""
synthetic.py — generatore di campionati finti.

Serve a due cose concrete:

  1. far girare test e demo senza toccare i dati reali (che non sono nel repo);
  2. avere un caso in cui la verità è nota: le forze delle squadre sono
     parametri del generatore, quindi si può verificare che i rating le
     recuperino e che le probabilità del modello siano calibrate.

Il generatore include quello che rende il problema realistico:
  - vantaggio casa per lega
  - forze che derivano lentamente nel tempo (mercato, allenatori, infortuni)
  - promozioni e retrocessioni fra tier, quindi squadre "nuove" ogni stagione
  - quote di bookmaker costruite dalle probabilità vere con margine, rumore
    e favourite-longshot bias, più una quota di chiusura più accurata
"""

import numpy as np
import pandas as pd

from .models.base import poisson_score_matrix, score_matrix_to_markets


def _team_names(n, prefix):
    return [f"{prefix} {i + 1:02d}" for i in range(n)]


def generate_league(n_seasons=4, n_teams=18, n_tiers=2, start_date="2021-08-01",
                    seed=42, home_advantage=0.25, margin=0.06,
                    longshot_bias=0.03, nation="Sinteland"):
    """
    Genera uno storico completo con risultati e quote.

    Restituisce un DataFrame nello schema di data.load_dataset (già con
    kickoff, target e quote), pronto per features.build_features.
    """
    rng = np.random.default_rng(seed)

    # forze latenti: attacco e difesa per squadra, per tier
    teams = {}
    tid = 1000
    for tier in range(1, n_tiers + 1):
        for name in _team_names(n_teams, f"T{tier}"):
            teams[tid] = {
                "name": f"{name}",
                "tier": tier,
                "att": rng.normal(0.15 - 0.25 * (tier - 1), 0.28),
                "def": rng.normal(-0.15 + 0.25 * (tier - 1), 0.25),
            }
            tid += 1

    base_goals = {1: 1.20, 2: 1.12, 3: 1.08}
    rows = []
    fixture_id = 1
    date = pd.Timestamp(start_date)

    for season in range(2021, 2021 + n_seasons):
        for tier in range(1, n_tiers + 1):
            squad = [t for t, v in teams.items() if v["tier"] == tier]
            rng.shuffle(squad)
            # doppio girone all'italiana
            pairs = [(h, a) for h in squad for a in squad if h != a]
            rng.shuffle(pairs)

            season_start = date
            for k, (h, a) in enumerate(pairs):
                kickoff = season_start + pd.Timedelta(days=int(k / max(len(squad) // 2, 1)) * 3.5)
                mu = np.log(base_goals.get(tier, 1.2))
                lam_h = np.exp(mu + teams[h]["att"] - teams[a]["def"] + home_advantage)
                lam_a = np.exp(mu + teams[a]["att"] - teams[h]["def"])
                lam_h = float(np.clip(lam_h, 0.15, 6.0))
                lam_a = float(np.clip(lam_a, 0.15, 6.0))

                hg = int(rng.poisson(lam_h))
                ag = int(rng.poisson(lam_a))

                # probabilità vere: servono a costruire quote coerenti
                mk = score_matrix_to_markets(poisson_score_matrix(lam_h, lam_a, 10, rho=-0.05))
                rows.append({
                    "fixture_id": fixture_id,
                    "league_id": tier,
                    "league_name": f"Lega T{tier}",
                    "tier": tier,
                    "season": season,
                    "round": f"R{k // max(len(squad) // 2, 1) + 1}",
                    "date": kickoff.normalize(),
                    "time": "15:00",
                    "played": 1,
                    "home_id": h, "home_name": teams[h]["name"],
                    "away_id": a, "away_name": teams[a]["name"],
                    "home_goals": hg, "away_goals": ag,
                    "_p_h": mk["1x2"][0], "_p_d": mk["1x2"][1], "_p_a": mk["1x2"][2],
                    "_p_o25": mk["over25"], "_p_btts": mk["btts"],
                })
                fixture_id += 1

            date = season_start + pd.Timedelta(days=300)

        # deriva delle forze + promozioni/retrocessioni
        for v in teams.values():
            v["att"] += rng.normal(0, 0.10)
            v["def"] += rng.normal(0, 0.10)
        if n_tiers > 1:
            _promote_relegate(teams, rows, season, n_promoted=2)
        date = date + pd.Timedelta(days=60)

    df = pd.DataFrame(rows)
    df = _add_odds(df, rng, margin=margin, longshot_bias=longshot_bias)
    df["nation"] = nation
    df["nation_code"] = nation.lower()
    df["league_key"] = df["nation_code"] + "|" + df["league_id"].astype(str)
    df["kickoff"] = pd.to_datetime(df["date"].dt.strftime("%Y-%m-%d") + " " + df["time"])
    df["home_xg"] = np.nan
    df["away_xg"] = np.nan

    from .data import add_targets
    df = add_targets(df)
    return df.sort_values(["kickoff", "fixture_id"]).reset_index(drop=True)


def _promote_relegate(teams, rows, season, n_promoted=2):
    """Le prime del tier inferiore salgono, le ultime del superiore scendono."""
    df = pd.DataFrame([r for r in rows if r["season"] == season])
    if df.empty:
        return
    pts = {}
    for r in df.itertuples(index=False):
        gd = r.home_goals - r.away_goals
        pts[r.home_id] = pts.get(r.home_id, 0) + (3 if gd > 0 else 1 if gd == 0 else 0)
        pts[r.away_id] = pts.get(r.away_id, 0) + (3 if gd < 0 else 1 if gd == 0 else 0)

    tiers = sorted({v["tier"] for v in teams.values()})
    for upper, lower in zip(tiers, tiers[1:]):
        up = [t for t, v in teams.items() if v["tier"] == upper]
        low = [t for t, v in teams.items() if v["tier"] == lower]
        if len(up) < n_promoted or len(low) < n_promoted:
            continue
        worst = sorted(up, key=lambda t: pts.get(t, 0))[:n_promoted]
        best = sorted(low, key=lambda t: pts.get(t, 0), reverse=True)[:n_promoted]
        for t in worst:
            teams[t]["tier"] = lower
        for t in best:
            teams[t]["tier"] = upper


def _add_odds(df, rng, margin=0.06, longshot_bias=0.03):
    """
    Quote del bookmaker: probabilità vere + rumore + margine + longshot bias.

    Le quote di chiusura (close_*) sono più vicine alla verità di quelle di
    apertura: è quello che rende misurabile il Closing Line Value.
    """
    out = df.copy()

    def build(p_cols, noise, bias_strength):
        P = out[p_cols].to_numpy(dtype=float)
        noisy = np.clip(P * np.exp(rng.normal(0, noise, P.shape)), 1e-4, None)
        noisy = noisy / noisy.sum(axis=1, keepdims=True)
        # longshot bias: gli esiti improbabili vengono sovrastimati dal book
        adj = noisy + bias_strength * (noisy < 0.25) * noisy
        adj = adj / adj.sum(axis=1, keepdims=True)
        return 1.0 / (adj * (1.0 + margin))

    q = build(["_p_h", "_p_d", "_p_a"], 0.16, longshot_bias)
    out["q1"], out["qx"], out["q2"] = q[:, 0], q[:, 1], q[:, 2]

    qc = build(["_p_h", "_p_d", "_p_a"], 0.05, longshot_bias * 0.5)
    out["close_q1"], out["close_qx"], out["close_q2"] = qc[:, 0], qc[:, 1], qc[:, 2]

    for col, oddcol, noise in [("_p_o25", "odd_o25", 0.10), ("_p_btts", "odd_btts", 0.10)]:
        p = out[col].to_numpy(dtype=float)
        pn = np.clip(p * np.exp(rng.normal(0, noise, len(p))), 1e-3, 1 - 1e-3)
        out[oddcol] = 1.0 / (pn * (1.0 + margin * 2 / 3))
        pc = np.clip(p * np.exp(rng.normal(0, 0.03, len(p))), 1e-3, 1 - 1e-3)
        out[f"close_{oddcol}"] = 1.0 / (pc * (1.0 + margin * 2 / 3))

    out["odd_o15"] = np.nan
    out["odd_1x"] = np.nan
    out["overround"] = 1.0 / out["q1"] + 1.0 / out["qx"] + 1.0 / out["q2"]
    return out


def split_future(df, n_future=40):
    """Trasforma le ultime n partite in 'fixture future' (goals ignoti)."""
    df = df.copy().sort_values("kickoff").reset_index(drop=True)
    idx = df.index[-n_future:]
    truth = df.loc[idx].copy()
    df.loc[idx, ["home_goals", "away_goals", "y_1x2", "y_over25", "y_btts",
                 "total_goals"]] = np.nan
    df.loc[idx, "played"] = 0
    return df, truth
