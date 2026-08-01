"""
features.py — matrice di feature per i modelli.

Unisce:
  - le feature di rating (ratings.py), sempre pre-match
  - feature derivate (differenze, log, interazioni) che aiutano i modelli lineari
  - le feature di mercato (quote de-viggate), tenute separate e disattivate
    di default nei modelli base

Perché il mercato è escluso di default: se i modelli vedono le quote, imparano
a replicarle. Le previsioni migliorano nelle metriche, ma l'edge sparisce e il
Closing Line Value tende a zero. Il mercato entra invece nel confronto finale
(filtro value) e, opzionalmente, nel meta-modello di stacking.
"""

import numpy as np
import pandas as pd

from .config import EnsembleConfig
from .ratings import RATING_FEATURES, RatingEngine
from .value import devig, devig_two_way, margin_from_1x2, two_way_margin_from_3way

DERIVED_FEATURES = [
    "elo_diff_norm", "gap_log_home", "gap_log_away", "gap_ratio",
    "pi_sum", "pi_diff", "form_gf_diff", "form_ga_diff",
    "exp_total_vs_league", "supr_x_total", "reliability",
]

MARKET_FEATURES = [
    "mkt_p_h", "mkt_p_d", "mkt_p_a", "mkt_p_o25", "mkt_p_btts",
    "mkt_margin", "mkt_logit_h", "mkt_supremacy",
]


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def build_derived(rat):
    """Feature derivate dai rating: rapporti e differenze che i modelli lineari non ricavano da soli."""
    out = pd.DataFrame(index=rat.index)
    out["elo_diff_norm"] = rat["elo_diff"] / 400.0
    out["gap_log_home"]  = np.log(np.maximum(rat["gap_exp_home_goals"], 0.05))
    out["gap_log_away"]  = np.log(np.maximum(rat["gap_exp_away_goals"], 0.05))
    out["gap_ratio"]     = (rat["gap_exp_home_goals"] /
                            np.maximum(rat["gap_exp_away_goals"], 0.05))
    out["pi_sum"]        = rat["pi_home_h"] + rat["pi_away_a"]
    out["pi_diff"]       = rat["pi_home_h"] - rat["pi_away_a"]
    out["form_gf_diff"]  = rat["form_gf_home"] - rat["form_gf_away"]
    out["form_ga_diff"]  = rat["form_ga_home"] - rat["form_ga_away"]
    out["exp_total_vs_league"] = (rat["gap_exp_total"] -
                                  rat["league_avg_goals"])
    out["supr_x_total"]  = rat["gap_exp_supr"] * rat["gap_exp_total"]
    # quanto sono affidabili i rating delle due squadre (partite osservate)
    out["reliability"]   = np.minimum(rat["n_matches_home"], rat["n_matches_away"]).clip(0, 60) / 60.0
    return out[DERIVED_FEATURES]


def build_market(df, method="shin"):
    """Probabilità de-viggate dei mercati disponibili nelle quote raccolte."""
    out = pd.DataFrame(index=df.index, columns=MARKET_FEATURES, dtype=float)

    q1 = pd.to_numeric(df.get("q1"), errors="coerce")
    qx = pd.to_numeric(df.get("qx"), errors="coerce")
    q2 = pd.to_numeric(df.get("q2"), errors="coerce")
    o25 = pd.to_numeric(df.get("odd_o25"), errors="coerce")
    btts = pd.to_numeric(df.get("odd_btts"), errors="coerce")

    has_1x2 = q1.notna() & qx.notna() & q2.notna() & (q1 > 1) & (qx > 1) & (q2 > 1)
    for idx in df.index[has_1x2]:
        p = devig([q1[idx], qx[idx], q2[idx]], method=method)
        out.loc[idx, ["mkt_p_h", "mkt_p_d", "mkt_p_a"]] = p
        out.loc[idx, "mkt_margin"] = margin_from_1x2(q1[idx], qx[idx], q2[idx])

    margin2 = out["mkt_margin"].map(two_way_margin_from_3way)
    for idx in df.index[o25.notna()]:
        out.loc[idx, "mkt_p_o25"] = devig_two_way(
            o25[idx], assumed_margin=margin2.get(idx, 0.045))
    for idx in df.index[btts.notna()]:
        out.loc[idx, "mkt_p_btts"] = devig_two_way(
            btts[idx], assumed_margin=margin2.get(idx, 0.045))

    out["mkt_logit_h"]   = _logit(out["mkt_p_h"])
    out["mkt_supremacy"] = _logit(out["mkt_p_h"]) - _logit(out["mkt_p_a"])
    return out


def build_features(df, cfg: EnsembleConfig = None, engine: RatingEngine = None,
                   with_market=True, update_ratings=True):
    """
    Costruisce la matrice completa di feature per `df` (ordinato cronologicamente).

    Restituisce (features, engine). L'engine viene restituito perché mantiene
    lo stato dei rating: lo stesso oggetto serve poi per le fixture future.
    """
    cfg = cfg or EnsembleConfig()
    engine = engine or RatingEngine(cfg.ratings)

    rat = engine.transform(df, update=update_ratings)
    parts = [rat, build_derived(rat)]
    if with_market:
        parts.append(build_market(df, method=cfg.betting.devig_method))

    feats = pd.concat(parts, axis=1)
    return feats, engine


def feature_columns(with_market=False):
    cols = list(RATING_FEATURES) + list(DERIVED_FEATURES)
    if with_market:
        cols += list(MARKET_FEATURES)
    return cols


def prepare_matrix(feats, columns=None, with_market=False):
    """
    Estrae la matrice numerica per i modelli.

    I NaN restano NaN per i backend che li gestiscono nativamente (LightGBM,
    XGBoost, HistGradientBoosting); i modelli lineari li imputano a parte.
    """
    columns = columns or feature_columns(with_market=with_market)
    missing = [c for c in columns if c not in feats.columns]
    if missing:
        for c in missing:
            feats[c] = np.nan
    X = feats[columns].astype(float)
    return X.replace([np.inf, -np.inf], np.nan), columns
