"""
hwe — Home Win Ensemble.

Probabilità di vittoria in casa nel calcio, combinando tre modelli:
regressione logistica, HMM sugli stati di forma, albero decisionale.

    from hwe.data import load_matches, split_played
    from hwe.features import build
    from hwe.ensemble import HomeWinEnsemble

    matches = load_matches("partite.csv")
    played, future = split_played(matches)
    ens = HomeWinEnsemble().fit(played)
    ens.set_history(played)
    p = ens.predict_proba(build(matches).query("played == 0"))
"""

__version__ = "1.0.0"
__all__ = ["data", "schema", "features", "hmm", "base_models", "ensemble",
           "evaluation", "synthetic", "util"]
