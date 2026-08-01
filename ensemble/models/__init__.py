"""Modelli base dell'ensemble: M1 (boosting), M2 (bayesiano ibrido), M3 (GAP diretto)."""

from .base import BaseModel, poisson_score_matrix, score_matrix_to_markets
from .m1_gbm import GBMModel
from .m2_bayes import BayesianHybridModel
from .m3_gap import GapDirectModel


def default_models(cfg, seed=42):
    """I tre modelli base nella configurazione raccomandata."""
    return [
        GBMModel(cfg.models, seed=seed),
        BayesianHybridModel(cfg.models),
        GapDirectModel(cfg.models, seed=seed),
    ]


__all__ = [
    "BaseModel", "GBMModel", "BayesianHybridModel", "GapDirectModel",
    "default_models", "poisson_score_matrix", "score_matrix_to_markets",
]
