"""
baseline.py — riferimento Dixon-Coles classico.

Serve a rispondere alla domanda che sta dietro tutto il progetto: l'ensemble
guadagna davvero qualcosa rispetto al modello di conteggio tradizionale?

È lo stesso Poisson gerarchico di M2 ma senza covariate di rating e con
shrinkage minimo: attacco e difesa stimati solo dai gol passati, vantaggio
casa per lega, correzione Dixon-Coles sui punteggi bassi, decadimento
temporale. Cioè il modello che la pipeline attuale implementa a mano.
"""

from dataclasses import replace

from .m2_bayes import BayesianHybridModel
from ..config import ModelConfig


class DixonColesBaseline(BayesianHybridModel):
    """Dixon-Coles con decadimento temporale, senza rating dinamici."""

    name = "M0_dixon_coles"

    def __init__(self, cfg: ModelConfig = None):
        # copia della config: il baseline non deve alterare quella di M2
        base = replace(cfg or ModelConfig())
        base.m2_alpha = min(base.m2_alpha, 0.02)
        super().__init__(base)
        self.covariates_ = []          # nessun ponte verso le squadre nuove
