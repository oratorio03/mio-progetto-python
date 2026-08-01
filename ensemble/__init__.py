"""
ensemble — alternativa a Poisson/Dixon-Coles basata su tre modelli combinati.

    M1  Gradient Boosting su pi-ratings, Elo, GAP/xG e forma
    M2  Modello bayesiano ibrido (Poisson gerarchico + rating dinamici)
    M3  Rating GAP/xG + modello diretto calibrato (logistica + isotonica)

I tre modelli vengono combinati con stacking per-mercato walk-forward,
calibrati con regressione isotonica e trasformati in giocate tramite
filtro value su quota de-viggata e staking a Kelly frazionario.

Entry point da riga di comando: 9_ensemble.py (nella root del progetto).
"""

from .config import EnsembleConfig, MARKETS, OUTCOME_1X2

__all__ = ["EnsembleConfig", "MARKETS", "OUTCOME_1X2"]
__version__ = "1.0.0"
