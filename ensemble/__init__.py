"""
ensemble/ — Ensemble Home Win: Logistica + HMM + Albero decisionale.

Moduli:
    paths.py     percorsi progetto e scoperta nazioni
    features.py  feature engineering strettamente causale (solo passato)
    hmm.py       HMM discreto a stati di forma (Baum-Welch, emissioni per venue)
    models.py    i tre modelli base + combinazione pesata calibrata
    evaluate.py  metriche, walk-forward backtest, valutazione EV vs quote
"""

__all__ = ["paths", "features", "hmm", "models", "evaluate"]
