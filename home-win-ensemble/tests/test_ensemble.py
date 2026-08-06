"""Ensemble: calibrazione, pesi, qualità fuori campione, salvataggio."""

import pickle

import numpy as np
import pytest

from hwe import evaluation as E
from hwe import features as F
from hwe.ensemble import MODELS, HomeWinEnsemble
from hwe.util import PlattCalibrator, log_loss, optimise_weights


# ── mattoni ───────────────────────────────────────────────────────────────────

def test_calibratore_corregge_la_sicurezza_eccessiva():
    """Probabilità troppo estreme devono essere schiacciate verso il centro."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000).astype(float)
    # p vera 0.35/0.65, ma il modello dichiara 0.10/0.90
    vera = np.where(y == 1, 0.65, 0.35)
    dichiarata = np.where(rng.random(2000) < vera, 0.90, 0.10)

    calibratore = PlattCalibrator().fit(dichiarata, y)
    assert calibratore.a < 1.0, "pendenza < 1 = modello sovra-sicuro corretto"
    assert log_loss(y, calibratore.transform(dichiarata)) < log_loss(y, dichiarata)


def test_pesi_premiano_il_modello_informativo():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 800).astype(float)
    buono = np.clip(y * 0.8 + 0.1 + rng.normal(0, 0.05, 800), 0.01, 0.99)
    inutile = np.full(800, 0.5)

    pesi = optimise_weights(np.column_stack([buono, inutile]), y, shrink=0.0)
    assert pesi[0] > 0.9
    assert np.isclose(pesi.sum(), 1.0)


def test_shrinkage_tiene_tutti_in_gioco():
    """Con lo shrink nessun modello viene azzerato del tutto."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 500).astype(float)
    buono = np.clip(y * 0.8 + 0.1, 0.01, 0.99)
    inutile = np.full(500, 0.5)

    pesi = optimise_weights(np.column_stack([buono, inutile]), y, shrink=0.15)
    assert pesi.min() > 0.05
    assert np.isclose(pesi.sum(), 1.0)


# ── ensemble addestrato ───────────────────────────────────────────────────────

def test_pesi_validi(trained):
    assert len(trained.weights) == len(MODELS)
    assert np.isclose(trained.weights.sum(), 1.0)
    assert (trained.weights >= 0).all()


def test_calibrazione_stimata_fuori_campione(trained):
    assert trained.n_oof > 200, "servono abbastanza predizioni out-of-fold"
    assert trained.n_oof < trained.n_train, "l'OOF è una parte del training"
    for nome in MODELS:
        assert nome in trained.calibrators
        assert trained.oof_scores[nome]["log_loss_cal"] <= \
               trained.oof_scores[nome]["log_loss"] + 1e-6, \
            f"la calibrazione ha peggiorato {nome}"


def test_batte_la_costante_fuori_campione(holdout):
    """Su dati mai visti deve fare meglio del prevedere sempre la media."""
    _, test, detail = holdout
    y = test["target"].to_numpy(dtype=float)

    perdita = log_loss(y, detail["p_ensemble"])
    costante = log_loss(y, np.full(len(y), float(y.mean())))
    assert perdita < costante, f"ensemble {perdita:.4f} vs costante {costante:.4f}"


def test_non_peggio_del_peggiore(holdout):
    """Fondere non deve fare danni: il risultato sta dentro i modelli base."""
    _, test, detail = holdout
    y = test["target"].to_numpy(dtype=float)

    perdite = {nome: log_loss(y, detail[f"p_{nome}"]) for nome in MODELS}
    assert log_loss(y, detail["p_ensemble"]) <= max(perdite.values()), perdite


def test_ordina_le_partite(holdout):
    _, test, detail = holdout
    y = test["target"].to_numpy(dtype=float)
    assert E.auc(y, detail["p_ensemble"]) > 0.55


def test_probabilita_sensate(trained, table):
    righe = table[table["played"] == 0]
    p = trained.predict_proba(righe)

    assert np.isfinite(p).all()
    assert ((p > 0) & (p < 1)).all()
    assert p.std() > 0.02, "il modello non distingue le partite fra loro"
    assert 0.2 < p.mean() < 0.7, f"probabilità media implausibile: {p.mean():.2f}"


def test_dettaglio_completo(trained, table):
    righe = table[table["played"] == 0]
    detail = trained.predict_proba(righe, detail=True)

    for nome in MODELS:
        assert f"p_{nome}" in detail.columns
        assert f"p_{nome}_grezza" in detail.columns
    assert len(detail) == len(righe)
    assert (detail["disaccordo"] >= 0).all()

    colonne = [f"p_{nome}" for nome in MODELS]
    atteso = detail[colonne].max(axis=1) - detail[colonne].min(axis=1)
    assert np.allclose(detail["disaccordo"], atteso)


def test_serve_un_minimo_di_dati(played):
    with pytest.raises(ValueError, match="almeno 100 partite"):
        HomeWinEnsemble(verbose=False).fit(played.iloc[:50])


def test_il_modello_salvato_resta_leggero(trained, tmp_path):
    """Le sequenze si ricostruiscono dal CSV: non devono finire nel pickle."""
    percorso = tmp_path / "modello.pkl"
    percorso.write_bytes(pickle.dumps(trained))
    ricaricato = pickle.loads(percorso.read_bytes())

    assert not ricaricato.hmm.sequences
    assert ricaricato.weights is not None
    assert ricaricato.n_train == trained.n_train


def test_predizioni_identiche_dopo_ricarica(trained, table, played, tmp_path):
    righe = table[table["played"] == 0]
    prima = trained.predict_proba(righe)

    ricaricato = pickle.loads(pickle.dumps(trained))
    ricaricato.set_history(played)
    dopo = ricaricato.predict_proba(righe)

    assert np.allclose(prima, dopo)


def test_riepilogo_leggibile(trained):
    testo = trained.summary()
    for nome in MODELS:
        assert nome in testo
    assert "out-of-fold" in testo
