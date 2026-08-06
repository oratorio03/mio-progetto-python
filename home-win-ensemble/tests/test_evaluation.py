"""Metriche, backtest walk-forward, valutazione economica."""

import numpy as np
import pandas as pd
import pytest

from hwe import evaluation as E
from hwe.util import log_loss


# ── metriche ──────────────────────────────────────────────────────────────────

def test_auc_casi_noti():
    y = np.array([0, 0, 1, 1])
    assert E.auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0    # ordine perfetto
    assert E.auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0    # ordine invertito
    assert E.auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5    # tutti pari merito
    assert np.isnan(E.auc(np.array([1, 1]), np.array([0.3, 0.7])))


def test_log_loss_premia_chi_ha_ragione():
    y = np.array([1.0, 1.0, 0.0, 0.0])
    sicuro_e_giusto = np.array([0.9, 0.9, 0.1, 0.1])
    sicuro_e_sbagliato = np.array([0.1, 0.1, 0.9, 0.9])
    assert log_loss(y, sicuro_e_giusto) < log_loss(y, np.full(4, 0.5))
    assert log_loss(y, sicuro_e_sbagliato) > log_loss(y, np.full(4, 0.5))


def test_tabella_soglie():
    y = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
    tabella = E.threshold_table(y, p, soglie=(0.5, 0.7))

    riga = tabella[tabella["soglia"] == 0.7].iloc[0]
    assert riga["n"] == 3                       # 0.9, 0.8, 0.7
    assert riga["riuscite_%"] == 100.0
    riga = tabella[tabella["soglia"] == 0.5].iloc[0]
    assert riga["n"] == 4
    assert riga["riuscite_%"] == 75.0


def test_tabella_calibrazione_misura_lo_scarto():
    """Un modello che dice 0.8 e ne azzecca il 50% deve mostrare scarto -0.3."""
    p = np.full(100, 0.8)
    y = np.array([1] * 50 + [0] * 50, dtype=float)
    tabella = E.calibration_table(y, p)
    riga = tabella.iloc[0]
    assert riga["n"] == 100
    assert np.isclose(riga["scarto"], -0.3, atol=0.01)


def test_metriche_complete():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300).astype(float)
    p = np.clip(y * 0.4 + 0.3 + rng.normal(0, 0.1, 300), 0.02, 0.98)
    m = E.metrics(y, p, "prova")

    assert m["modello"] == "prova"
    assert m["n"] == 300
    assert 0 < m["log_loss"] < 1
    assert m["skill"] > 0                       # meglio della costante
    assert 0.5 < m["auc"] <= 1.0


# ── valutazione economica ─────────────────────────────────────────────────────

def test_value_bet_conta_bene_il_profitto():
    preds = pd.DataFrame({
        "target":     [1.0, 0.0, 1.0],
        "p_ensemble": [0.60, 0.60, 0.60],
        "odds_home":  [2.00, 2.00, 1.20],       # la terza non ha edge
    })
    riepilogo, giocate = E.value_bets(preds, min_edge=0.05)

    assert riepilogo["n_giocate"] == 2          # solo le prime due
    assert np.isclose(riepilogo["profitto"], 0.0)   # +1.00 e -1.00
    assert riepilogo["riuscite_%"] == 50.0


def test_value_bet_senza_quote():
    preds = pd.DataFrame({"target": [1.0], "p_ensemble": [0.6],
                          "odds_home": [np.nan]})
    riepilogo, _ = E.value_bets(preds)
    assert riepilogo["n_giocate"] == 0


# ── walk-forward ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def backtest(played):
    return E.walk_forward(played, min_train=500, step_days=60, verbose=False)


def test_backtest_produce_predizioni(backtest):
    assert not backtest.empty
    assert len(backtest) > 300
    assert backtest["match_id"].is_unique, "una partita prevista due volte"
    assert backtest["p_ensemble"].between(0, 1).all()
    assert backtest["target"].notna().all()


def test_backtest_prevede_sempre_in_avanti(backtest):
    """Ogni fold deve prevedere partite successive all'inizio del fold."""
    for inizio, gruppo in backtest.groupby("fold"):
        assert (gruppo["date"] >= inizio).all()


def test_batte_il_prior_noto_al_training(backtest):
    """
    Il confronto onesto: la frequenza di vittorie casalinghe che si conosceva
    al momento di addestrare. Non la media del campione di test, che è un
    oracolo.
    """
    y = backtest["target"]
    ensemble = log_loss(y, backtest["p_ensemble"])
    prior = log_loss(y, backtest["p_prior"])
    assert ensemble < prior, f"ensemble {ensemble:.4f} vs prior {prior:.4f}"
    assert E.auc(y, backtest["p_ensemble"]) > 0.55


def test_probabilita_non_sovra_sicure(backtest):
    """
    Ricalibrando le predizioni del backtest la pendenza deve essere vicina a 1:
    se fosse molto sotto, il modello starebbe sparando numeri troppo estremi.
    """
    from hwe.util import PlattCalibrator
    pendenza = PlattCalibrator().fit(backtest["p_ensemble"], backtest["target"]).a
    assert pendenza > 0.6, f"pendenza {pendenza:.2f}: probabilità troppo estreme"


def test_report_confronta_tutti_i_modelli(backtest):
    tabella = E.report(backtest)
    modelli = set(tabella["modello"])
    assert {"p_ensemble", "p_logistica", "p_hmm", "p_albero", "p_prior"} <= modelli
    assert "quota (implicita)" in modelli       # i dati sintetici hanno le quote


def test_min_train_rispettato(played):
    with pytest.raises(ValueError, match="dati insufficienti"):
        E.walk_forward(played, min_train=100_000)
