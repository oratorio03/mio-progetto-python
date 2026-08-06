"""
Feature: il test che conta è che non guardino il futuro.

Il modo più diretto di verificarlo è ricostruire la tabella su un file
troncato: se la riga i usasse informazione successiva a i, cambierebbe.
"""

import numpy as np
import pandas as pd

from hwe import features as F


def test_nessuna_informazione_dal_futuro(played):
    completa = F.build(played)
    troncata = F.build(played.iloc[:400])
    n = len(troncata)

    a = completa.iloc[:n][F.FEATURES].to_numpy(dtype=float)
    b = troncata[F.FEATURES].to_numpy(dtype=float)
    # NaN e NaN devono contare come uguali
    assert np.array_equal(a, b, equal_nan=True), (
        "le feature delle prime 400 partite cambiano se il file continua: "
        "c'è informazione futura che filtra")


def test_le_partite_future_non_sporcano_il_passato(matches, played):
    """Aggiungere le fixture da giocare non deve toccare le righe già giocate."""
    solo_giocate = F.build(played)
    con_futuro = F.build(matches)
    righe_giocate = con_futuro[con_futuro["played"] == 1].reset_index(drop=True)

    assert np.array_equal(solo_giocate[F.FEATURES].to_numpy(dtype=float),
                          righe_giocate[F.FEATURES].to_numpy(dtype=float),
                          equal_nan=True)


def test_prima_partita_senza_storia(table):
    prima = table.iloc[0]
    assert prima["history_home"] == 0
    assert prima["history_away"] == 0
    assert pd.isna(prima["h2h_home_rate"])
    assert pd.isna(prima["home_ppg_venue"])       # nessuna media inventata
    assert prima["elo_diff"] == 0.0               # tutti partono da 1500


def test_target_coerente(matches):
    tabella = F.build(matches)
    giocate = matches[matches["played"] == 1].set_index("match_id")
    campione = tabella[tabella["played"] == 1].sample(50, random_state=0)
    for row in campione.itertuples(index=False):
        partita = giocate.loc[row.match_id]
        atteso = float(partita["home_goals"] > partita["away_goals"])
        assert row.target == atteso

    assert tabella[tabella["played"] == 0]["target"].isna().all()


def test_elo_si_muove_nella_direzione_giusta(played):
    """Chi vince sale, chi perde scende, e la somma resta invariata."""
    book = F.EloBook()
    prima_casa = book.get("L", 2024, "Alfa")
    prima_ospite = book.get("L", 2024, "Beta")
    book.update("L", "Alfa", "Beta", 1.0)
    dopo_casa = book.get("L", 2024, "Alfa")
    dopo_ospite = book.get("L", 2024, "Beta")

    assert dopo_casa > prima_casa
    assert dopo_ospite < prima_ospite
    assert np.isclose(dopo_casa + dopo_ospite, prima_casa + prima_ospite)


def test_vantaggio_campo_nella_attesa_elo():
    """A parità di rating la squadra di casa deve essere favorita."""
    assert F.elo_expected(1500, 1500) > 0.5
    assert F.elo_expected(1400, 1600) < F.elo_expected(1600, 1400)


def test_imputer_usa_le_mediane_del_training(table):
    done = table[table["target"].notna()]
    train, test = done.iloc[:500], done.iloc[500:]

    imputer = F.Imputer().fit(train)
    X = imputer.transform(test)
    assert np.isfinite(X).all(), "dopo l'imputazione non devono restare NaN"

    # le mediane vengono dal training, non dal blocco che si sta trasformando
    atteso = float(pd.to_numeric(train["elo_diff"]).median())
    assert np.isclose(imputer.medians["elo_diff"], atteso)


def test_riposo_limitato(table):
    riposo = table[["rest_home", "rest_away"]].to_numpy(dtype=float)
    valori = riposo[~np.isnan(riposo)]
    assert valori.min() >= F.REST_CLIP[0]
    assert valori.max() <= F.REST_CLIP[1]
