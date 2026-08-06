"""
HMM: verifichiamo che ritrovi la struttura che il generatore ci ha messo.

I dati sintetici hanno una forma latente che evolve come catena di Markov
persistente e un vantaggio del campo. Se l'HMM è implementato bene deve
ritrovare entrambe le cose senza che gliele si dica.
"""

import numpy as np
import pytest

from hwe import hmm as H


def test_codifica_simboli():
    assert H.encode(3, 0) == 4      # vittoria larga
    assert H.encode(2, 1) == 3      # vittoria di misura
    assert H.encode(1, 1) == 2      # pareggio
    assert H.encode(0, 1) == 1      # sconfitta di misura
    assert H.encode(0, 4) == 0      # sconfitta larga


def test_sequenze_per_squadra(played):
    sequenze = H.build_sequences(played)
    assert len(sequenze) == 20

    for team, seq in sequenze.items():
        assert len(seq["dates"]) == len(seq["venues"]) == len(seq["symbols"])
        assert np.all(np.diff(seq["dates"].astype("int64")) >= 0), \
            "le sequenze devono essere cronologiche"
        assert set(np.unique(seq["venues"])) <= {0, 1}
        assert seq["symbols"].min() >= 0 and seq["symbols"].max() <= 4


@pytest.fixture(scope="module")
def modello(played):
    sequenze = H.build_sequences(played)
    return H.FormHMM(n_states=3, n_restarts=2, n_iter=25).fit(
        [(s["venues"], s["symbols"]) for s in sequenze.values()])


def test_parametri_sono_distribuzioni(modello):
    assert np.isclose(modello.pi.sum(), 1.0)
    assert np.allclose(modello.A.sum(axis=1), 1.0)
    assert np.allclose(modello.B.sum(axis=2), 1.0)
    assert (modello.A >= 0).all() and (modello.B >= 0).all()


def test_stati_ordinati_per_forza(modello):
    """s0 deve essere lo stato peggiore e s(K-1) il migliore."""
    griglia = np.arange(H.N_SYMBOLS)
    forza = (modello.B[:, 0, :] @ griglia + modello.B[:, 1, :] @ griglia) / 2
    assert np.all(np.diff(forza) > 0), f"stati non ordinati: {forza}"


def test_forma_persistente(modello):
    """La forma cambia lentamente: la diagonale deve dominare."""
    assert np.mean(np.diag(modello.A)) > 0.35


def test_impara_il_vantaggio_del_campo(modello):
    """In ogni stato si deve vincere più in casa che in trasferta."""
    casa = modello.B[:, 0, H.SYM_WIN].sum(axis=1)
    fuori = modello.B[:, 1, H.SYM_WIN].sum(axis=1)
    assert np.all(casa > fuori), f"casa {casa} vs trasferta {fuori}"


def test_verosimiglianza_non_peggiora(played):
    """EM è monotono: ogni iterazione non deve abbassare la verosimiglianza."""
    sequenze = [(s["venues"], s["symbols"])
                for s in H.build_sequences(played.iloc[:400]).values()]
    modello = H.FormHMM(n_states=3)
    rng = np.random.default_rng(0)
    modello.pi, modello.A, modello.B = modello._init(rng)

    valori = [modello._em_step(sequenze) for _ in range(12)]
    for prima, dopo in zip(valori, valori[1:]):
        assert dopo >= prima - 1e-6, f"la verosimiglianza è scesa: {prima} → {dopo}"


def test_filtraggio_non_usa_la_partita_corrente(modello):
    """
    La riga t del percorso filtrato deve dipendere solo dalle osservazioni
    precedenti: cambiare l'esito della partita t non deve toccarla.
    """
    venues = np.array([0, 1, 0, 1, 0, 1])
    symbols = np.array([4, 0, 3, 1, 4, 2])
    percorso = modello.filter_path(venues, symbols)

    modificati = symbols.copy()
    modificati[3] = 4                      # cambio l'esito della quarta partita
    percorso2 = modello.filter_path(venues, modificati)

    assert np.allclose(percorso[:4], percorso2[:4]), \
        "le righe fino a t=3 non devono cambiare"
    assert not np.allclose(percorso[4], percorso2[4]), \
        "dalla riga successiva invece deve cambiare"


def test_percorso_ha_una_riga_in_piu(modello):
    percorso = modello.filter_path([0, 1, 0], [3, 2, 4])
    assert percorso.shape == (4, 3)
    assert np.allclose(percorso[0], modello.pi)      # prior prima di tutto
    assert np.allclose(percorso.sum(axis=1), 1.0)


def test_classificatore_taglia_per_data(played, table):
    """
    Il classificatore deve dare lo stesso risultato con lo storico completo o
    con quello troncato alla vigilia: quel che viene dopo non lo tocca.
    """
    done = table[table["target"].notna()]
    train, test = done.iloc[:600], done.iloc[600:700]
    past = played[played["date"] < test["date"].iloc[0]]

    modello = H.HMMHomeWin(n_states=3).fit(
        past, train, train["target"].to_numpy(dtype=float), history=past)

    modello.set_history(past)
    solo_passato = modello.predict_proba(test.iloc[:1])
    modello.set_history(played)                 # storico completo
    tutto = modello.predict_proba(test.iloc[:1])

    assert np.allclose(solo_passato, tutto), \
        "il filtraggio sta usando partite successive a quella prevista"


def test_probabilita_valide(played, table):
    done = table[table["target"].notna()]
    train = done.iloc[:800]
    modello = H.HMMHomeWin(n_states=3).fit(
        played, train, train["target"].to_numpy(dtype=float))
    modello.set_history(played)

    p = modello.predict_proba(done.iloc[800:])
    assert np.isfinite(p).all()
    assert ((p > 0) & (p < 1)).all()
    assert p.std() > 0.01, "il modello non distingue una partita dall'altra"
