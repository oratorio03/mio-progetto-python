"""
conftest.py — Dati e modello condivisi fra i test.

Il campionato sintetico e l'ensemble si generano una volta sola per sessione:
addestrare costa qualche secondo e non c'è motivo di rifarlo per ogni test.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hwe import data as D              # noqa: E402
from hwe import features as F          # noqa: E402
from hwe import synthetic              # noqa: E402
from hwe.ensemble import HomeWinEnsemble   # noqa: E402


@pytest.fixture(scope="session")
def csv_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("dati") / "partite.csv"
    synthetic.write_csv(path, seasons=3, seed=7)
    return path


@pytest.fixture(scope="session")
def matches(csv_path):
    return D.load_matches(csv_path)


@pytest.fixture(scope="session")
def played(matches):
    return D.split_played(matches)[0]


@pytest.fixture(scope="session")
def future(matches):
    return D.split_played(matches)[1]


@pytest.fixture(scope="session")
def table(matches):
    return F.build(matches)


@pytest.fixture(scope="session")
def trained(played):
    """Ensemble addestrato su tutte le partite giocate."""
    ens = HomeWinEnsemble(verbose=False).fit(played)
    ens.set_history(played)
    return ens


@pytest.fixture(scope="session")
def holdout(played, table):
    """Split cronologico 75/25 con ensemble addestrato solo sul primo tratto."""
    import pandas as pd

    done = table[table["target"].notna()].reset_index(drop=True)
    cut = int(len(done) * 0.75)
    split_date = done["date"].iloc[cut]
    train, test = done.iloc[:cut], done.iloc[cut:]
    past = played[pd.to_datetime(played["date"]) < split_date]

    ens = HomeWinEnsemble(verbose=False).fit(past, table=train)
    ens.set_history(past)
    detail = ens.predict_proba(test, detail=True)
    return ens, test, detail
