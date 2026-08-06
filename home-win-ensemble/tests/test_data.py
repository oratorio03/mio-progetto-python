"""Riconoscimento dello schema e pulizia dei dati."""

import numpy as np
import pandas as pd
import pytest

from hwe import data as D
from hwe import schema as S


def test_riconosce_football_data(csv_path):
    """Le colonne di football-data.co.uk devono essere capite senza --map."""
    raw = pd.read_csv(csv_path)
    mapping = S.detect(raw.columns)
    assert mapping["date"] == "Date"
    assert mapping["home"] == "HomeTeam"
    assert mapping["away"] == "AwayTeam"
    assert mapping["home_goals"] == "FTHG"
    assert mapping["away_goals"] == "FTAG"
    assert mapping["odds_home"] == "B365H"
    assert mapping["league"] == "Div"


@pytest.mark.parametrize("colonne, attesi", [
    (["Data", "Squadra casa", "Squadra trasferta", "Gol casa", "Gol trasferta"],
     {"date": "Data", "home": "Squadra casa", "away": "Squadra trasferta"}),
    (["date", "home_team", "away_team", "home_score", "away_score"],
     {"date": "date", "home": "home_team", "away": "away_team",
      "home_goals": "home_score", "away_goals": "away_score"}),
    (["kickoff", "HomeTeam", "AwayTeam", "FTHG", "FTAG"],
     {"date": "kickoff", "home": "HomeTeam"}),
])
def test_riconosce_altri_nomi(colonne, attesi):
    mapping = S.detect(colonne)
    for canonica, colonna in attesi.items():
        assert mapping[canonica] == colonna


def test_errore_utile_se_mancano_colonne():
    with pytest.raises(S.SchemaError) as errore:
        S.detect(["pippo", "pluto", "paperino"])
    messaggio = str(errore.value)
    assert "--map" in messaggio          # dice come rimediare
    assert "pippo" in messaggio          # e cosa ha trovato


def test_override_manuale():
    mapping = S.detect(["quando", "A", "B", "gA", "gB"],
                       overrides={"date": "quando", "home": "A", "away": "B",
                                  "home_goals": "gA", "away_goals": "gB"})
    assert mapping["home_goals"] == "gA"

    with pytest.raises(S.SchemaError):
        S.detect(["a", "b"], overrides={"home": "colonna_inesistente"})


def test_partite_giocate_e_future(matches):
    played, future = D.split_played(matches)
    assert len(future) > 0, "il generatore lascia delle partite da giocare"
    assert played["home_goals"].notna().all()
    assert future["home_goals"].isna().all()
    assert played["date"].max() <= future["date"].max()
    assert matches["date"].is_monotonic_increasing


def test_scarta_righe_rotte(tmp_path):
    """Date illeggibili, squadre vuote e derby impossibili vanno via."""
    path = tmp_path / "sporco.csv"
    pd.DataFrame({
        "Date": ["01/09/2024", "non-una-data", "08/09/2024", "15/09/2024"],
        "HomeTeam": ["Alfa", "Beta", "", "Delta"],
        "AwayTeam": ["Beta", "Alfa", "Gamma", "Delta"],
        "FTHG": [2, 1, 0, 3],
        "FTAG": [0, 1, 1, 1],
    }).to_csv(path, index=False)

    df = D.load_matches(path)
    assert len(df) == 1                      # sopravvive solo la prima riga
    assert df.iloc[0]["home"] == "Alfa"


def test_file_senza_risultati(tmp_path):
    path = tmp_path / "solo_futuro.csv"
    pd.DataFrame({"Date": ["01/09/2024"], "HomeTeam": ["Alfa"],
                  "AwayTeam": ["Beta"], "FTHG": [np.nan],
                  "FTAG": [np.nan]}).to_csv(path, index=False)
    with pytest.raises(D.DataError, match="nessuna partita con risultato"):
        D.load_matches(path)


def test_stagione_dedotta_da_date(tmp_path):
    """Senza colonna stagione, agosto-luglio con etichetta l'anno d'inizio."""
    path = tmp_path / "senza_stagione.csv"
    pd.DataFrame({
        "Date": ["10/08/2023", "10/03/2024", "10/09/2024"],
        "HomeTeam": ["Alfa", "Beta", "Gamma"],
        "AwayTeam": ["Beta", "Gamma", "Alfa"],
        "FTHG": [1, 2, 3], "FTAG": [0, 0, 0],
    }).to_csv(path, index=False)
    df = D.load_matches(path)
    assert list(df["season"]) == [2023, 2023, 2024]


def test_riepilogo(matches):
    info = D.summary(matches)
    assert info["squadre"] == 20
    assert info["partite_giocate"] > 1000
    assert 30 < info["vittorie_casa_%"] < 60
