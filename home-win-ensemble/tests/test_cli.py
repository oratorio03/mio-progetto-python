"""Riga di comando: i comandi devono girare davvero, non solo importare."""

import pandas as pd
import pytest

from hwe import synthetic
from hwe.cli import main


def test_schema(csv_path, capsys):
    assert main(["schema", "--data", str(csv_path)]) == 0
    stampato = capsys.readouterr().out
    assert "HomeTeam" in stampato
    assert "partite_giocate" in stampato


def test_train_predict_inspect(csv_path, tmp_path, capsys):
    modello = tmp_path / "modello.pkl"
    previsioni = tmp_path / "previsioni.csv"

    assert main(["train", "--data", str(csv_path), "--model", str(modello)]) == 0
    assert modello.exists()

    assert main(["inspect", "--model", str(modello)]) == 0
    stampato = capsys.readouterr().out
    assert "transizioni" in stampato.lower()

    assert main(["predict", "--data", str(csv_path), "--model", str(modello),
                 "--out", str(previsioni), "--top", "5"]) == 0

    out = pd.read_csv(previsioni)
    assert len(out) == 40                         # 4 giornate da 10 partite
    assert out["p_ensemble"].between(0, 1).all()
    assert out["p_ensemble"].is_monotonic_decreasing
    assert {"p_logistica", "p_hmm", "p_albero", "edge", "value"} <= set(out.columns)


def test_backtest(csv_path, tmp_path):
    uscita = tmp_path / "backtest.csv"
    assert main(["backtest", "--data", str(csv_path), "--min-train", "500",
                 "--step-days", "90", "--max-steps", "2",
                 "--out", str(uscita)]) == 0

    preds = pd.read_csv(uscita)
    assert len(preds) > 100
    assert "p_prior" in preds.columns


def test_inspect_senza_modello(tmp_path, capsys):
    assert main(["inspect", "--model", str(tmp_path / "assente.pkl")]) == 1
    assert "train" in capsys.readouterr().out


def test_errore_schema_ha_uscita_dedicata(tmp_path, capsys):
    brutto = tmp_path / "brutto.csv"
    pd.DataFrame({"pippo": [1], "pluto": [2]}).to_csv(brutto, index=False)

    assert main(["schema", "--data", str(brutto)]) == 2
    assert "--map" in capsys.readouterr().err


def test_map_manuale(tmp_path):
    """Un CSV con nomi inventati deve funzionare passando --map."""
    strano = tmp_path / "strano.csv"
    df = synthetic.generate(n_teams=10, seasons=1, future_matchdays=1, seed=3)
    df = df.rename(columns={"Date": "quando", "HomeTeam": "chi_gioca_in_casa",
                            "AwayTeam": "chi_viene", "FTHG": "reti1",
                            "FTAG": "reti2"})
    df.to_csv(strano, index=False)

    assert main(["schema", "--data", str(strano),
                 "--map", "date=quando",
                 "--map", "home=chi_gioca_in_casa",
                 "--map", "away=chi_viene",
                 "--map", "home_goals=reti1",
                 "--map", "away_goals=reti2"]) == 0


def test_predict_senza_partite_future(tmp_path, capsys):
    solo_passato = tmp_path / "passato.csv"
    synthetic.write_csv(solo_passato, seasons=1, future_matchdays=0, seed=5)

    assert main(["predict", "--data", str(solo_passato),
                 "--model", str(tmp_path / "m.pkl")]) == 1
    assert "Nessuna partita da prevedere" in capsys.readouterr().out


@pytest.mark.slow
def test_demo(tmp_path, capsys):
    """La demo è il biglietto da visita: deve girare dall'inizio alla fine."""
    assert main(["demo", "--workdir", str(tmp_path), "--seasons", "2",
                 "--min-train", "400", "--max-steps", "2"]) == 0

    stampato = capsys.readouterr().out
    for sezione in ("LETTURA DEL FILE", "ADDESTRAMENTO", "COSA HA IMPARATO",
                    "PREVISIONI", "BACKTEST"):
        assert sezione in stampato

    assert (tmp_path / "partite_demo.csv").exists()
    assert (tmp_path / "previsioni.csv").exists()
    assert (tmp_path / "backtest.csv").exists()
