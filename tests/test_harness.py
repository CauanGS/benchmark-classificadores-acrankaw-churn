"""Testes do harness do benchmark (CLI + K-fold estratificado + avaliação + CSV)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.run_benchmark import rodar_fase1


@pytest.fixture
def dataset_mini(tmp_path):
    """Dataset sintético pequeno com sinal conhecido: Churn=Yes quando tenure alto."""

    rng = np.random.default_rng(42)
    n = 60
    tenure = rng.uniform(0, 72, size=n)
    monthly = rng.uniform(20, 120, size=n)
    churn = np.where(tenure + 0.2 * monthly > 60, "Yes", "No")
    df = pd.DataFrame(
        {
            "customerID": [f"C{i}" for i in range(n)],
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": tenure * monthly,
            "SeniorCitizen": rng.choice([0, 1], size=n),
            "Churn": churn,
        }
    )
    caminho = tmp_path / "mini.csv"
    df["TotalCharges"] = df["TotalCharges"].map("{:.2f}".format)
    df.to_csv(caminho, index=False)
    return caminho


def test_rodar_fase1_gera_csv_bem_formado(dataset_mini, tmp_path):
    saida = tmp_path / "resultados.csv"
    rodar_fase1(caminho_dataset=str(dataset_mini), caminho_saida=str(saida))
    df = pd.read_csv(saida)
    assert set(df.columns) == {
        "modelo",
        "fold",
        "f1_macro",
        "precisao_No",
        "recall_No",
        "precisao_Yes",
        "recall_Yes",
        "L",
    }
    dummy = df[df["modelo"] == "dummy_no"]
    assert sorted(dummy["fold"].unique()) == [str(i) for i in range(10)] + ["média"]
    assert (dummy["L"] == 0).all()
    assert df["f1_macro"].between(0, 1).all()


def test_rodar_fase1_bloco_de_medias(dataset_mini, tmp_path):
    saida = tmp_path / "resultados.csv"
    rodar_fase1(caminho_dataset=str(dataset_mini), caminho_saida=str(saida))
    df = pd.read_csv(saida)
    dummy = df[df["modelo"] == "dummy_no"]
    folds = dummy[dummy["fold"].astype(str).str.isdigit()]
    medias = df[df["fold"] == "média"]
    assert len(medias) == 2  # um bloco por modelo: dummy_no e cba
    assert sorted(medias["modelo"]) == ["cba", "dummy_no"]
    esperado = folds["f1_macro"].mean()
    bloco_dummy = medias[medias["modelo"] == "dummy_no"]
    assert bloco_dummy.iloc[0]["f1_macro"] == pytest.approx(esperado, abs=2e-6)


def test_rodar_fase1_deterministico(dataset_mini, tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    rodar_fase1(caminho_dataset=str(dataset_mini), caminho_saida=str(a))
    rodar_fase1(caminho_dataset=str(dataset_mini), caminho_saida=str(b))
    assert a.read_text() == b.read_text()


def test_rodar_fase1_dummy_tipo_precision(dataset_mini, tmp_path):
    saida = tmp_path / "resultados.csv"
    rodar_fase1(caminho_dataset=str(dataset_mini), caminho_saida=str(saida))
    df = pd.read_csv(saida)
    dummy = df[df["modelo"] == "dummy_no"]
    folds = dummy[dummy["fold"].astype(str).str.isdigit()]
    assert (folds["precisao_Yes"] == 0).all()
    assert (folds["recall_Yes"] == 0).all()


def test_rodar_fase1_cba_linhas_com_L_positivo(dataset_mini, tmp_path):
    saida = tmp_path / "resultados.csv"
    rodar_fase1(caminho_dataset=str(dataset_mini), caminho_saida=str(saida))
    df = pd.read_csv(saida)
    cba = df[df["modelo"] == "cba"]
    folds_cba = cba[cba["fold"].astype(str).str.isdigit()]
    assert len(folds_cba) == 10
    assert folds_cba["f1_macro"].between(0, 1).all()
    assert (folds_cba["L"] > 0).all()
