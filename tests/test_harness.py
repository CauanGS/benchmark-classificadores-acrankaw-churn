"""Testes do harness do benchmark (CLI + K-fold estratificado + avaliação + CSV)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.run_benchmark import frames_do_fold, rodar_benchmark


def test_frames_do_fold_separam_o_ramo_continuo_do_ramo_discretizado():
    """SDD §2: o XGBoost usa o frame ORIGINAL (ramo 1) e os associativos usam
    os bins MDLP (ramo 2). O helper é o seam público dessa separação."""
    n = 24
    X = pd.DataFrame(
        {
            "customerID": [f"C{i}" for i in range(n)],
            "tenure": [float(i) for i in range(n)],
            "Contract": ["One year" if i % 2 else "Month-to-month" for i in range(n)],
        }
    )
    y = pd.Series(["No" if i < 12 else "Yes" for i in range(n)])
    frames = frames_do_fold(X, y, X)

    # ramo 1 (contínuos/dummies): frame original, sem customerID, valores intactos
    cont_treino, cont_teste = frames["continuos"]
    assert "customerID" not in cont_treino.columns
    assert list(cont_treino["tenure"]) == list(X["tenure"])
    assert list(cont_treino["Contract"]) == list(X["Contract"])

    # ramo 2 (discretizado): bins MDLP ajustados no treino
    disc_treino, disc_teste = frames["discretizados"]
    assert disc_treino["tenure"].tolist() != cont_treino["tenure"].tolist()
    assert all("tenure" in str(v) for v in disc_treino["tenure"])
    assert len(disc_teste) == len(cont_teste)


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


def test_rodar_benchmark_gera_csv_bem_formado(dataset_mini, tmp_path):
    saida = tmp_path / "resultados.csv"
    rodar_benchmark(caminho_dataset=str(dataset_mini),
                    caminho_saida=str(saida),
                    diretorio_shap=str(tmp_path))
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
        "delta_L",
    }
    dummy = df[df["modelo"] == "dummy_no"]
    assert sorted(dummy["fold"].unique()) == [str(i) for i in range(10)] + ["média"]
    assert (dummy["L"] == 0).all()
    assert df["f1_macro"].between(0, 1).all()


def test_rodar_benchmark_bloco_de_medias(dataset_mini, tmp_path):
    saida = tmp_path / "resultados.csv"
    rodar_benchmark(caminho_dataset=str(dataset_mini), caminho_saida=str(saida))
    df = pd.read_csv(saida)
    medias = df[df["fold"] == "média"]
    assert len(medias) == 5  # um bloco por modelo: dummy_no, xgboost, cba, ac_ranka, ac_rankaw
    for modelo in ("dummy_no", "xgboost", "cba", "ac_ranka", "ac_rankaw"):
        folds = df[(df["modelo"] == modelo) & (df["fold"].astype(str).str.isdigit())]
        esperado = folds["f1_macro"].mean()
        registrado = medias[medias["modelo"] == modelo].iloc[0]["f1_macro"]
        assert registrado == pytest.approx(esperado, abs=2e-6)


def test_rodar_benchmark_deterministico(dataset_mini, tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    rodar_benchmark(caminho_dataset=str(dataset_mini), caminho_saida=str(a))
    rodar_benchmark(caminho_dataset=str(dataset_mini), caminho_saida=str(b))
    assert a.read_text() == b.read_text()


def test_rodar_benchmark_dummy_tipo_precision(dataset_mini, tmp_path):
    saida = tmp_path / "resultados.csv"
    rodar_benchmark(caminho_dataset=str(dataset_mini), caminho_saida=str(saida))
    df = pd.read_csv(saida)
    dummy = df[df["modelo"] == "dummy_no"]
    folds = dummy[dummy["fold"].astype(str).str.isdigit()]
    assert (folds["precisao_Yes"] == 0).all()
    assert (folds["recall_Yes"] == 0).all()


def test_rodar_benchmark_cba_linhas_com_L_positivo(dataset_mini, tmp_path):
    saida = tmp_path / "resultados.csv"
    rodar_benchmark(caminho_dataset=str(dataset_mini), caminho_saida=str(saida))
    df = pd.read_csv(saida)
    cba = df[df["modelo"] == "cba"]
    folds_cba = cba[cba["fold"].astype(str).str.isdigit()]
    assert len(folds_cba) == 10
    assert folds_cba["f1_macro"].between(0, 1).all()
    assert (folds_cba["L"] > 0).all()


def test_rodar_benchmark_ac_ranka_linhas_com_L_positivo(dataset_mini, tmp_path):
    saida = tmp_path / "resultados.csv"
    rodar_benchmark(caminho_dataset=str(dataset_mini), caminho_saida=str(saida))
    df = pd.read_csv(saida)
    ac = df[df["modelo"] == "ac_ranka"]
    folds_ac = ac[ac["fold"].astype(str).str.isdigit()]
    assert len(folds_ac) == 10
    assert folds_ac["f1_macro"].between(0, 1).all()
    assert (folds_ac["L"] > 0).all()


def test_rodar_benchmark_ac_rankaw_linhas_com_L_positivo(dataset_mini, tmp_path):
    saida = tmp_path / "resultados.csv"
    rodar_benchmark(caminho_dataset=str(dataset_mini), caminho_saida=str(saida))
    df = pd.read_csv(saida)
    aw = df[df["modelo"] == "ac_rankaw"]
    folds_aw = aw[aw["fold"].astype(str).str.isdigit()]
    assert len(folds_aw) == 10
    assert folds_aw["f1_macro"].between(0, 1).all()
    assert (folds_aw["L"] > 0).all()


# ---- ΔL (SDD 5.2): (L_CBA − L_AC.RankAW)/L_CBA × 100 ----

def test_rodar_benchmark_delta_l_somente_linhas_ac_rankaw(dataset_mini, tmp_path):
    saida = tmp_path / "resultados.csv"
    rodar_benchmark(caminho_dataset=str(dataset_mini), caminho_saida=str(saida))
    df = pd.read_csv(saida)
    outros = df[df["modelo"] != "ac_rankaw"]
    assert outros["delta_L"].isna().all()
    aw = df[(df["modelo"] == "ac_rankaw") & (df["fold"].astype(str).str.isdigit())]
    assert aw["delta_L"].notna().all()
    # por fold: ΔL = (L_CBA_fold − L_AW_fold)/L_CBA_fold × 100
    digitos = df["fold"].astype(str).str.isdigit()
    cba_l = df[(df["modelo"] == "cba") & digitos].set_index("fold")["L"]
    awl = aw.set_index("fold")
    esperado = (cba_l - awl["L"]) / cba_l * 100
    comparado = (awl["delta_L"] - esperado).abs()
    assert (comparado < 2e-6).all()


def test_rodar_benchmark_delta_l_media_das_medias_de_L(dataset_mini, tmp_path):
    # No bloco "média": ΔL é recalculado a partir das MÉDIAS de L dos folds do
    # run (a média de uma razão não é a razão das médias) — não é a média dos
    # ΔL por fold.
    saida = tmp_path / "resultados.csv"
    rodar_benchmark(caminho_dataset=str(dataset_mini), caminho_saida=str(saida))
    df = pd.read_csv(saida)
    digitos = df["fold"].astype(str).str.isdigit()
    media_L = (
        df[digitos]
        .groupby("modelo")["L"]
        .mean()
        .reindex(["cba", "ac_rankaw"])
    )
    esperado = (media_L["cba"] - media_L["ac_rankaw"]) / media_L["cba"] * 100
    medias = df[df["fold"] == "média"]
    registrado = medias[medias["modelo"] == "ac_rankaw"].iloc[0]["delta_L"]
    assert registrado == pytest.approx(esperado, abs=2e-6)


def test_rodar_benchmark_fase2_sem_arquivo_shap(dataset_mini, tmp_path):
    # Fase 2 = mesma pipeline da fase 1 (todos os modelos); diferenças: só
    # grava o CSV dedicado da fase e NÃO gera artefatos SHAP.
    saida = tmp_path / "resultados_fase2.csv"
    rodar_benchmark(
        caminho_dataset=str(dataset_mini),
        caminho_saida=str(saida),
        diretorio_shap=None,
    )
    df = pd.read_csv(saida)
    assert set(df.columns) == {
        "modelo", "fold", "f1_macro", "precisao_No", "recall_No",
        "precisao_Yes", "recall_Yes", "L", "delta_L",
    }
    aw = df[(df["modelo"] == "ac_rankaw") & (df["fold"].astype(str).str.isdigit())]
    assert len(aw) == 10
    assert aw["delta_L"].notna().all()
    assert list((tmp_path).glob("**/*.png")) == []
