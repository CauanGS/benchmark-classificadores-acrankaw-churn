"""Testes do harness do benchmark (CLI + K-fold estratificado + avaliação + CSV)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.run_benchmark import COLUNAS_CSV, COLUNAS_DESVIO, frames_do_fold, rodar_fase_1, rodar_fase_2

RAIZ = str(Path(__file__).resolve().parents[1])


@pytest.fixture(scope="module")
def dataset_mini(tmp_path_factory):
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
    caminho = tmp_path_factory.mktemp("dados") / "mini.csv"
    df["TotalCharges"] = df["TotalCharges"].map("{:.2f}".format)
    df.to_csv(caminho, index=False)
    return caminho


@pytest.fixture(scope="module")
def fase_1(dataset_mini, tmp_path_factory):
    """Run da Fase 1 (dummy, xgboost, cba, ac_ranka) + artefatos SHAP."""
    dir_saida = tmp_path_factory.mktemp("fase1")
    saida = dir_saida / "benchmark_results.csv"
    rodar_fase_1(
        caminho_dataset=str(dataset_mini),
        caminho_saida=str(saida),
        diretorio_shap=str(dir_saida),
    )
    return pd.read_csv(saida), dir_saida


@pytest.fixture(scope="module")
def fase_2(dataset_mini, tmp_path_factory):
    """Run da Fase 2 (cba, ac_ranka, ac_rankaw + ΔL) — sem XGBoost."""
    saida = tmp_path_factory.mktemp("fase2") / "benchmark_results_phase2.csv"
    rodar_fase_2(caminho_dataset=str(dataset_mini), caminho_saida=str(saida))
    return pd.read_csv(saida)


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


# ---- Fase 1: dummy_no, xgboost, cba, ac_ranka ----

def test_fase_1_gera_csv_bem_formado(fase_1):
    df, _ = fase_1
    assert set(df.columns) == set(COLUNAS_CSV) | set(COLUNAS_DESVIO)
    dummy = df[df["modelo"] == "dummy_no"]
    assert sorted(dummy["fold"].unique()) == [str(i) for i in range(10)] + ["média"]
    assert (dummy["L"] == 0).all()
    assert df["f1_macro"].between(0, 1).all()


def test_fase_1_tem_somente_os_modelos_da_fase_1(fase_1):
    """Fase 1 = dummy, xgboost, cba, ac_ranka (o AC.RankAW é da Fase 2)."""
    df, _ = fase_1
    assert set(df["modelo"]) == {"dummy_no", "xgboost", "cba", "ac_ranka"}
    medias = df[df["fold"] == "média"]
    assert len(medias) == 4


def test_fase_1_bloco_de_medias_tem_desvio_padrao(fase_1):
    """Média ± desvio: o bloco "média" traz o desvio padrão (ddof=1) das
    dobras de cada métrica; as linhas de fold não têm desvio."""
    df, _ = fase_1
    digitos = df["fold"].astype(str).str.isdigit()
    medias = df[df["fold"] == "média"]

    for metrica in ("f1_macro", "precisao_No", "recall_No", "precisao_Yes",
                    "recall_Yes", "L", "delta_L"):
        assert f"{metrica}_desvio" in df.columns

    for modelo in ("dummy_no", "xgboost", "cba", "ac_ranka"):
        folds = df[(df["modelo"] == modelo) & digitos]
        linha = medias[medias["modelo"] == modelo].iloc[0]
        assert linha["f1_macro_desvio"] == pytest.approx(
            folds["f1_macro"].std(ddof=1), abs=2e-6
        )
        assert linha["L_desvio"] == pytest.approx(folds["L"].std(ddof=1), abs=2e-6)

    # linhas de dobra não têm desvio (só o bloco de médias tem)
    assert df[digitos]["f1_macro_desvio"].isna().all()
    assert df[digitos]["L_desvio"].isna().all()


def test_fase_1_bloco_de_medias(fase_1):
    df, _ = fase_1
    digitos = df["fold"].astype(str).str.isdigit()
    medias = df[df["fold"] == "média"]
    for modelo in ("dummy_no", "xgboost", "cba", "ac_ranka"):
        folds = df[(df["modelo"] == modelo) & digitos]
        esperado = folds["f1_macro"].mean()
        registrado = medias[medias["modelo"] == modelo].iloc[0]["f1_macro"]
        assert registrado == pytest.approx(esperado, abs=2e-6)


def test_fase_1_dummy_tipo_precision(fase_1):
    df, _ = fase_1
    dummy = df[df["modelo"] == "dummy_no"]
    folds = dummy[dummy["fold"].astype(str).str.isdigit()]
    assert (folds["precisao_Yes"] == 0).all()
    assert (folds["recall_Yes"] == 0).all()


def test_fase_1_modelos_associativos_com_L_positivo(fase_1):
    df, _ = fase_1
    digitos = df["fold"].astype(str).str.isdigit()
    for modelo in ("cba", "ac_ranka"):
        folds = df[(df["modelo"] == modelo) & digitos]
        assert len(folds) == 10
        assert folds["f1_macro"].between(0, 1).all()
        assert (folds["L"] > 0).all()


def test_fase_1_gera_artefatos_shap(fase_1):
    _, dir_saida = fase_1
    pngs = sorted(p.name for p in (dir_saida / "shap").glob("*.png"))
    assert len(pngs) == 6


# ---- Fase 2: ac_rankaw + ΔL (sem reexecutar o XGBoost) ----

def test_fase_2_tem_somente_os_modelos_da_fase_2(fase_2):
    """Fase 2 = cba, ac_ranka (referências do ΔL) + ac_rankaw; sem XGBoost."""
    df = fase_2
    assert set(df["modelo"]) == {"cba", "ac_ranka", "ac_rankaw"}
    assert "xgboost" not in set(df["modelo"])
    assert "dummy_no" not in set(df["modelo"])
    medias = df[df["fold"] == "média"]
    assert len(medias) == 3


def test_fase_2_ac_rankaw_com_L_positivo(fase_2):
    df = fase_2
    digitos = df["fold"].astype(str).str.isdigit()
    aw = df[(df["modelo"] == "ac_rankaw") & digitos]
    assert len(aw) == 10
    assert aw["f1_macro"].between(0, 1).all()
    assert (aw["L"] > 0).all()


def test_fase_2_delta_l_somente_linhas_ac_rankaw(fase_2):
    df = fase_2
    outros = df[df["modelo"] != "ac_rankaw"]
    assert outros["delta_L"].isna().all()
    assert outros["delta_L_vs_acranka"].isna().all()
    aw = df[(df["modelo"] == "ac_rankaw") & df["fold"].astype(str).str.isdigit()]
    assert aw["delta_L"].notna().all()
    assert aw["delta_L_vs_acranka"].notna().all()
    # por fold: ΔL = (L_ref_fold − L_AW_fold)/L_ref_fold × 100
    digitos = df["fold"].astype(str).str.isdigit()
    awl = aw.set_index("fold")
    for referencia, coluna in (("cba", "delta_L"), ("ac_ranka", "delta_L_vs_acranka")):
        ref = df[(df["modelo"] == referencia) & digitos].set_index("fold")["L"]
        esperado = (ref - awl["L"]) / ref * 100
        assert ((awl[coluna] - esperado).abs() < 2e-6).all()


def test_fase_2_delta_l_media_das_medias_de_L(fase_2):
    # No bloco "média": ΔL é recalculado a partir das MÉDIAS de L dos folds do
    # run (a média de uma razão não é a razão das médias) — não é a média dos
    # ΔL por fold.
    df = fase_2
    digitos = df["fold"].astype(str).str.isdigit()
    media_L = (
        df[digitos]
        .groupby("modelo")["L"]
        .mean()
        .reindex(["cba", "ac_ranka", "ac_rankaw"])
    )
    medias = df[df["fold"] == "média"]
    linha = medias[medias["modelo"] == "ac_rankaw"].iloc[0]
    esperado_cba = (media_L["cba"] - media_L["ac_rankaw"]) / media_L["cba"] * 100
    esperado_a = (media_L["ac_ranka"] - media_L["ac_rankaw"]) / media_L["ac_ranka"] * 100
    assert linha["delta_L"] == pytest.approx(esperado_cba, abs=2e-6)
    assert linha["delta_L_vs_acranka"] == pytest.approx(esperado_a, abs=2e-6)


def test_fase_2_nao_gera_artefatos_shap(dataset_mini, tmp_path):
    saida = tmp_path / "resultados_fase2.csv"
    rodar_fase_2(caminho_dataset=str(dataset_mini), caminho_saida=str(saida))
    assert list(tmp_path.glob("**/*.png")) == []


def test_fase_2_deterministico(dataset_mini, tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    rodar_fase_2(caminho_dataset=str(dataset_mini), caminho_saida=str(a))
    rodar_fase_2(caminho_dataset=str(dataset_mini), caminho_saida=str(b))
    assert a.read_text() == b.read_text()


# ---- CLI: --phase 1 e --phase 2 ----

def test_cli_phase_1_gera_csv_da_fase_1(dataset_mini, tmp_path):
    saida = tmp_path / "outputs"
    proc = subprocess.run(
        [sys.executable, "-m", "src.run_benchmark", "--phase", "1",
         "--dataset", str(dataset_mini), "--saida", str(saida)],
        capture_output=True, text=True, cwd=RAIZ,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    csv = saida / "benchmark_results.csv"
    assert csv.exists()
    df = pd.read_csv(csv)
    assert set(df["modelo"]) == {"dummy_no", "xgboost", "cba", "ac_ranka"}
    assert "Fase 1" in proc.stdout


def test_cli_phase_2_gera_csv_da_fase_2(dataset_mini, tmp_path):
    saida = tmp_path / "outputs"
    proc = subprocess.run(
        [sys.executable, "-m", "src.run_benchmark", "--phase", "2",
         "--dataset", str(dataset_mini), "--saida", str(saida)],
        capture_output=True, text=True, cwd=RAIZ,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    csv = saida / "benchmark_results_phase2.csv"
    assert csv.exists()
    df = pd.read_csv(csv)
    assert "ac_rankaw" in set(df["modelo"])
    assert "xgboost" not in set(df["modelo"])
    assert "Fase 2" in proc.stdout


def test_csvs_das_fases_1_e_2_tem_o_mesmo_schema(dataset_mini, tmp_path):
    """A leitura dos CSVs das duas fases não quebra: mesmo schema."""
    saida = tmp_path / "outputs"
    for fase in (1, 2):
        proc = subprocess.run(
            [sys.executable, "-m", "src.run_benchmark", "--phase", str(fase),
             "--dataset", str(dataset_mini), "--saida", str(saida)],
            capture_output=True, text=True, cwd=RAIZ,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
    df1 = pd.read_csv(saida / "benchmark_results.csv")
    df2 = pd.read_csv(saida / "benchmark_results_phase2.csv")
    assert list(df1.columns) == list(df2.columns)
