"""Runner CLI do benchmark AC.RankAW Churn.

Uso: python -m src.run_benchmark --phase 1|2|3

Fase 1: benchmark dos modelos sobre o Telco Churn com StratifiedKFold
(10 folds, shuffle, random_state=42), discretização MDLP por fold de
treino e registro das métricas em outputs/benchmark_results.csv.
Nesta fase o slot de modelo recebe um DummyClassifier (sempre "No"),
fatorado por um modelo real nos próximos tickets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold

from src.eda import ALVO, carregar_dataset
from src.evaluation import f1_macro, precision_recall_por_classe
from src.preprocessing import MDLPDiscretizer
from src.xgboost_model import gerar_artefatos_shap, treinar_e_prever

ATRIBUTOS_IGNORADOS = ["customerID"]
RANDOM_STATE = 42
DIR_SAIDA_SHAP = "outputs"
MODELOS_FASE1 = ("dummy_no", "xgboost")
COLUNAS_CSV = [
    "modelo",
    "fold",
    "f1_macro",
    "precisao_No",
    "recall_No",
    "precisao_Yes",
    "recall_Yes",
    "L",
]
MENSAGEM_DATASET_AUSENTE = """\
Dataset não encontrado em {caminho}.
Baixe o Telco Customer Churn com:

    python -c "import kagglehub; kagglehub.dataset_download('blastchar/telco-customer-churn')"

e copie o CSV 'WA_Fn-UseC_-Telco-Customer-Churn.csv' para data/raw/.
"""


def _linha_metricas(modelo: str, fold, y_teste, y_pred, L: int) -> dict:
    pr = precision_recall_por_classe(list(y_teste), list(y_pred))
    pr_no, pr_yes = pr.get("No"), pr.get("Yes")
    return {
        "modelo": modelo,
        "fold": fold,
        "f1_macro": f1_macro(list(y_teste), list(y_pred)),
        "precisao_No": pr_no[0] if pr_no else 0.0,
        "recall_No": pr_no[1] if pr_no else 0.0,
        "precisao_Yes": pr_yes[0] if pr_yes else 0.0,
        "recall_Yes": pr_yes[1] if pr_yes else 0.0,
        "L": L,
    }


def _linhas_da_fase1(df: pd.DataFrame, dir_saida: str | None = None) -> list[dict]:
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    X = df.drop(columns=[ALVO] + ATRIBUTOS_IGNORADOS)
    y = df[ALVO]

    linhas: list[dict] = []
    for fold, (idx_treino, idx_teste) in enumerate(skf.split(X, y)):
        X_treino, X_teste = X.iloc[idx_treino], X.iloc[idx_teste]
        y_treino, y_teste = y.iloc[idx_treino].to_numpy(), y.iloc[idx_teste].to_numpy()

        discretizador = MDLPDiscretizer().fit(X_treino, y_treino)
        X_treino_disc = discretizador.fit_transform(X_treino, y_treino)
        X_teste_disc = discretizador.transform(X_teste)

        # dummy_no: baseline always "No"
        modelo_dummy = DummyClassifier(strategy="constant", constant="No", random_state=RANDOM_STATE)
        modelo_dummy.fit(X_treino_disc, y_treino)
        y_pred_dummy = modelo_dummy.predict(X_teste_disc)
        linhas.append(_linha_metricas("dummy_no", fold, y_teste, y_pred_dummy, L=0))

        # xgboost: busca nested dentro do fold de treino (sem leakage)
        y_pred_xgb, melhor_pipeline = treinar_e_prever(
            X_treino_disc,
            y.iloc[idx_treino],
            X_teste_disc,
            retornar_pipeline=True,
        )
        linhas.append(_linha_metricas("xgboost", fold, y_teste, y_pred_xgb, L=1))

        # fold de referência (fold 0): artefatos SHAP do melhor modelo
        if fold == 0 and dir_saida is not None:
            gerar_artefatos_shap(melhor_pipeline, X_teste_disc, Path(dir_saida))
    return linhas


def rodar_fase1(caminho_dataset: str, caminho_saida: str) -> None:
    arq_dataset, arq_saida = Path(caminho_dataset), Path(caminho_saida)
    if not arq_dataset.exists():
        raise FileNotFoundError(MENSAGEM_DATASET_AUSENTE.format(caminho=caminho_dataset))

    df = carregar_dataset(caminho_dataset).dropna(subset=["TotalCharges"]).reset_index(drop=True)
    linhas = _linhas_da_fase1(df, dir_saida=str(arq_saida.parent))

    folds = pd.DataFrame(linhas)
    blocos_medias = []
    for modelo in MODELOS_FASE1:
        bloco = folds[folds["modelo"] == modelo]
        media = {c: round(float(bloco[c].mean()), 6) for c in COLUNAS_CSV if c not in ("modelo", "fold")}
        blocos_medias.append({"modelo": modelo, "fold": "média", **media})

    arq_saida.parent.mkdir(parents=True, exist_ok=True)
    out = pd.concat([folds, pd.DataFrame(blocos_medias)], ignore_index=True)[COLUNAS_CSV]
    out.to_csv(arq_saida, index=False, float_format="%.6f")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Runner do benchmark AC.RankAW Churn")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], required=True, help="Fase do benchmark")
    args = parser.parse_args(argv)

    if args.phase != 1:
        print(f"Fase {args.phase} ainda não implementada neste ticket (planejada em issues futuras).")
        return 2

    try:
        rodar_fase1(caminho_dataset="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
                    caminho_saida="outputs/benchmark_results.csv")
    except FileNotFoundError as exc:
        print(str(exc))
        return 1
    print("Fase 1 concluída: outputs/benchmark_results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
