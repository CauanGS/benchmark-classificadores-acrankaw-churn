"""Runner CLI do benchmark AC.RankAW Churn.

Uso: python -m src.run_benchmark --phase 1|2|3

Pipeline comum às fases 1 e 2: benchmark dos modelos sobre o Telco Churn com
StratifiedKFold (10 folds, shuffle, random_state=42), discretização MDLP por
fold de treino e registro das métricas em CSV. Nesta etapa os modelos
registrados são dummy_no, xgboost (L=1), cba, ac_ranka e ac_rankaw
(`[GF][PM-SD][TS]` — pesos por desvio padrão, Algoritmo 1 do relatório).

Semântica das fases (decisão de interface do Ticket 7):
- **--phase 1**: pipeline completa, escrevendo `outputs/benchmark_results.csv`
  e gerando artefatos SHAP do fold 0.
- **--phase 2**: MESMA pipeline (todos os modelos; sem skip de dummy/xgboost
  para simplificar o runner e o CSV harmonizado), a diferença principal é o
  relatório de ΔL colunado em saída dedicada
  (`outputs/benchmark_results_phase2.csv`), sem regenerar artefatos SHAP.
  A explicação ao usuário final do portfólio fica no Ticket 8.

Coluna `delta_L` (SDD 5.2, preenchida APENAS nas linhas de ac_rankaw):

    ΔL = (L_CBA − L_AC.RankAW) / L_CBA × 100%

- **Por fold**: L do CBA e do AC.RankAW no próprio fold.
- **Bloco "média"**: ΔL recalculado a partir das MÉDIAS de L dos folds do
  run atual — ΔL é uma razão, e a média de uma razão não é a razão das
  médias; por isso NÃO é a média dos ΔL por fold.
- Linhas de outros modelos: célula vazia (NaN).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold

from src.associative import ACRankA, ACRankAW, CBA
from src.eda import ALVO, carregar_dataset
from src.evaluation import f1_macro, precision_recall_por_classe
from src.preprocessing import MDLPDiscretizer
from src.xgboost_model import gerar_artefatos_shap, treinar_e_prever

ATRIBUTOS_IGNORADOS = ["customerID"]
RANDOM_STATE = 42
MODELOS_BENCHMARK = ("dummy_no", "xgboost", "cba", "ac_ranka", "ac_rankaw")
COLUNAS_CSV = [
    "modelo",
    "fold",
    "f1_macro",
    "precisao_No",
    "recall_No",
    "precisao_Yes",
    "recall_Yes",
    "L",
    "delta_L",
]
MENSAGEM_DATASET_AUSENTE = """\
Dataset não encontrado em {caminho}.
Baixe o Telco Customer Churn com:

    python -c "import kagglehub; kagglehub.dataset_download('blastchar/telco-customer-churn')"

e copie o CSV 'WA_Fn-UseC_-Telco-Customer-Churn.csv' para data/raw/.
"""


def delta_l(L_cba: int | float, L_aw: int | float) -> float:
    """ΔL (SDD 5.2): (L_CBA − L_AC.RankAW)/L_CBA × 100.

    Compara a média de L das 2 abordagens (poda M1 resultante); L_CBA ≥ 1
    sempre (classe default), sem divisão por zero.
    """
    return (L_cba - L_aw) / L_cba * 100.0


def _linha_metricas(modelo: str, fold, y_teste, y_pred, L: int,
                    delta_L: float | None = None) -> dict:
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
        "delta_L": float(delta_L) if delta_L is not None else np.nan,
    }


def _linhas_do_benchmark(df: pd.DataFrame, dir_saida: str | None = None) -> list[dict]:
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

        # cba: extração de regras de associação sobre o fold de treino
        modelo_cba = CBA()
        modelo_cba.fit(X_treino_disc, y_treino)
        y_pred_cba = modelo_cba.predict(X_teste_disc)
        linhas.append(_linha_metricas("cba", fold, y_teste, y_pred_cba, L=int(modelo_cba.L)))

        # ac_ranka: idem cba, ordenando as regras por TOPSIS com pesos uniformes
        modelo_acranka = ACRankA()
        modelo_acranka.fit(X_treino_disc, y_treino)
        y_pred_acranka = modelo_acranka.predict(X_teste_disc)
        linhas.append(_linha_metricas("ac_ranka", fold, y_teste, y_pred_acranka,
                                      L=int(modelo_acranka.L)))

        # ac_rankaw: idem ac_ranka, pesos [PM-SD] recalculados por fold; ΔL
        # é contra o CBA do MESMO fold (médias dentro do fold, não no run)
        modelo_acrankaw = ACRankAW()
        modelo_acrankaw.fit(X_treino_disc, y_treino)
        y_pred_acrankaw = modelo_acrankaw.predict(X_teste_disc)
        linhas.append(_linha_metricas("ac_rankaw", fold, y_teste, y_pred_acrankaw,
                                      L=int(modelo_acrankaw.L),
                                      delta_L=delta_l(int(modelo_cba.L), int(modelo_acrankaw.L))))

        # fold de referência (fold 0): artefatos SHAP do melhor modelo
        if fold == 0 and dir_saida is not None:
            gerar_artefatos_shap(melhor_pipeline, X_teste_disc, Path(dir_saida))
    return linhas


def rodar_benchmark(caminho_dataset: str, caminho_saida: str,
                    diretorio_shap: str | None = None) -> None:
    """Pipeline comum às fases 1 e 2 (ver docstring do módulo).

    `diretorio_shap`: gera artefatos SHAP do fold 0 (fase 1) ou None (fase 2,
    que só registra o CSV com a coluna ΔL). Extração/poda/TOPSIS por fold;
    pesos [PM-SD] recalculados por fold dentro de cada `fit` do AC.RankAW.
    """
    arq_dataset, arq_saida = Path(caminho_dataset), Path(caminho_saida)
    if not arq_dataset.exists():
        raise FileNotFoundError(MENSAGEM_DATASET_AUSENTE.format(caminho=caminho_dataset))

    df = carregar_dataset(caminho_dataset).dropna(subset=["TotalCharges"]).reset_index(drop=True)
    linhas = _linhas_do_benchmark(df, dir_saida=diretorio_shap)

    folds = pd.DataFrame(linhas)
    media_L_cba = folds[folds["modelo"] == "cba"]["L"].mean()
    media_L_aw = folds[folds["modelo"] == "ac_rankaw"]["L"].mean()
    delta_L_medio = delta_l(media_L_cba, media_L_aw)

    blocos_medias = []
    for modelo in MODELOS_BENCHMARK:
        bloco = folds[folds["modelo"] == modelo]
        media = {c: round(float(bloco[c].mean()), 6) for c in COLUNAS_CSV
                 if c not in ("modelo", "fold")}
        if modelo == "ac_rankaw":
            # ΔL da média ≠ média dos ΔL (razão não-linear): recalculado a
            # partir das médias de L do run atual (ver docstring do módulo).
            media["delta_L"] = round(float(delta_L_medio), 6)
        blocos_medias.append({"modelo": modelo, "fold": "média", **media})

    folds = folds.assign(fold=folds["fold"].astype(str))
    arq_saida.parent.mkdir(parents=True, exist_ok=True)
    out = pd.concat([folds, pd.DataFrame(blocos_medias)], ignore_index=True)[COLUNAS_CSV]
    out.to_csv(arq_saida, index=False, float_format="%.6f")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Runner do benchmark AC.RankAW Churn")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], required=True, help="Fase do benchmark")
    args = parser.parse_args(argv)

    try:
        if args.phase == 1:
            rodar_benchmark(
                caminho_dataset="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
                caminho_saida="outputs/benchmark_results.csv",
                diretorio_shap="outputs",
            )
            print("Fase 1 concluída: outputs/benchmark_results.csv")
        elif args.phase == 2:
            rodar_benchmark(
                caminho_dataset="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
                caminho_saida="outputs/benchmark_results_phase2.csv",
                diretorio_shap=None,
            )
            print("Fase 2 concluída: outputs/benchmark_results_phase2.csv")
        else:
            print(f"Fase {args.phase} ainda não implementada neste ticket (planejada em issues futuras).")
            return 2
    except FileNotFoundError as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
