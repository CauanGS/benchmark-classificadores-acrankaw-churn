"""Runner CLI do benchmark AC.RankAW Churn.

Uso: python -m src.run_benchmark --phase 1|2|3

Protocolo comum às fases: benchmark sobre o Telco Churn com
StratifiedKFold (10 dobras, shuffle, random_state=42), discretização MDLP por
fold de treino e registro das métricas em CSV. O protocolo (constantes,
dobras, discretização e métricas) vive em `src/protocolo.py`, compartilhado
com o export da Fase 3 (`src/rule_export.py`).

**Semântica das fases** (coerente com o SDD — §1.2 objetivos e §2 pipeline):

- **--phase 1** — *Benchmark Inicial*: compara o baseline caixa-preta
  (`dummy_no` + `xgboost`, ramo 1 "Dados Contínuos/Dummies" do SDD §2) com
  os classificadores associativos da linha de base (`cba` e `ac_ranka`, ramo 2
  "Dados Categóricos Discretizados"). Escreve
  `outputs/benchmark_results.csv` e gera os artefatos SHAP do fold 0.
- **--phase 2** — *Aplicação do AC.RankAW Ponderado*: roda o método proposto
  (`[GF][PM-SD][TS]`) e mede a redução de regras (ΔL) frente ao `cba`
  (referência do SDD 5.2) e frente ao `ac_ranka`. Escreve
  `outputs/benchmark_results_phase2.csv` e **não** reexecuta a busca *nested*
  do XGBoost (o baseline já foi medido na Fase 1) nem os artefatos SHAP.
  Os modelos `cba` e `ac_ranka` são reavaliados neste run porque são as
  referências do ΔL (mesmo fold, mesma discretização).
- **--phase 3** — *Empacotamento & Portfólio*: exporta as regras de negócio
  do **AC.RankAW Ponderado** (JSON + SQL) — ver `src/rule_export.py`.

Os dois CSVs de benchmark têm o **mesmo schema** (mesmas colunas), para que
a leitura dos artefatos das fases 1 e 2 não quebre.

Caminhos configuráveis (padrões = layout do repositório): `--dataset` (CSV
do Telco Churn) e `--saida` (diretório dos artefatos).

Colunas de ΔL (SDD 5.2, preenchidas APENAS nas linhas de ac_rankaw):

    ΔL = (L_referência − L_AC.RankAW) / L_referência × 100%

- `delta_L`: referência = **CBA** (a referência do SDD 5.2).
- `delta_L_vs_acranka`: referência = **AC.RankA** (mesmo motor, pesos
  uniformes) — quantifica o efeito da ponderação `[PM-SD]` no tamanho.

Para ambas:

- **Por fold**: L da referência e L do AC.RankAW no próprio fold.
- **Bloco "média"**: ΔL recalculado a partir das MÉDIAS de L dos folds do
  run atual — ΔL é uma razão, e a média de uma razão não é a razão das
  médias; por isso NÃO é a média dos ΔL por fold.
- Linhas de outros modelos: célula vazia (NaN).

Colunas de desvio-padrão (`<métrica>_desvio`, tickets #4/#5 — "média ±
desvio"): o bloco "média" de cada modelo traz, além da média, o desvio
padrão amostral (ddof=1) das 10 dobras de cada métrica; as linhas de dobra
têm a célula vazia (NaN). Exceção documentada: `delta_L_desvio` é o desvio
dos ΔL **por dobra** (dispersão da razão), enquanto o `delta_L` do bloco
"média" é a razão das médias de L — grandezas diferentes por construção.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.protocolo import (
    MODELOS_ASSOCIATIVOS,
    RANDOM_STATE,
    Dobra,
    avaliar_associativos_dobra,
    carregar_dataset_churn,
    delta_l,
    features_e_alvo,
    frames_do_fold as _frames_do_fold,
    metricas_da_dobra,
    preparar_dobras,
)
from src.xgboost_model import gerar_artefatos_shap, treinar_e_prever

# Modelos de cada fase (SDD §1.2): a Fase 1 compara os baselines; a Fase 2
# aplica o método proposto e mede o ΔL contra as referências do mesmo run.
MODELOS_FASE_1 = ("dummy_no", "xgboost", "cba", "ac_ranka")
MODELOS_FASE_2 = ("cba", "ac_ranka", "ac_rankaw")
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
    "delta_L_vs_acranka",
]
# Métricas com desvio-padrão no bloco "média" (média ± desvio, tickets #4/#5).
METRICAS_COM_DESVIO = (
    "f1_macro",
    "precisao_No",
    "recall_No",
    "precisao_Yes",
    "recall_Yes",
    "L",
    "delta_L",
    "delta_L_vs_acranka",
)
COLUNAS_DESVIO = [f"{m}_desvio" for m in METRICAS_COM_DESVIO]


def _linha_metricas(modelo: str, fold, y_teste, y_pred, L: int,
                    delta_L: float | None = None,
                    delta_L_vs_acranka: float | None = None) -> dict:
    """Linha do CSV de uma dobra: métricas do protocolo + ΔL (quando houver)."""
    return {
        "modelo": modelo,
        "fold": fold,
        **metricas_da_dobra(y_teste, y_pred, L),
        "delta_L": float(delta_L) if delta_L is not None else np.nan,
        "delta_L_vs_acranka": (
            float(delta_L_vs_acranka) if delta_L_vs_acranka is not None else np.nan
        ),
    }


def frames_do_fold(
    X_treino: pd.DataFrame, y_treino, X_teste: pd.DataFrame
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Frames de entrada dos dois ramos do pipeline experimental (SDD §2).

    Seam público do runner; a implementação vive no protocolo comum
    (`src/protocolo.py`), compartilhado com o export da Fase 3.
    """
    return _frames_do_fold(X_treino, y_treino, X_teste)


def _linhas_do_benchmark(
    df: pd.DataFrame,
    modelos: tuple[str, ...],
    dir_saida: str | None = None,
) -> list[dict]:
    """Roda os modelos pedidos no benchmark de 10 dobras (uma linha por dobra).

    `modelos` define o conjunto de modelos do run (Fase 1 ou Fase 2); o
    XGBoost só entra na Fase 1 (a Fase 2 não reexecuta a busca *nested*).
    As dobras (split estratificado + MDLP por dobra) vêm do protocolo comum
    (`src/protocolo.py`).
    """
    X, y = features_e_alvo(df)
    dobras = preparar_dobras(X, y)
    associativos = {
        nome: MODELOS_ASSOCIATIVOS[nome]
        for nome in modelos
        if nome in MODELOS_ASSOCIATIVOS
    }

    linhas: list[dict] = []
    for dobra in dobras:
        linhas.extend(_linhas_do_ramo_continuo(dobra, modelos, dir_saida))
        # Classificadores associativos (ramo 2 — dados discretizados).
        linhas_associativas = avaliar_associativos_dobra(dobra, associativos)
        _aplica_delta_l(linhas_associativas)
        linhas.extend(linhas_associativas)
    return linhas


def _linhas_do_ramo_continuo(
    dobra: Dobra, modelos: tuple[str, ...], dir_saida: str | None
) -> list[dict]:
    """Linhas do ramo 1 do SDD §2 (dados contínuos/dummies): dummy e XGBoost."""
    linhas: list[dict] = []
    if "dummy_no" in modelos:
        # dummy_no: baseline que sempre responde "No" (atributos irrelevantes) — ramo 1
        modelo_dummy = DummyClassifier(strategy="constant", constant="No", random_state=RANDOM_STATE)
        modelo_dummy.fit(dobra.X_treino_cont, dobra.y_treino)
        y_pred_dummy = modelo_dummy.predict(dobra.X_teste_cont)
        linhas.append(_linha_metricas("dummy_no", dobra.fold, dobra.y_teste, y_pred_dummy, L=0))

    if "xgboost" in modelos:
        # xgboost: ramo 1 (atributos contínuos + dummies das categóricas),
        # busca nested dentro do fold de treino (sem leakage)
        y_pred_xgb, melhor_pipeline = treinar_e_prever(
            dobra.X_treino_cont,
            dobra.y_treino,
            dobra.X_teste_cont,
            retornar_pipeline=True,
        )
        linhas.append(_linha_metricas("xgboost", dobra.fold, dobra.y_teste, y_pred_xgb, L=1))

        # fold de referência (fold 0): artefatos SHAP do melhor modelo, no
        # frame do ramo 1 (contínuos/dummies)
        if dobra.fold == 0 and dir_saida is not None:
            gerar_artefatos_shap(melhor_pipeline, dobra.X_teste_cont, Path(dir_saida))
    return linhas


def _aplica_delta_l(linhas_associativas: list[dict]) -> None:
    """Preenche o ΔL do AC.RankAW contra o CBA e o AC.RankA do MESMO fold.

    O ΔL é calculado dentro da dobra (L da referência e L do AC.RankAW no
    próprio fold); as linhas dos demais modelos ficam vazias (NaN).
    """
    L_por_modelo = {linha["modelo"]: linha["L"] for linha in linhas_associativas}
    for linha in linhas_associativas:
        delta = np.nan
        delta_vs_acranka = np.nan
        if linha["modelo"] == "ac_rankaw":
            L_cba = L_por_modelo.get("cba")
            L_acranka = L_por_modelo.get("ac_ranka")
            if L_cba:
                delta = delta_l(L_cba, linha["L"])
            if L_acranka:
                delta_vs_acranka = delta_l(L_acranka, linha["L"])
        linha["delta_L"] = delta
        linha["delta_L_vs_acranka"] = delta_vs_acranka


def _escrever_csv(
    linhas: list[dict],
    modelos: tuple[str, ...],
    caminho_saida: str,
) -> None:
    """Monta o CSV de resultados: linhas de dobra + bloco "média" (média ± desvio)."""
    folds = pd.DataFrame(linhas)
    medias_L = folds.groupby("modelo")["L"].mean().to_dict()
    media_L_aw = medias_L.get("ac_rankaw")

    blocos_medias = []
    for modelo in modelos:
        bloco = folds[folds["modelo"] == modelo]
        media = {c: round(float(bloco[c].mean()), 6) for c in COLUNAS_CSV
                 if c not in ("modelo", "fold")}
        # média ± desvio: desvio padrão amostral (ddof=1) das dobras do run
        desvio = {
            f"{c}_desvio": round(float(bloco[c].std(ddof=1)), 6)
            for c in METRICAS_COM_DESVIO
        }
        if modelo == "ac_rankaw" and media_L_aw is not None:
            # ΔL da média ≠ média dos ΔL (razão não-linear): recalculado a
            # partir das médias de L do run atual (ver docstring do módulo).
            if "cba" in medias_L:
                media["delta_L"] = round(float(delta_l(medias_L["cba"], media_L_aw)), 6)
            if "ac_ranka" in medias_L:
                media["delta_L_vs_acranka"] = round(
                    float(delta_l(medias_L["ac_ranka"], media_L_aw)), 6
                )
        blocos_medias.append({"modelo": modelo, "fold": "média", **media, **desvio})

    folds = folds.assign(fold=folds["fold"].astype(str))
    for coluna in COLUNAS_DESVIO:
        folds[coluna] = np.nan
    arq_saida = Path(caminho_saida)
    arq_saida.parent.mkdir(parents=True, exist_ok=True)
    out = pd.concat([folds, pd.DataFrame(blocos_medias)], ignore_index=True)[
        COLUNAS_CSV + COLUNAS_DESVIO
    ]
    out.to_csv(arq_saida, index=False, float_format="%.6f")


def rodar_fase_1(
    caminho_dataset: str,
    caminho_saida: str,
    diretorio_shap: str | None = None,
) -> None:
    """**Fase 1 (Benchmark Inicial)** — dummy_no, xgboost, cba e ac_ranka.

    Escreve `caminho_saida` (CSV com linhas de dobra + bloco "média" com
    média ± desvio). `diretorio_shap`: gera os artefatos SHAP do fold 0
    (o frame do ramo 1 — contínuos/dummies). Extração/poda/TOPSIS por fold;
    pesos [PM-SD] recalculados por fold dentro de cada `fit` do AC.RankAW.
    """
    df = carregar_dataset_churn(caminho_dataset)
    linhas = _linhas_do_benchmark(df, MODELOS_FASE_1, dir_saida=diretorio_shap)
    _escrever_csv(linhas, MODELOS_FASE_1, caminho_saida)


def rodar_fase_2(caminho_dataset: str, caminho_saida: str) -> None:
    """**Fase 2 (AC.RankAW Ponderado)** — cba, ac_ranka e ac_rankaw + ΔL.

    Reavalia as referências do ΔL (CBA e AC.RankA) no mesmo run e mede o
    AC.RankAW Ponderado; **não** reexecuta a busca *nested* do XGBoost (o
    baseline da Fase 1) nem gera artefatos SHAP. Escreve `caminho_saida`.
    """
    df = carregar_dataset_churn(caminho_dataset)
    linhas = _linhas_do_benchmark(df, MODELOS_FASE_2, dir_saida=None)
    _escrever_csv(linhas, MODELOS_FASE_2, caminho_saida)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Runner do benchmark AC.RankAW Churn")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], required=True, help="Fase do benchmark")
    parser.add_argument(
        "--dataset",
        default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help="CSV do dataset Telco Customer Churn",
    )
    parser.add_argument(
        "--saida", default="outputs", help="Diretório de saída dos artefatos"
    )
    args = parser.parse_args(argv)
    dir_saida = Path(args.saida)

    try:
        if args.phase == 1:
            rodar_fase_1(
                caminho_dataset=args.dataset,
                caminho_saida=str(dir_saida / "benchmark_results.csv"),
                diretorio_shap=str(dir_saida),
            )
            print(
                f"Fase 1 concluída: {dir_saida / 'benchmark_results.csv'} "
                f"(modelos: {', '.join(MODELOS_FASE_1)})"
            )
        elif args.phase == 2:
            rodar_fase_2(
                caminho_dataset=args.dataset,
                caminho_saida=str(dir_saida / "benchmark_results_phase2.csv"),
            )
            print(
                f"Fase 2 concluída: {dir_saida / 'benchmark_results_phase2.csv'} "
                f"(modelos: {', '.join(MODELOS_FASE_2)})"
            )
        else:
            from src.rule_export import exportar_regras

            caminho_json = dir_saida / "rules_exported.json"
            caminho_sql = dir_saida / "rules_exported.sql"
            payload = exportar_regras(
                caminho_dataset=args.dataset,
                caminho_json=str(caminho_json),
                caminho_sql=str(caminho_sql),
            )
            meta = payload["metadados"]
            print(
                f"Fase 3 concluída: {caminho_json} e {caminho_sql} "
                f"(modelo: {meta['modelo']}, L={meta['L']}, "
                f"F1-Macro médio={meta['metricas']['f1_macro']})"
            )
    except FileNotFoundError as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
