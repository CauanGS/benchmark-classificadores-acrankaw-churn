"""Runner CLI do benchmark AC.RankAW Churn.

Uso: python -m src.run_benchmark --phase 1|2|3

Protocolo comum às fases: benchmark sobre o Telco Churn com
StratifiedKFold (10 folds, shuffle, random_state=42), discretização MDLP por
fold de treino e registro das métricas em CSV.

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
from sklearn.model_selection import StratifiedKFold

from src.associative import ACRankA, ACRankAW, CBA
from src.eda import ALVO, carregar_dataset
from src.evaluation import f1_macro, precision_recall_por_classe
from src.preprocessing import MDLPDiscretizer
from src.xgboost_model import gerar_artefatos_shap, treinar_e_prever

ATRIBUTOS_IGNORADOS = ["customerID"]
RANDOM_STATE = 42
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
MENSAGEM_DATASET_AUSENTE = """\
Dataset não encontrado em {caminho}.
Baixe o Telco Customer Churn com:

    python -c "import kagglehub; kagglehub.dataset_download('blastchar/telco-customer-churn')"

e copie o CSV 'WA_Fn-UseC_-Telco-Customer-Churn.csv' para data/raw/.
"""


def delta_l(L_referencia: int | float, L_aw: int | float) -> float:
    """ΔL (SDD 5.2): (L_referência − L_AC.RankAW)/L_referência × 100.

    Compara a média de L das 2 abordagens (poda M1 resultante); L_referência
    ≥ 1 sempre (classe default), sem divisão por zero. A referência é o CBA
    (SDD 5.2) ou o AC.RankA (comparação entre os dois ranqueamentos).
    """
    return (L_referencia - L_aw) / L_referencia * 100.0


def _linha_metricas(modelo: str, fold, y_teste, y_pred, L: int,
                    delta_L: float | None = None,
                    delta_L_vs_acranka: float | None = None) -> dict:
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
        "delta_L_vs_acranka": (
            float(delta_L_vs_acranka) if delta_L_vs_acranka is not None else np.nan
        ),
    }


def frames_do_fold(
    X_treino: pd.DataFrame, y_treino, X_teste: pd.DataFrame
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Frames de entrada dos dois ramos do pipeline experimental (SDD §2).

    - ``"continuos"`` — **ramo 1 (Dados Contínuos/Dummies)**: frame ORIGINAL
      (sem `customerID`), numéricas como estão e categóricas em texto (o
      one-hot das categóricas vive dentro do pipeline do XGBoost, para não
      vazar informação do fold de teste). É o frame do baseline caixa-preta.
    - ``"discretizados"`` — **ramo 2 (Dados Categóricos Discretizados)**: bins
      MDLP (Fayyad–Irani) ajustados **só** na dobra de treino (anti-leakage).
      É o frame dos classificadores associativos (CBA, AC.RankA, AC.RankAW).

    Devolve ``{"continuos": (X_treino, X_teste), "discretizados": (...)}``.
    """
    X_cont = X_treino.drop(columns=[c for c in ATRIBUTOS_IGNORADOS if c in X_treino.columns])
    X_cont_teste = X_teste.drop(columns=[c for c in ATRIBUTOS_IGNORADOS if c in X_teste.columns])
    discretizador = MDLPDiscretizer().fit(X_treino, y_treino)
    X_disc = discretizador.fit_transform(X_treino, y_treino)
    X_disc_teste = discretizador.transform(X_teste)
    return {
        "continuos": (X_cont, X_cont_teste),
        "discretizados": (X_disc, X_disc_teste),
    }


def _linhas_do_benchmark(
    df: pd.DataFrame,
    modelos: tuple[str, ...],
    dir_saida: str | None = None,
) -> list[dict]:
    """Roda os modelos pedidos no benchmark de 10 dobras (uma linha por dobra).

    `modelos` define o conjunto de modelos do run (Fase 1 ou Fase 2); o
    XGBoost só entra na Fase 1 (a Fase 2 não reexecuta a busca *nested*).
    """
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    X = df.drop(columns=[ALVO] + ATRIBUTOS_IGNORADOS)
    y = df[ALVO]

    linhas: list[dict] = []
    for fold, (idx_treino, idx_teste) in enumerate(skf.split(X, y)):
        X_treino, X_teste = X.iloc[idx_treino], X.iloc[idx_teste]
        y_treino, y_teste = y.iloc[idx_treino].to_numpy(), y.iloc[idx_teste].to_numpy()

        frames = frames_do_fold(X_treino, y_treino, X_teste)
        X_treino_cont, X_teste_cont = frames["continuos"]
        X_treino_disc, X_teste_disc = frames["discretizados"]

        melhor_pipeline = None
        if "dummy_no" in modelos:
            # dummy_no: baseline always "No" (features irrelevantes) — ramo 1
            modelo_dummy = DummyClassifier(strategy="constant", constant="No", random_state=RANDOM_STATE)
            modelo_dummy.fit(X_treino_cont, y_treino)
            y_pred_dummy = modelo_dummy.predict(X_teste_cont)
            linhas.append(_linha_metricas("dummy_no", fold, y_teste, y_pred_dummy, L=0))

        if "xgboost" in modelos:
            # xgboost: ramo 1 (atributos contínuos + dummies das categóricas),
            # busca nested dentro do fold de treino (sem leakage)
            y_pred_xgb, melhor_pipeline = treinar_e_prever(
                X_treino_cont,
                y.iloc[idx_treino],
                X_teste_cont,
                retornar_pipeline=True,
            )
            linhas.append(_linha_metricas("xgboost", fold, y_teste, y_pred_xgb, L=1))

        L_cba = L_acranka = None
        if "cba" in modelos:
            # cba: extração de regras de associação sobre o fold de treino (ramo 2)
            modelo_cba = CBA()
            modelo_cba.fit(X_treino_disc, y_treino)
            y_pred_cba = modelo_cba.predict(X_teste_disc)
            L_cba = int(modelo_cba.L)
            linhas.append(_linha_metricas("cba", fold, y_teste, y_pred_cba, L=L_cba))

        if "ac_ranka" in modelos:
            # ac_ranka: idem cba, ordenando as regras por TOPSIS com pesos uniformes
            modelo_acranka = ACRankA()
            modelo_acranka.fit(X_treino_disc, y_treino)
            y_pred_acranka = modelo_acranka.predict(X_teste_disc)
            L_acranka = int(modelo_acranka.L)
            linhas.append(_linha_metricas("ac_ranka", fold, y_teste, y_pred_acranka,
                                          L=L_acranka))

        if "ac_rankaw" in modelos:
            # ac_rankaw: idem ac_ranka, pesos [PM-SD] recalculados por fold; ΔL
            # é contra o CBA e o AC.RankA do MESMO fold (médias dentro do
            # fold, não no run)
            modelo_acrankaw = ACRankAW()
            modelo_acrankaw.fit(X_treino_disc, y_treino)
            y_pred_acrankaw = modelo_acrankaw.predict(X_teste_disc)
            linhas.append(_linha_metricas(
                "ac_rankaw", fold, y_teste, y_pred_acrankaw,
                L=int(modelo_acrankaw.L),
                delta_L=delta_l(L_cba, int(modelo_acrankaw.L)) if L_cba else None,
                delta_L_vs_acranka=(
                    delta_l(L_acranka, int(modelo_acrankaw.L)) if L_acranka else None
                ),
            ))

        # fold de referência (fold 0): artefatos SHAP do melhor modelo, no
        # frame do ramo 1 (contínuos/dummies)
        if fold == 0 and dir_saida is not None and melhor_pipeline is not None:
            gerar_artefatos_shap(melhor_pipeline, X_teste_cont, Path(dir_saida))
    return linhas


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


def _carregar(caminho_dataset: str) -> pd.DataFrame:
    """Dataset de churn validado (EDA) e sem `TotalCharges` nulo."""
    arq = Path(caminho_dataset)
    if not arq.exists():
        raise FileNotFoundError(MENSAGEM_DATASET_AUSENTE.format(caminho=caminho_dataset))
    return carregar_dataset(caminho_dataset).dropna(subset=["TotalCharges"]).reset_index(drop=True)


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
    df = _carregar(caminho_dataset)
    linhas = _linhas_do_benchmark(df, MODELOS_FASE_1, dir_saida=diretorio_shap)
    _escrever_csv(linhas, MODELOS_FASE_1, caminho_saida)


def rodar_fase_2(caminho_dataset: str, caminho_saida: str) -> None:
    """**Fase 2 (AC.RankAW Ponderado)** — cba, ac_ranka e ac_rankaw + ΔL.

    Reavalia as referências do ΔL (CBA e AC.RankA) no mesmo run e mede o
    AC.RankAW Ponderado; **não** reexecuta a busca *nested* do XGBoost (o
    baseline da Fase 1) nem gera artefatos SHAP. Escreve `caminho_saida`.
    """
    df = _carregar(caminho_dataset)
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
