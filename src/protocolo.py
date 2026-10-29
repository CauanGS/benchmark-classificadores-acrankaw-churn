"""Protocolo comum do benchmark — dobras, discretização e métricas.

Fonte única das decisões de protocolo compartilhadas pelo runner
(`src/run_benchmark.py`) e pelo export (`src/rule_export.py`):

- as constantes do protocolo: `RANDOM_STATE` (seed 42), `N_SPLITS` (10
  dobras), `ATRIBUTOS_IGNORADOS` (`customerID`) e `MENSAGEM_DATASET_AUSENTE`;
- o split estratificado e a discretização MDLP ajustada **só** na dobra de
  treino (anti-leakage), com os dois ramos do pipeline do SDD §2
  ("Dados Contínuos/Dummies" e "Dados Categóricos Discretizados");
- o cálculo das métricas por dobra (F1-Macro, precisão/recall por classe, L)
  e a agregação (média ± desvio amostral, ddof=1);
- o registro dos classificadores associativos avaliados pelo protocolo
  (`MODELOS_ASSOCIATIVOS`).

As chaves de métrica são as mesmas nos dois consumidores: `L` é o tamanho do
modelo (nº de regras pós-poda + 1 classe default) — na média das dobras, é
a **média de L por dobra**; o export a publica no JSON como `L_medio_dobras`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.associative import ACRankA, ACRankAW, CBA
from src.eda import ALVO, carregar_dataset
from src.evaluation import f1_macro, precision_recall_por_classe
from src.preprocessing import MDLPDiscretizer

RANDOM_STATE = 42
N_SPLITS = 10
ATRIBUTOS_IGNORADOS = ["customerID"]
MENSAGEM_DATASET_AUSENTE = """\
Dataset não encontrado em {caminho}.
Baixe o Telco Customer Churn com:

    python -c "import kagglehub; kagglehub.dataset_download('blastchar/telco-customer-churn')"

e copie o CSV 'WA_Fn-UseC_-Telco-Customer-Churn.csv' para data/raw/.
"""
# Métricas por dobra (mesmas chaves em todos os consumidores do protocolo).
CAMPOS_METRICAS = (
    "f1_macro",
    "precisao_No",
    "recall_No",
    "precisao_Yes",
    "recall_Yes",
    "L",
)
# Classificadores associativos avaliados pelo protocolo (nome → classe).
MODELOS_ASSOCIATIVOS: dict[str, Callable[[], Any]] = {
    "cba": CBA,
    "ac_ranka": ACRankA,
    "ac_rankaw": ACRankAW,
}


def delta_l(L_referencia: int | float, L_aw: int | float) -> float:
    """ΔL (SDD 5.2): (L_referência − L_AC.RankAW)/L_referência × 100.

    Compara a média de L das 2 abordagens (poda M1 resultante); L_referência
    ≥ 1 sempre (classe default), sem divisão por zero. A referência é o CBA
    (SDD 5.2) ou o AC.RankA (comparação entre os dois ranqueamentos).
    """
    return (L_referencia - L_aw) / L_referencia * 100.0


@dataclass(frozen=True)
class Dobra:
    """Uma dobra do protocolo: índices, frames dos dois ramos e rótulos."""

    fold: int
    idx_treino: np.ndarray
    idx_teste: np.ndarray
    X_treino: pd.DataFrame
    X_teste: pd.DataFrame
    y_treino: pd.Series
    y_teste: pd.Series
    X_treino_cont: pd.DataFrame
    X_teste_cont: pd.DataFrame
    X_treino_disc: pd.DataFrame
    X_teste_disc: pd.DataFrame


# ------------------------------------------------------------------ dados


def carregar_dataset_churn(caminho_dataset: str) -> pd.DataFrame:
    """Dataset de churn já validado (EDA) e sem `TotalCharges` nulo."""
    caminho = Path(caminho_dataset)
    if not caminho.exists():
        raise FileNotFoundError(MENSAGEM_DATASET_AUSENTE.format(caminho=caminho_dataset))
    return carregar_dataset(caminho_dataset).dropna(subset=["TotalCharges"]).reset_index(drop=True)


def features_e_alvo(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """X (sem o alvo e sem os atributos ignorados) e y do dataset de churn."""
    return df.drop(columns=[ALVO] + ATRIBUTOS_IGNORADOS), df[ALVO]


# -------------------------------------------------------------- dobras


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


def preparar_dobras(
    X: pd.DataFrame,
    y,
    seed: int = RANDOM_STATE,
    n_splits: int = N_SPLITS,
) -> list[Dobra]:
    """As dobras do protocolo: split estratificado + MDLP por dobra.

    `StratifiedKFold(n_splits, shuffle=True, random_state=seed)` — a ordem das
    dobras e o conteúdo de cada uma são os do protocolo do benchmark (seed 42).
    A discretização MDLP é ajustada **só** na dobra de treino (anti-leakage).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dobras: list[Dobra] = []
    for fold, (idx_treino, idx_teste) in enumerate(skf.split(X, y)):
        X_treino, X_teste = X.iloc[idx_treino], X.iloc[idx_teste]
        y_treino, y_teste = y.iloc[idx_treino], y.iloc[idx_teste]
        frames = frames_do_fold(X_treino, y_treino, X_teste)
        X_treino_cont, X_teste_cont = frames["continuos"]
        X_treino_disc, X_teste_disc = frames["discretizados"]
        dobras.append(
            Dobra(
                fold=fold,
                idx_treino=idx_treino,
                idx_teste=idx_teste,
                X_treino=X_treino,
                X_teste=X_teste,
                y_treino=y_treino,
                y_teste=y_teste,
                X_treino_cont=X_treino_cont,
                X_teste_cont=X_teste_cont,
                X_treino_disc=X_treino_disc,
                X_teste_disc=X_teste_disc,
            )
        )
    return dobras


# ------------------------------------------------------------- métricas


def metricas_da_dobra(y_teste, y_pred, L: int) -> dict[str, float]:
    """Métricas de uma dobra: F1-Macro, precisão/recall por classe e L."""
    pr = precision_recall_por_classe(list(y_teste), list(y_pred))
    pr_no, pr_yes = pr.get("No"), pr.get("Yes")
    return {
        "f1_macro": f1_macro(list(y_teste), list(y_pred)),
        "precisao_No": pr_no[0] if pr_no else 0.0,
        "recall_No": pr_no[1] if pr_no else 0.0,
        "precisao_Yes": pr_yes[0] if pr_yes else 0.0,
        "recall_Yes": pr_yes[1] if pr_yes else 0.0,
        "L": int(L),
    }


def agregar_dobras(linhas: list[dict]) -> dict[str, dict]:
    """Agrega as linhas por modelo: média, desvio amostral e séries por dobra.

    `L` na média é a média de L por dobra (o export a publica no JSON como
    `L_medio_dobras`); `L_folds` traz o L de cada dobra.
    """
    por_modelo: dict[str, list[dict]] = {}
    for linha in linhas:
        por_modelo.setdefault(linha["modelo"], []).append(linha)
    metricas: dict[str, dict] = {}
    for nome, dobras in por_modelo.items():
        medias = {
            campo: float(np.mean([d[campo] for d in dobras]))
            for campo in CAMPOS_METRICAS
        }
        metricas[nome] = {
            **medias,
            "f1_macro_desvio": float(np.std([d["f1_macro"] for d in dobras], ddof=1)),
            "f1_macro_folds": [round(float(d["f1_macro"]), 6) for d in dobras],
            "L_folds": [int(d["L"]) for d in dobras],
        }
    return metricas


def avaliar_associativos_dobra(
    dobra: Dobra, modelos: Mapping[str, Callable[[], Any]]
) -> list[dict]:
    """Métricas dos classificadores associativos em UMA dobra do protocolo."""
    linhas: list[dict] = []
    for nome, fabrica in modelos.items():
        modelo = fabrica().fit(dobra.X_treino_disc, dobra.y_treino)
        y_pred = modelo.predict(dobra.X_teste_disc)
        linhas.append(
            {
                "modelo": nome,
                "fold": dobra.fold,
                **metricas_da_dobra(dobra.y_teste, y_pred, modelo.L),
            }
        )
    return linhas


def avaliar_associativos(
    dobras: list[Dobra], modelos: Mapping[str, Callable[[], Any]]
) -> tuple[list[dict], dict[str, dict]]:
    """Roda as dobras do protocolo para os classificadores associativos.

    Devolve `(linhas por dobra e modelo, métricas agregadas por modelo)`.
    """
    linhas: list[dict] = []
    for dobra in dobras:
        linhas.extend(avaliar_associativos_dobra(dobra, modelos))
    return linhas, agregar_dobras(linhas)
