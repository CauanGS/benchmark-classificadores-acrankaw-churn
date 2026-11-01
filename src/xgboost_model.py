"""Integração do XGBoost ao benchmark — busca nested, encoding e SHAP.

Interface do slot de modelo caixa-preta usado pelo runner da fase 1
(**ramo 1 do SDD §2 — "Dados Contínuos/Dummies"**):

- `criar_pipeline`: Pipeline OneHotEncoder + XGBClassifier. As features chegam
  no frame **ORIGINAL** (sem `customerID`): as numéricas como estão e as
  categóricas em texto — o one-hot vive dentro do pipeline interno, então a
  busca nunca vê o fold de teste. A discretização MDLP é exclusiva do ramo 2
  (classificadores associativos).
- `buscar_hiperparams`: RandomizedSearchCV (30 iters) com CV interna
  StratifiedKFold (5 folds, shuffle, seed 42), scoring `f1_macro`.
- `treinar_e_prever`: busca + predição nas classes originais, com encoding
  binário consistente [No, Yes] -> [0, 1].
- `gerar_artefatos_shap`: PNGs legíveis em PT por classe — beeswarm,
  bar mean|SHAP| e top-10 pares feature-valor — em `outputs/shap/`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42
CLASSE_POSITIVA = "Yes"
CLASSE_NEGATIVA = "No"
CODIFICACAO = {CLASSE_NEGATIVA: 0, CLASSE_POSITIVA: 1}
DECODIFICACAO = {0: CLASSE_NEGATIVA, 1: CLASSE_POSITIVA}

ESPACO_HIPERPARAMS = {
    "xgb__n_estimators": randint(100, 400),
    "xgb__max_depth": randint(2, 9),
    "xgb__learning_rate": uniform(0.01, 0.29),
    "xgb__subsample": uniform(0.6, 0.4),
    "xgb__colsample_bytree": uniform(0.6, 0.4),
    "xgb__min_child_weight": randint(1, 6),
}


def criar_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """Pipeline one-hot (categóricas) + numéricas intactas + XGBoost binário.

    Ramo 1 do SDD §2 ("Dados Contínuos/Dummies"): as categóricas viram
    dummies (one-hot dentro do pipeline, para não vazar o fold de teste) e as
    numéricas entram **como estão** (o `OneHotEncoder` do scikit-learn
    codificaria todas as colunas, inclusive as numéricas — por isso o
    `ColumnTransformer` separa os dois grupos por dtype).
    """
    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                make_column_selector(dtype_include=["object", "category", "bool"]),
            ),
            (
                "num",
                "passthrough",
                make_column_selector(dtype_exclude=["object", "category", "bool"]),
            ),
        ],
        verbose_feature_names_out=False,
    )
    modelo = xgb.XGBClassifier(
        objective="binary:logistic",
        random_state=random_state,
        n_jobs=1,
        eval_metric="logloss",
        tree_method="hist",
    )
    return Pipeline([("pre", pre), ("xgb", modelo)])


def buscar_hiperparams(
    pipeline: Pipeline,
    X_treino: pd.DataFrame,
    y_treino: np.ndarray,
    n_iter: int = 30,
    cv_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> RandomizedSearchCV:
    """RandomizedSearch nested — só vê o fold de treino externo.
    Encoding do alvo: [No, Yes] -> [0, 1]."""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    busca = RandomizedSearchCV(
        pipeline,
        param_distributions=ESPACO_HIPERPARAMS,
        n_iter=n_iter,
        cv=cv,
        scoring="f1_macro",
        random_state=random_state,
        refit=True,
        n_jobs=1,
    )
    y_num = np.where(np.asarray(y_treino) == CLASSE_POSITIVA, 1, 0)
    busca.fit(X_treino, y_num)
    return busca


def treinar_e_prever(
    X_treino: pd.DataFrame,
    y_treino: pd.Series,
    X_pred: pd.DataFrame,
    n_iter: int = 30,
    cv_splits: int = 5,
    random_state: int = RANDOM_STATE,
    retornar_pipeline: bool = False,
) -> np.ndarray | tuple[np.ndarray, Pipeline]:
    """Busca nested no treino; predição nas classes originais ("No"/"Yes").

    Com `retornar_pipeline=True`, devolve também o melhor pipeline ajustado
    (usado pelo runner para gerar os artefatos SHAP do fold de referência).
    """
    busca = buscar_hiperparams(criar_pipeline(random_state), X_treino, y_treino, n_iter, cv_splits, random_state)
    y_pred_num = busca.best_estimator_.predict(X_pred)
    y_pred = np.asarray([DECODIFICACAO[int(v)] for v in y_pred_num], dtype=object)
    return (y_pred, busca.best_estimator_) if retornar_pipeline else y_pred


def _matriz_shap(pipeline: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Valores SHAP (classe positiva, log-odds), matriz de entrada e nomes."""
    pre = pipeline.named_steps["pre"]
    X_cod = pre.transform(X)
    if hasattr(X_cod, "toarray"):
        X_cod = X_cod.toarray()
    nomes = list(pre.get_feature_names_out())
    explainer = shap.TreeExplainer(pipeline.named_steps["xgb"])
    return explainer.shap_values(X_cod), X_cod, nomes


def _salvar(fig: plt.Figure, caminho: Path) -> None:
    fig.savefig(caminho, dpi=160, bbox_inches="tight")
    plt.close(fig)


def gerar_artefatos_shap(pipeline: Pipeline, X: pd.DataFrame, dir_saida: Path) -> None:
    """PNGs de SHAP por classe (positiva e negativa) em `dir_saida`/"shap"/."""
    dir_alvo = Path(dir_saida) / "shap"
    dir_alvo.mkdir(parents=True, exist_ok=True)

    shap_vals, X_cod, nomes = _matriz_shap(pipeline, X)
    media_abs = np.abs(shap_vals).mean(axis=0)

    for classe, sinal in [(CLASSE_POSITIVA, 1.0), (CLASSE_NEGATIVA, -1.0)]:
        vals_classe = sinal * shap_vals  # binário: classe oposta = sinal invertido
        sufixo = f"classe_{classe}"

        # 1) Beeswarm por classe
        shap.summary_plot(vals_classe, X_cod, feature_names=nomes, show=False)
        plt.gcf().suptitle(f"Beeswarm SHAP — classe {classe}", fontsize=12)
        _salvar(plt.gcf(), dir_alvo / f"beeswarm_{sufixo}.png")
        plt.close("all")

        # 2) Bar mean |SHAP| por classe
        ordem = np.argsort(media_abs)[::-1]
        fig = plt.figure(figsize=(8, 5))
        plt.barh([nomes[i] for i in ordem[::-1]], media_abs[ordem[::-1]], color="#1f77b4")
        plt.xlabel("média |valor SHAP| (importância)")
        plt.title(f"Importância mean|SHAP| — classe {classe}")
        plt.tight_layout()
        _salvar(fig, dir_alvo / f"bar_mean_shap_{sufixo}.png")

        # 3) Top-10 pares feature-valor por classe
        contrib = np.abs(vals_classe).mean(axis=0)
        ordem_pares = np.argsort(contrib)[::-1][:10][::-1]
        pares = []
        for i in ordem_pares:
            nome = nomes[i]
            base = nome.split("_", 1)
            if len(base) == 1:
                # coluna numérica do ramo 1 (sem dummy): não há par valor
                pares.append(nome)
            else:
                pares.append(f"{base[0]}={base[1]}")
        fig = plt.figure(figsize=(9, 5))
        plt.barh(pares, contrib[ordem_pares], color="#d62728")
        plt.xlabel("média |valor SHAP| (par feature=valor)")
        plt.title(f"Top 10 pares feature=valor — classe {classe}")
        plt.tight_layout()
        _salvar(fig, dir_alvo / f"top10_pares_{sufixo}.png")
