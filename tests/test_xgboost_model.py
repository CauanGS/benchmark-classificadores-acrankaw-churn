"""Testes do modelo XGBoost (busca nested, determinismo, SHAP e integração)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.evaluation import f1_macro
from src.xgboost_model import (
    buscar_hiperparams,
    criar_pipeline,
    gerar_artefatos_shap,
    treinar_e_prever,
)


@pytest.fixture
def fixture_binaria() -> tuple[pd.DataFrame, pd.Series]:
    """Fixture sintética pequena e separável: Churn=Yes quando tenure é alto.

    Features em formato pós-discretização (strings de bins), como o
    MDLPDiscretizer produz, para exercitar o one-hot do pipeline interno.
    """
    rng = np.random.default_rng(42)
    n = 200
    tenure_bin = rng.choice(["(-∞, 5]", "(5, 20]", "(20, +∞]"], size=n, p=[0.3, 0.4, 0.3])
    contrato = rng.choice(["month", "year"], size=n, p=[0.5, 0.5])
    churn = np.where(
        (tenure_bin == "(20, +∞]") & (contrato == "month"), "Yes", "No"
    )
    X = pd.DataFrame({"tenure_bin": tenure_bin, "Contrato": contrato.astype(object)})
    y = pd.Series(churn, name="Churn")
    return X, y


def test_busca_nested_roda_e_retorna_melhores_params(fixture_binaria):
    X, y = fixture_binaria
    pipeline = criar_pipeline()
    busca = buscar_hiperparams(
        pipeline, X, y, n_iter=4, cv_splits=3, random_state=42
    )
    assert hasattr(busca, "best_params_")
    assert set(busca.best_params_) <= {
        "xgb__n_estimators",
        "xgb__max_depth",
        "xgb__learning_rate",
        "xgb__subsample",
        "xgb__colsample_bytree",
        "xgb__min_child_weight",
    }
    busca.best_estimator_.predict(X)
    best_score = max(s["mean_test_score"] for s in busca.cv_results_["params"]) if False else busca.best_score_
    assert 0.0 <= best_score <= 1.0


def test_treinar_e_prever_labels_consistentes(fixture_binaria):
    X, y = fixture_binaria
    y_pred = treinar_e_prever(X, y, X, n_iter=4, cv_splits=3)
    assert set(y_pred) <= {"No", "Yes"}


def test_determinismo_seed42(fixture_binaria):
    X, y = fixture_binaria
    p1 = treinar_e_prever(X, y, X, n_iter=4, cv_splits=3)
    p2 = treinar_e_prever(X, y, X, n_iter=4, cv_splits=3)
    assert list(p1) == list(p2)


def test_integracao_f1_macro_conhecido(fixture_binaria):
    """Com a regra conhecida (Yes = tenure alto & contrato month), o modelo
    aprendido pelo nested search deve recuperar f1_macro alto na própria base."""
    X, y = fixture_binaria
    y_regra = np.where(
        (X["tenure_bin"] == "(20, +∞]") & (X["Contrato"] == "month"), "Yes", "No"
    )
    y_pred = treinar_e_prever(X, y, X, n_iter=10, cv_splits=3)
    score = f1_macro(list(y), list(y_pred))
    assert score == pytest.approx(f1_macro(list(y), list(y_regra)), abs=0.15)
    assert score > 0.6


def test_shap_pngs_existem_e_nao_vazios(fixture_binaria, tmp_path):
    X, y = fixture_binaria
    pipeline = criar_pipeline()
    busca = buscar_hiperparams(pipeline, X, y, n_iter=2, cv_splits=3)
    gerar_artefatos_shap(busca.best_estimator_, X, tmp_path)
    esperados = [
        "beeswarm_classe_Yes.png",
        "beeswarm_classe_No.png",
        "bar_mean_shap_classe_Yes.png",
        "bar_mean_shap_classe_No.png",
        "top10_pares_classe_Yes.png",
        "top10_pares_classe_No.png",
    ]
    for nome in esperados:
        arq = tmp_path / "shap" / nome
        assert arq.exists(), f"faltou {nome}"
        assert arq.stat().st_size > 10_000, f"PNG vazio ou ilegível: {nome}"
