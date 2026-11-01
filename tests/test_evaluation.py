"""Testes do módulo de avaliação (métricas) do benchmark."""

from __future__ import annotations

import pytest

from src.evaluation import f1_macro, precision_recall_por_classe


def test_f1_macro_valores_conhecidos():
    y_true = ["No", "No", "Yes", "Yes", "Yes"]
    y_pred = ["No", "Yes", "Yes", "Yes", "No"]
    esperado = (0.5 + 2 / 3) / 2
    assert f1_macro(y_true, y_pred) == pytest.approx(esperado)


def test_precision_recall_por_classe():
    y_true = ["No", "No", "Yes", "Yes"]
    y_pred = ["No", "Yes", "Yes", "Yes"]
    pr = precision_recall_por_classe(y_true, y_pred)
    assert pr["No"] == (pytest.approx(1.0), pytest.approx(0.5))
    assert pr["Yes"] == (pytest.approx(2 / 3), pytest.approx(1.0))


def test_predicao_perfeita():
    y_true = ["No", "No"]
    y_pred = ["No", "No"]
    pr = precision_recall_por_classe(y_true, y_pred)
    assert pr["No"] == (1.0, 1.0)
    assert f1_macro(["No", "No", "Yes", "Yes"], ["No", "No", "Yes", "Yes"]) == 1.0
