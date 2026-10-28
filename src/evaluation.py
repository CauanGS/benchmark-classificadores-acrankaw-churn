"""Avaliação de classificadores — métricas reutilizáveis do benchmark.

Encapsula sklearn para que os classificadores (dummy, CBA, AC.RankAW...)
usem a mesma interface de cálculo: F1-Macro e precisão/recall por classe.

Convenção do benchmark: classe positiva alvo = "Yes" (churn); F1-Macro
média não-ponderada das classes, como no paper AC.RankA.
"""

from __future__ import annotations

from sklearn.metrics import f1_score, precision_recall_fscore_support


def f1_macro(y_true: list, y_pred: list) -> float:
    """F1-Macro (média não-ponderada das classes)."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def precision_recall_por_classe(y_true: list, y_pred: list) -> dict[str, tuple[float, float]]:
    """Precisão e recall por classe: {classe: (precisão, recall)}."""
    classes = sorted(set(y_true) | set(y_pred))
    p, r, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    return {c: (float(pi), float(ri)) for c, pi, ri in zip(classes, p, r)}
