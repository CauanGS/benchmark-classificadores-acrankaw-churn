"""Regra de associação — o tipo nomeado do domínio dos classificadores.

A regra é o átomo do classificador associativo: um **antecedente** (conjunto
de itens "coluna=valor"), a **classe** consequente e as métricas da extração
(**suporte** e **confianca**). O **score** é o número que justifica a posição
da regra no ranking — TOPSIS no AC.RankA/AC.RankAW, confiança no CBA — e
fica `None` enquanto a regra ainda não foi ordenada.

A regra é **imutável** (`frozen=True`) e, por isso, hasheável: serve de chave
para o dicionário de scores do classificador (`scores_`) e permite comparar
listas de regras por valor (determinismo do classificador).
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Regra:
    """Regra de classe `A ⇒ B` — campos nomeados (sem tupla mágica)."""

    antecedente: frozenset[str]
    classe: str
    suporte: float
    confianca: float
    score: float | None = None

    def cobre(self, itens: frozenset[str]) -> bool:
        """A regra casa com a transação `itens` (antecedente ⊆ itens)."""
        return self.antecedente <= itens

    def com_score(self, score: float) -> "Regra":
        """Cópia da regra com o score de ordenação preenchido."""
        return replace(self, score=float(score))
