"""Extração de regras de classe via Apriori (mlxtend) — protocolo do paper.

Protocolo (docs/DECISIONS.md Q10 / paper AC.RankA Seção de extração):
- transações = instância discretizada; item = "coluna=valor" (mutuamente
  exclusivos dentro de cada coluna); consequente = rótulo da classe.
- Apriori com sup_min calibrado por fold: inicia 0.10 e decai ×0.5 até o
  número de regras de classe ser ≤ `max_regras` (padrão 10.000).
- conf-min = 0 (todas as regras do sup-min); regras cujo antecedente tem
  no máximo `max_antecedente` itens (itemset de classe ≤ max + 1 itens).

As regras extraídas são do tipo nomeado `Regra` (`src/regra.py`); o score de
ordenação (`Regra.score`) é preenchido depois, pelo classificador.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
from mlxtend.frequent_patterns import apriori

from src.regra import Regra

SUP_MIN_INICIAL = 0.10
FATOR_DECAY = 0.5
MAX_REGRAS = 10_000
MAX_ANTECEDENTE = 4
MAX_CICLOS = 8  # guarda-chuva: 0.10 → 7 decaimentos ≈ 7.8e-4 mínimo


def _one_hot_transacoes(X: pd.DataFrame, y, prefixo_classe: str = "Churn=") -> pd.DataFrame:
    """Matriz booleana transação × item (itens de atributos + itens de classe)."""
    X2 = X.reset_index(drop=True)
    y_arr = pd.Series(y).reset_index(drop=True)
    dados: dict[str, "pd.Series[bool]"] = {}
    for col in X2.columns:
        vals = sorted(X2[col].astype(str).unique())
        col_str = X2[col].astype(str).to_numpy()
        for val in vals:
            dados[f"{col}={val}"] = (col_str == val).astype(bool)
    for classe in sorted(set(y_arr)):
        dados[f"{prefixo_classe}{classe}"] = (y_arr == classe).to_numpy().astype(bool)
    return pd.DataFrame(dados)


def extrair_regras(
    X: pd.DataFrame,
    y,
    sup_min: float | None = None,
    max_regras: int = MAX_REGRAS,
    max_antecedente: int = MAX_ANTECEDENTE,
) -> list[Regra]:
    """Extrai regras de classe (`Regra`: antecedente, classe, suporte, confiança).

    O antecedente é um frozenset de itens "coluna=valor" (1..max_antecedente
    itens); o consequente é o rótulo de classe. O suporte/confiança são
    relativos às N transações de `X`. Ordenação estável (por itemset).
    O `score` de ordenação fica `None` — é preenchido pelo classificador
    (TOPSIS no AC.RankA/AC.RankAW, confiança no CBA).
    """
    if sup_min is None:
        def contagem(s: float) -> int:
            return len(extrair_regras(X, y, sup_min=s, max_regras=max_regras, max_antecedente=max_antecedente))

        sup_min = suporte_minimo_calibrado(n_max_regras=max_regras, contagem=contagem)

    onehot = _one_hot_transacoes(X, y)
    itemsets = apriori(onehot.astype(bool), min_support=sup_min, use_colnames=True, max_len=max_antecedente + 1)
    suportes = {frozenset(s): p for s, p in zip(itemsets["itemsets"], itemsets["support"])}

    classes = sorted(set(pd.Series(y)))
    regras: list[Regra] = []
    for chave, sup in suportes.items():
        if len(chave) < 1:
            continue
        itens_classe = [it for it in chave if it.startswith("Churn=")]
        if len(itens_classe) != 1:
            continue
        classe = itens_classe[0].removeprefix("Churn=")
        if classe not in classes:
            continue
        antecedente = frozenset(chave - {itens_classe[0]})
        if not (1 <= len(antecedente) <= max_antecedente):
            continue
        sup_ant = suportes.get(antecedente)
        if sup_ant is None or sup_ant == 0:
            continue
        regras.append(Regra(antecedente, classe, float(sup), float(sup / sup_ant)))

    regras.sort(key=lambda r: (tuple(sorted(r.antecedente)), r.classe))
    return regras


def suporte_minimo_calibrado(
    n_max_regras: int,
    contagem: Callable[[float], int],
    sup_inicial: float = SUP_MIN_INICIAL,
    fator: float = FATOR_DECAY,
    max_ciclos: int = MAX_CICLOS,
) -> float:
    """Sup_min inicial 0.10 com decaimento ×0.5 até contagem ≤ n_max_regras.

    `contagem(sup_min)` retorna o nº de regras com esse suporte mínimo.
    Após `max_ciclos` decaimentos, aceita o último valor (regra "olhos
    fechados"): o limite-runtime importa menos do que terminar.
    """
    sup = sup_inicial
    retorno = sup
    for _ in range(max_ciclos):
        retorno = sup
        if contagem(sup) <= n_max_regras:
            return sup
        sup *= fator
    return retorno


__all__ = ["extrair_regras", "suporte_minimo_calibrado", "SUP_MIN_INICIAL"]
