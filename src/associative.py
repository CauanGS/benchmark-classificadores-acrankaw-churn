"""Classificador CBA (Liu et al. 1998) — precedência, poda M1 e predição [O].

- Precedência CBA: confiança ↓, suporte ↓, cardinalidade do antecedente ↑.
- Poda [M1] varre as regras nessa ordem mantendo as transações vivas:
  uma regra entra no classificador se acertar (classificar corretamente)
  pelo menos uma transação viva dentre as que ela cobre; as transações
  dentre as cobertas que ela ACERTA são deletadas (escolha do protocolo
  AC.RankAW — as cobertas com erro permanecem vivas para regras seguintes).
- Classe default = maioria das transações ainda vivas após o esgotamento
  (empate resolvido pelo rótulo mais comum… empate: menor rótulo em ordem
  alfabética → determinismo).
- Predição [O]: classe da primeira regra do ranking que casa com a
  instância; fallback para a classe default.
- L = nº de regras pós-poda + 1 (a classe default).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.rule_extraction import MAX_REGRAS, extrair_regras


def _itens_da_linha(X: pd.DataFrame) -> list[frozenset[str]]:
    return [frozenset(f"{col}={str(X[col].iloc[i])}" for col in X.columns) for i in range(len(X))]


def ordenar_precedencia(regras: list[tuple]) -> list[tuple]:
    """Ordena regras pela precedência CBA (conf ↓, sup ↓, antecedente menor)."""
    return sorted(
        regras,
        key=lambda r2: (-round(r2[3], 12), -round(r2[2], 12), len(r2[0]),
                        tuple(sorted(r2[0])), r2[1]),
    )


def poda_m1(regras_ordenadas, transacoes) -> tuple[list[tuple], str]:
    """Poda cover-style [M1] sobre `(antecedente, classe, sup, conf)` ordenadas.

    transacoes : lista [(itens_atributos: frozenset, classe: str), ...]
    Retorno: (regras mantidas na ordem, classe default do restante).
    """
    vivas = list(range(len(transacoes)))
    mantidas: list[tuple] = []
    for regra in regras_ordenadas:
        antecedente, classe = regra[0], regra[1]
        acertos = [
            i for i in vivas
            if antecedente <= transacoes[i][0] and transacoes[i][1] == classe
        ]
        if acertos:
            mantidas.append(regra)
            conjunto = set(acertos)
            vivas = [i for i in vivas if i not in conjunto]
    if vivas:
        contagens: dict[str, int] = {}
        for i in vivas:
            contagens.setdefault(transacoes[i][1], 0)
            contagens[transacoes[i][1]] += 1
        classe_default = sorted(contagens.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    else:
        contagens: dict[str, int] = {}  # algo sempre existe em treino
        for _, classe in transacoes:
            contagens.setdefault(classe, 0)
            contagens[classe] += 1
        classe_default = sorted(contagens.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return mantidas, classe_default


def matriz_gf(regras: list[tuple], X_disc: pd.DataFrame, y) -> pd.DataFrame:
    """Matriz regras × 13 MOs [GF] (linhas na ordem de entrada de `regras`).

    A contingência de cada regra é calculada sobre as N instâncias
    discretizadas de `X_disc` (fold de treino): n11 = cobertas com a classe
    correta, n10 = cobertas com classe errada, n01/n00 = não cobertas.
    """
    from src.objective_measures import MEDIDAS_GF, medidas_gf

    y_arr = np.asarray(y)
    itens_por_instancia = _itens_da_linha(X_disc)
    itens_unicos = sorted({it for t in itens_por_instancia for it in t})
    mascara_item: dict[str, np.ndarray] = {}
    for it in itens_unicos:
        mascara_item[it] = np.array([it in t for t in itens_por_instancia], dtype=bool)
    classes = np.array([str(c) for c in y_arr])
    classes_unicas = sorted(set(classes))

    linhas = []
    for antecedente, classe, *_ in regras:
        cobre = np.all([mascara_item[it] for it in sorted(antecedente)], axis=0) \
            if antecedente else np.ones(len(itens_por_instancia), dtype=bool)
        da_classe = classes == str(classe)
        n11 = int((cobre & da_classe).sum())
        n10 = int((cobre & ~da_classe).sum())
        n01 = int((~cobre & da_classe).sum())
        n00 = len(classes) - n11 - n10 - n01
        linhas.append(medidas_gf(n11, n10, n01, n00))
    return pd.DataFrame(linhas, columns=list(MEDIDAS_GF))


class CBA:
    """Classification Based on Association Rules (poda M1, predição [O])."""

    def __init__(self, sup_min: float | None = None, max_regras: int = MAX_REGRAS):
        self.sup_min = sup_min
        self.max_regras = max_regras

    def fit(self, X: pd.DataFrame, y) -> "CBA":
        y_arr = [str(v) for v in y]
        regras = extrair_regras(X, y_arr, sup_min=self.sup_min, max_regras=self.max_regras)
        ordenadas = ordenar_precedencia(regras)
        transacoes = list(zip(_itens_da_linha(X), y_arr))
        self.regras_, self.classe_default_ = poda_m1(ordenadas, transacoes)
        return self

    @property
    def L(self) -> int:
        """Tamanho do modelo: regras pós-poda + 1 (classe default)."""
        return len(self.regras_) + 1

    def predict(self, X: pd.DataFrame) -> list[str]:
        """Predição [O]: primeira regra do ranking que casa; else default."""
        preds = []
        for itens in _itens_da_linha(X):
            pred = None
            for antecedente, classe, *_ in self.regras_:
                if antecedente <= itens:
                    pred = classe
                    break
            preds.append(pred if pred is not None else self.classe_default_)
        return preds
