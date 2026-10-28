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

AC.RankA (Dall'Agnol & Carvalho 2024) reutiliza o mesmo motor, trocando
apenas a ordenação: TOPSIS (pymcdm, normalização vetorial pesada — Eq. 3 do
paper) sobre a matriz regras × 13 MOs [GF] com pesos uniformes; empate no
score cai na ordem de extração (determinismo).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pymcdm import normalizations
from pymcdm.methods import TOPSIS
from pymcdm.weights import standard_deviation_weights

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


def pesos_sd(matriz: pd.DataFrame) -> np.ndarray:
    """Vetor de pesos [PM-SD]: desvio padrão por coluna, normalizado pela soma.

    Algoritmo 1 do relatório de iniciação científica (STD weights do pymcdm,
    Q11 de docs/DECISIONS.md): `std(coluna, ddof=1) / Σ std`. Reproduz o
    exemplo literal do relatório (matriz [[1,10,100],[2,20,105],[3,30,95]] ⇒
    pesos [0.0625, 0.625, 0.3125] — conferido no DECISIONS.md).

    Matriz inteira constante (todas as colunas com norma de variância nula ⇒
    divisão por zero no pymcdm) recebe pesos igualitários 1/k: degenera na
    variante uniforme do AC.RankA, que é o comportamento esperado quando
    nenhum critério discrimina. Colunas individuais com std = 0 recebem
    peso 0 (guarda finesta do pymcdm: soma total > 0).
    """
    m = np.asarray(matriz, dtype=float)
    k = m.shape[1]
    std = np.std(m, axis=0, ddof=1)
    total = std.sum()
    if not np.isfinite(total) or total == 0:
        return np.full(k, 1.0 / k)
    return standard_deviation_weights(m)


def scores_topsis(matriz: pd.DataFrame, pesos: np.ndarray | None = None) -> np.ndarray:
    """Score TOPSIS (maior = melhor) por linha da matriz regras × MOs [GF].

    TOPSIS com normalização vetorial pesada (Eq. 3 do paper): pymcdm 1.2.0
    com `vector_normalization`, `types` todos +1 (critério benefício) e pesos
    uniformes (AC.RankA original) ou o vetor `pesos` informado (AC.RankAW,
    [PM-SD]). Colunas de norma zero (MO constante em 0 — sem poder de
    discriminação) são descartadas antes da normalização: norma
    zero ⇒ 0/0 = NaN no pymcdm, e coluna nula contribui com distância nula
    para todas as linhas (ranking equivalente com pesos uniformes). Scores
    NaN (linhas idênticas ⇒ D⁺ = D⁻ = 0) viram 0.0: empate cai na ordem de
    extração pelo desempate do ranking.
    """
    m = np.asarray(matriz, dtype=float)
    if m.shape[0] <= 1:
        return np.zeros(m.shape[0])
    ativos = np.sqrt((m ** 2).sum(axis=0)) > 0
    m_ativa = m[:, ativos]
    n_crit = m_ativa.shape[1]
    if n_crit == 0:
        return np.zeros(m.shape[0])
    if pesos is None:
        pesos_ativos = np.ones(n_crit) / n_crit
    else:
        pesos_ativos = np.asarray(pesos, dtype=float)[ativos]
        pesos_ativos = pesos_ativos / pesos_ativos.sum()
    topsis = TOPSIS(normalization_function=normalizations.vector_normalization)
    with np.errstate(invalid="ignore", divide="ignore"):
        scores = topsis(m_ativa, pesos_ativos, np.ones(n_crit))
    return np.nan_to_num(scores, nan=0.0)


def ranking_topsis(matriz: pd.DataFrame, pesos: np.ndarray | None = None) -> list[int]:
    """Ranking decrescente do score TOPSIS; empate ⇒ ordem de extração.

    Retorna os índices das linhas de `matriz` (ordem de extração das regras).
    `pesos` (opcional) segue a mesma semântica de `scores_topsis`.
    """
    scores = scores_topsis(matriz, pesos)
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


class _MotorCBA:
    """Motor comum a CBA e AC.RankA: extração → ordenação → poda M1 → predição.

    O único ponto de variação entre os classificadores é `_ordenar`, que
    recebe as regras extraídas (ordem de extração) e devolve a ordem usada
    pela poda M1 e pela predição [O].
    """

    def __init__(self, sup_min: float | None = None, max_regras: int = MAX_REGRAS):
        self.sup_min = sup_min
        self.max_regras = max_regras

    def _ordenar(self, regras: list[tuple], X: pd.DataFrame, y: list[str]) -> list[tuple]:
        return ordenar_precedencia(regras)

    def fit(self, X: pd.DataFrame, y) -> "_MotorCBA":
        y_arr = [str(v) for v in y]
        regras = extrair_regras(X, y_arr, sup_min=self.sup_min, max_regras=self.max_regras)
        ordenadas = self._ordenar(regras, X, y_arr)
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


class CBA(_MotorCBA):
    """Classification Based on Association Rules (poda M1, predição [O])."""

    def __init__(self, sup_min: float | None = None, max_regras: int = MAX_REGRAS):
        super().__init__(sup_min=sup_min, max_regras=max_regras)


class ACRankA(_MotorCBA):
    """AC.RankA (Dall'Agnol & Carvalho 2024) — CBA com ordenação por TOPSIS.

    Igual ao CBA, mudando SOMENTE a ordenação das regras: a matriz
    regras × 13 MOs [GF] entra no TOPSIS do pymcdm com normalização vetorial
    pesada (Eq. 3 do paper) e vetor de pesos uniformes; ranking decrescente
    do score, empate ⇒ ordem de extração. Poda M1 e predição [O] idênticas.
    """

    def _ordenar(self, regras: list[tuple], X: pd.DataFrame, y: list[str]) -> list[tuple]:
        if not regras:
            return []
        matriz = matriz_gf(regras, X, y)
        return [regras[i] for i in ranking_topsis(matriz)]


class ACRankAW(ACRankA):
    """AC.RankAW Ponderado `[GF][PM-SD][TS]` — AC.RankA com pesos [PM-SD].

    Igual ao AC.RankA, mudando SOMENTE o vetor de pesos do TOPSIS: pesos
    uniformes ⇒ pesos [PM-SD] via `pesos_sd(matriz_rules_x_mos)` (Algoritmo 1
    do relatório — desvio padrão por coluna normalizado pela soma). Ranking,
    poda M1 e predição [O] idênticas ao motor compartilhado. Como a matriz
    regras × MOs é reconstruída a cada `fit` (um por fold de treino), os
    pesos [PM-SD] são recalculados por fold — o algoritmo 1 do relatório.
    """

    def _ordenar(self, regras: list[tuple], X: pd.DataFrame, y: list[str]) -> list[tuple]:
        if not regras:
            return []
        matriz = matriz_gf(regras, X, y)
        return [regras[i] for i in ranking_topsis(matriz, pesos=pesos_sd(matriz))]
