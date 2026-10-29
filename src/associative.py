"""Classificador CBA (Liu et al. 1998) — precedência, poda M1 e predição [O].

- Precedência CBA: confiança ↓, suporte ↓, cardinalidade do antecedente ↑.
- Poda [M1] varre as regras nessa ordem mantendo as transações vivas:
  uma regra entra no classificador se acertar (classificar corretamente)
  pelo menos uma transação viva dentre as que ela cobre; ao entrar, TODAS
  as transações cobertas por ela são deletadas — acertadas ou não
  (protocolo do paper: Algorithm 1, linhas 11-14).
- Classe default = maioria das transações ainda vivas, recalculada a cada
  iteração da poda; a default final é a última calculada (ou a maioria
  global do treino, se nada restou) — empate resolvido pelo rótulo mais
  comum… empate: menor rótulo em ordem alfabética → determinismo.
- Predição [O]: classe da primeira regra do ranking que casa com a
  instância; fallback para a classe default.
- L = nº de regras pós-poda + 1 (a classe default).

AC.RankA (Dall'Agnol & Carvalho 2024) reutiliza o mesmo motor, trocando
apenas a ordenação: TOPSIS (pymcdm, normalização vetorial pesada — Eq. 3 do
paper) sobre a matriz regras × 13 MOs [GF] com pesos uniformes; empate no
score cai na ordem de extração (determinismo).

Todo classificador expõe `scores_`: o score de ordenação de cada regra
extraída (TOPSIS no AC.RankA/AC.RankAW, confiança no CBA) — é o número que
justifica a ordem do ranking e é exportado na Fase 3 (JSON/SQL).
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


def _classe_maioria(transacoes, indices) -> str:
    """Classe majoritária entre as transações de `indices` (empate: alfabética)."""
    contagens: dict[str, int] = {}
    for i in indices:
        classe = transacoes[i][1]
        contagens[classe] = contagens.get(classe, 0) + 1
    return sorted(contagens.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def poda_m1(regras_ordenadas, transacoes) -> tuple[list[tuple], str]:
    """Poda cover-style [M1] sobre `(antecedente, classe, sup, conf)` ordenadas.

    Protocolo do paper (Dall'Agnol & Carvalho 2024, Algorithm 1 — linhas 3-15;
    Seção II-C-2 "Pruning"): para cada regra, varre-se o conjunto de transações
    vivas D buscando as transações cobertas pela regra; se a regra classifica
    corretamente pelo menos uma transação coberta viva, ela é marcada e entra
    no classificador e, ao entrar, **TODAS** as transações cobertas são
    deletadas de D (cobertas acertadas ou não). A classe default (maioria do
    restante de D) é recalculada a cada iteração; a classe default final é a
    última calculada (ou a maioria global do treino, se nada restou).

    transacoes : lista [(itens_atributos: frozenset, classe: str), ...]
    Retorno: (regras mantidas na ordem, classe default do restante).
    """
    if not transacoes:
        raise ValueError("poda_m1 exige ao menos uma transação de treino.")
    vivas = list(range(len(transacoes)))
    mantidas: list[tuple] = []
    classe_default = _classe_maioria(transacoes, vivas)  # maioria global (fallback)
    for regra in regras_ordenadas:
        antecedente, classe = regra[0], regra[1]
        cobertas = [i for i in vivas if antecedente <= transacoes[i][0]]
        if any(transacoes[i][1] == classe for i in cobertas):
            mantidas.append(regra)
            # TODAS as transações cobertas são deletadas (acertos e erros).
            deletadas = set(cobertas)
            vivas = [i for i in vivas if i not in deletadas]
            if vivas:
                # Classe default recalculada a cada iteração (maioria do
                # restante); se nada restou, mantém-se a última calculada.
                classe_default = _classe_maioria(transacoes, vivas)
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


def _rank_por_scores(scores: np.ndarray) -> list[int]:
    """Índices ordenados pelo score (decrescente); empate ⇒ índice original."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def ranking_topsis(matriz: pd.DataFrame, pesos: np.ndarray | None = None) -> list[int]:
    """Ranking decrescente do score TOPSIS; empate ⇒ ordem de extração.

    Retorna os índices das linhas de `matriz` (ordem de extração das regras).
    `pesos` (opcional) segue a mesma semântica de `scores_topsis`.
    """
    scores = scores_topsis(matriz, pesos)
    return _rank_por_scores(scores)


class _MotorCBA:
    """Motor comum a CBA e AC.RankA: extração → ordenação → poda M1 → predição.

    O único ponto de variação entre os classificadores é `_ordenar`, que
    recebe as regras extraídas (ordem de extração) e devolve a ordem usada
    pela poda M1 e pela predição [O].
    """

    def __init__(self, sup_min: float | None = None, max_regras: int = MAX_REGRAS):
        self.sup_min = sup_min
        self.max_regras = max_regras
        # Score de ordenação de cada regra extraída (ver `_ordenar`); é o
        # número que justifica a ordem do ranking (usado no export Fase 3).
        self.scores_: dict[tuple, float] = {}

    def _ordenar(self, regras: list[tuple], X: pd.DataFrame, y: list[str]) -> list[tuple]:
        # CBA: o score de ordenação é a CONFIANÇA — critério primário da
        # precedência CBA (confiança ↓, suporte ↓, cardinalidade ↑).
        self.scores_ = {regra: float(regra[3]) for regra in regras}
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

    def _pesos(self, matriz: pd.DataFrame) -> np.ndarray | None:
        """Vetor de pesos do TOPSIS: uniforme (AC.RankA original)."""
        return None

    def _ordenar(self, regras: list[tuple], X: pd.DataFrame, y: list[str]) -> list[tuple]:
        if not regras:
            self.scores_ = {}
            return []
        matriz = matriz_gf(regras, X, y)
        pesos = self._pesos(matriz)
        scores = scores_topsis(matriz, pesos)
        # Score de ordenação de cada regra (TOPSIS sobre as 13 MOs [GF]).
        self.scores_ = {regra: float(score) for regra, score in zip(regras, scores)}
        return [regras[i] for i in _rank_por_scores(scores)]


class ACRankAW(ACRankA):
    """AC.RankAW Ponderado `[GF][PM-SD][TS]` — AC.RankA com pesos [PM-SD].

    Igual ao AC.RankA, mudando SOMENTE o vetor de pesos do TOPSIS: pesos
    uniformes ⇒ pesos [PM-SD] via `pesos_sd(matriz_rules_x_mos)` (Algoritmo 1
    do relatório — desvio padrão por coluna normalizado pela soma). Ranking,
    poda M1 e predição [O] idênticas ao motor compartilhado. Como a matriz
    regras × MOs é reconstruída a cada `fit` (um por fold de treino), os
    pesos [PM-SD] são recalculados por fold — o algoritmo 1 do relatório.
    """

    def _pesos(self, matriz: pd.DataFrame) -> np.ndarray | None:
        """Pesos [PM-SD] (desvio padrão por coluna normalizado pela soma)."""
        return pesos_sd(matriz)
