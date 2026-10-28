"""Discretização Fayyad–Irani (MDLP) e utilidades de pré-processamento.

Replica fielmente o protocolo do paper Dall'Agnol & Carvalho (2024):
discretização MDLP (Fayyad & Irani, IJCAI-93) por fold de treino,
candidatos apenas em mudanças de valor, entropia em bits,
critério de aceitação MDLPC estrito (>), e remoção de atributos
single-valued pós-discretização.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _entropy(counts: np.ndarray) -> float:
    """Entropia de Shannon em bits para contagens de classe."""
    counts = counts[counts > 0].astype(float)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _recursive_cuts(x: np.ndarray, y: np.ndarray, lo: int, hi: int) -> list[float]:
    """MDLP recursivo sobre o intervalo [lo, hi) de (x, y) ordenados por x."""
    n = hi - lo
    if n < 2:
        return []

    counts_all = pd.Series(y[lo:hi]).value_counts().to_numpy()
    k = int((counts_all > 0).sum())
    if k < 2:
        return []  # um única classe no intervalo => folha (gain 0)

    ent_s = _entropy(counts_all)

    classes = pd.unique(y[lo:hi])
    class_pos = {c: i for i, c in enumerate(classes)}

    # Melhor corte: mínima E(A,T;S), empate -> mais à esquerda (comparação estrita)
    best_j = -1
    best_seen = np.inf

    # Candidatos: apenas entre valores CONSECUTIVOS DIFERENTES (WEKA/Orange;
    # ordem-independente, compatível com o Teorema 1)
    x_v = x[lo:hi]
    y_v = y[lo:hi]
    c = np.zeros((hi - lo, len(classes)), dtype=np.int64)
    for j in range(hi - lo):
        c[j, class_pos[y_v[j]]] = 1
    cum = np.cumsum(c, axis=0)
    # para cada candidato j (split após a posição j-1 do sub-bloco):
    for j in range(1, hi - lo):
        if x_v[j - 1] == x_v[j]:
            continue  # mesma valor: não é ponto de fronteira válido
        n1 = j
        n2 = n - j
        e1 = _entropy(cum[j - 1])
        e2 = _entropy(cum[-1] - cum[j - 1])
        e = (n1 * e1 + n2 * e2) / n
        if e < best_seen:  # estrito: primeira ocorrência do mínimo vence
            best_seen = e
            best_j = j
            best_cnts = (e1, e2)

    if best_j == -1 or best_seen >= ent_s:  # gain <= 0 => folha
        return []

    j = best_j
    cut = (x_v[j - 1] + x_v[j]) / 2.0
    n1, n2 = j, n - j
    e1, e2 = best_cnts
    k1 = int((cum[j - 1] > 0).sum())
    k2 = int(((cum[-1] - cum[j - 1]) > 0).sum())
    gain = ent_s - best_seen
    delta = np.log2(3.0**k - 2) - (k * ent_s - k1 * e1 - k2 * e2)
    threshold = (np.log2(n - 1) + delta) / n

    if gain > threshold:  # MDLPC estrito (paper, p. 1026)
        left = _recursive_cuts(x, y, lo, lo + j)
        right = _recursive_cuts(x, y, lo + j, hi)
        return left + [float(cut)] + right
    return []


def fayyad_irani_cuts(x: np.ndarray, y: np.ndarray) -> list[float]:
    """Cortes MDLP de Fayyad–Irani para uma coluna numérica e um alvo.

    Parameters
    ----------
    x : array 1-D numérico.
    y : array 1-D de rótulos de classe.

    Returns
    -------
    Lista ordenada de cortes (midpoints). Vazia se nenhuma partição for aceita.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    order = np.argsort(x, kind="stable")
    return _recursive_cuts(x[order], y[order], 0, len(x))


def _fmt(v: float) -> str:
    if np.isposinf(v):
        return "+∞"
    if np.isneginf(v):
        return "-∞"
    if float(v).is_integer():
        return str(int(v))
    return f"{round(v, 3):g}"


def cuts_to_interval_labels(cuts: list[float]) -> list[str]:
    """Rótulos dos bins definidos pelos cortes: [(-∞, c1], (c1, c2], ..., (cm, +∞]]."""
    edges = [-np.inf] + sorted(cuts) + [np.inf]
    return [f"({_fmt(a)}, {_fmt(b)}]" for a, b in zip(edges[:-1], edges[1:])]


def assign_bins(x: np.ndarray, cuts: list[float]) -> np.ndarray:
    """Índice do bin de cada valor; bins são (a, b] (valor == corte -> bin esquerdo)."""
    if not cuts:
        return np.zeros(len(x), dtype=int)
    return np.searchsorted(np.sort(np.asarray(cuts)), np.asarray(x, dtype=float), side="left")


class MDLPDiscretizer:
    """Discretizador Fayyad–Irani em API estilo scikit-learn.

    Colunas numéricas com mais de `min_numeric_cardinality` valores distintos
    viram bins MDLP; as demais são mantidas (categóricas/integer baixa card.)
    e convertidas para string. Após o fit, colunas sem cortes (sem ganho de
    inf.) são sinalizadas via `dropped_columns_` para remoção.
    """

    def __init__(self, min_numeric_cardinality: int = 10):
        self.min_numeric_cardinality = min_numeric_cardinality

    def fit(self, X: pd.DataFrame, y) -> "MDLPDiscretizer":
        y = np.asarray(y)
        self.cut_points_: dict[str, list[float]] = {}
        self.dropped_columns_: list[str] = []
        self.passthrough_columns_: list[str] = []
        for col in X.columns:
            s = X[col]
            if pd.api.types.is_numeric_dtype(s) and s.nunique() > self.min_numeric_cardinality:
                cuts = fayyad_irani_cuts(s.to_numpy(), y)
                if not cuts:
                    self.dropped_columns_.append(col)
                else:
                    self.cut_points_[col] = sorted(cuts)
            else:
                self.passthrough_columns_.append(col)
        return self

    def transform(self, X: pd.DataFrame, prefix: bool = True) -> pd.DataFrame:
        out = {}
        for col in self.passthrough_columns_:
            out[col] = X[col].astype(str).to_numpy()
        for col, cuts in self.cut_points_.items():
            bins = assign_bins(X[col].to_numpy(), cuts)
            labels = cuts_to_interval_labels(cuts)
            if prefix:
                out[col] = [f"{col}_{labels[b]}" for b in bins]
            else:
                out[col] = [labels[b] for b in bins]
        return pd.DataFrame(
            {c: out[c] for c in self.passthrough_columns_ + list(self.cut_points_.keys())},
            index=X.index,
        )

    def fit_transform(self, X: pd.DataFrame, y) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
