"""13 MOs do grupo [GF] (Yang & Cui 2015; Apêndice B, Table 11 do paper).

Fórmulas em função da tabela de contingência da regra A ⇒ B no fold de treino:

- n11: transações cobertas pelo antecedente com a classe correta (P(AB)·N)
- n10: transações cobertas pelo antecedente com a classe errada (P(AB̄)·N)
- n01: transações não cobertas com a classe correta (P(ĀB)·N)
- n00: transações não cobertas com a classe errada (P(ĀB̄)·N)

Cada MO tem uma **função de cálculo** própria; `CALCULOS_GF` (nome → função)
é o mapa de despacho, na ordem canônica de `MEDIDAS_GF` (sem cascata de `if`).

Convenção para divisão por zero: o resultado da MO é 0.0 (valor neutro na
matriz de scores do TOPSIS, que usa os VALORES das MOs — não ranks; divisões
por zero ocorrem em regras "perfeitas" (n10 = 0) e 0.0 mantém as contas
finitas).

Referências das fórmulas:
- Odd Multiplier (32), Complement Class Support (5), Odds Ratio (33),
  Confirm Causal (9), Confirmed Confidence Causal (11), J-Measure (20),
  Confidence (7): Appendix B (Table 11) do paper
  DOI 10.1109/ACCESS.2024.3419130.
- Conviction (Brin et al. 1997), Sebag–Schoenauer (54), Yule-Q (59), Yule-Y
  (60), Example and Counterexample Rate (22), Ganascia (24): Tabela 2.2 da
  TU thesis Somyanonthanakul & Theeramunkong (2022), mesmas refinida em
  Tew et al. (2014) — fonte das medidas originais de Yang & Cui.
"""

from __future__ import annotations

import math
from typing import Callable

# Nomes canônicos (ordem = lista do paper/self-ref); usado como eixo da matriz
# regras × MOs [GF] no TOPSIS determinístico por coluna.
MEDIDAS_GF: dict[str, str] = {
    "Odd Multiplier": "UMA",
    "Complement Class Support": "CPS",
    "Conviction": "CONV",
    "Yule-Q": "YQ",
    "Sebag-Schoenauer": "SS",
    "Yule-Y": "YY",
    "Odds Ratio": "OR",
    "Confirm Causal": "CF_CA",
    "Confirmed Confidence Causal": "CF_CN_CA",
    "Example and Counterexample Rate": "ECR",
    "Ganascia": "GAN",
    "J-Measure": "JME",
    "Confidence": "CONF",
}


def _razao(num: float, den: float) -> float:
    """num/den com divisão por zero -> 0.0 (convenção deste repo)."""
    try:
        return num / den if den != 0 else 0.0
    except ZeroDivisionError:  # pragma: no cover — defesa
        return 0.0


def _contagens(n11: int, n10: int, n01: int, n00: int) -> tuple[int, int, int, int]:
    """Contagens derivadas da contingência: (n_a, n_b, n_bbarra, n)."""
    return n11 + n10, n11 + n01, n10 + n00, n11 + n10 + n01 + n00


def _odd_multiplier(n11: int, n10: int, n01: int, n00: int) -> float:
    """Odd Multiplier: P(AB)·P(B̄) / (P(B)·P(AB̄))."""
    _, n_b, n_bbarra, _ = _contagens(n11, n10, n01, n00)
    return _razao(n11 * n_bbarra, n_b * n10)


def _complement_class_support(n11: int, n10: int, n01: int, n00: int) -> float:
    """Complement Class Support: P(AB̄)/P(B̄) = n10/(n10+n00).

    O CPS mede a força de cobertura da classe COMPLEMENTAR (P(AB̄)/P(B̄)):
    para ele, MENOR é melhor. O paper, porém, aplica o TOPSIS (Eq. 4-5)
    com `max{C_j}` como ideal positivo para TODOS os critérios — sem
    distinguir critérios de custo —, e é esse protocolo que o repo
    reproduz (ver `src.associative.scores_topsis` e a seção
    "Limitações conhecidas" do README).
    """
    _, _, n_bbarra, _ = _contagens(n11, n10, n01, n00)
    return _razao(n10, n_bbarra)


def _conviction(n11: int, n10: int, n01: int, n00: int) -> float:
    """Conviction: P(A)·P(B̄) / P(AB̄)."""
    n_a, _, n_bbarra, n = _contagens(n11, n10, n01, n00)
    return _razao(n_a * n_bbarra, n * n10)


def _yule_q(n11: int, n10: int, n01: int, n00: int) -> float:
    """Yule-Q: (n11·n00 − n10·n01) / (n11·n00 + n10·n01)."""
    num = n11 * n00 - n10 * n01
    den = n11 * n00 + n10 * n01
    return _razao(num, den)


def _sebag_schoenauer(n11: int, n10: int, n01: int, n00: int) -> float:
    """Sebag–Schoenauer: P(AB)/P(AB̄) = n11/n10."""
    return _razao(n11, n10)


def _yule_y(n11: int, n10: int, n01: int, n00: int) -> float:
    """Yule-Y: (√(n11·n00) − √(n10·n01)) / (√(n11·n00) + √(n10·n01))."""
    r1 = math.sqrt(n11 * n00)
    r2 = math.sqrt(n10 * n01)
    return _razao(r1 - r2, r1 + r2)


def _odds_ratio(n11: int, n10: int, n01: int, n00: int) -> float:
    """Odds Ratio: P(AB)·P(ĀB̄) / (P(AB̄)·P(ĀB))."""
    return _razao(n11 * n00, n10 * n01)


def _confirm_causal(n11: int, n10: int, n01: int, n00: int) -> float:
    """Confirm Causal: P(AB) + P(ĀB̄) − 2·P(AB̄)."""
    _, _, _, n = _contagens(n11, n10, n01, n00)
    return (n11 + n00 - 2 * n10) / n if n > 0 else 0.0


def _confirmed_confidence_causal(n11: int, n10: int, n01: int, n00: int) -> float:
    """Confirmed Confidence Causal: ½(P(B|A)+P(Ā|B̄)) − P(B̄|A)."""
    n_a, _, n_bbarra, _ = _contagens(n11, n10, n01, n00)
    p_b_a = _razao(n11, n_a)
    p_a_bbarra = _razao(n00, n_bbarra)
    p_bbarra_a = _razao(n10, n_a)
    return 0.5 * (p_b_a + p_a_bbarra) - p_bbarra_a


def _example_counterexample_rate(n11: int, n10: int, n01: int, n00: int) -> float:
    """Example and Counterexample Rate: 1 − P(AB̄)/P(AB)."""
    return 1.0 - _razao(n10, n11)


def _ganascia(n11: int, n10: int, n01: int, n00: int) -> float:
    """Ganascia: 2·P(B|A) − 1."""
    n_a, _, _, _ = _contagens(n11, n10, n01, n00)
    return 2.0 * _razao(n11, n_a) - 1.0


def _j_measure(n11: int, n10: int, n01: int, n00: int) -> float:
    """J-Measure: P(AB)log2(P(B|A)/P(B)) + P(AB̄)log2(P(B̄|A)/P(B̄))."""
    n_a, n_b, n_bbarra, n = _contagens(n11, n10, n01, n00)
    valor = 0.0
    p_ab = n11 / n if n > 0 else 0.0
    p_abbarra = n10 / n if n > 0 else 0.0
    if p_ab > 0:
        valor += p_ab * math.log2(_razao(n11, n_a) / _razao(n_b, n))
    if p_abbarra > 0:
        valor += p_abbarra * math.log2(_razao(n10, n_a) / _razao(n_bbarra, n))
    return valor


def _confidence(n11: int, n10: int, n01: int, n00: int) -> float:
    """Confidence: P(B|A) = n11/(n11+n10)."""
    n_a, _, _, _ = _contagens(n11, n10, n01, n00)
    return _razao(n11, n_a)


# Mapa de despacho nome → função de cálculo (ordem canônica = MEDIDAS_GF).
CALCULOS_GF: dict[str, Callable[[int, int, int, int], float]] = {
    "Odd Multiplier": _odd_multiplier,
    "Complement Class Support": _complement_class_support,
    "Conviction": _conviction,
    "Yule-Q": _yule_q,
    "Sebag-Schoenauer": _sebag_schoenauer,
    "Yule-Y": _yule_y,
    "Odds Ratio": _odds_ratio,
    "Confirm Causal": _confirm_causal,
    "Confirmed Confidence Causal": _confirmed_confidence_causal,
    "Example and Counterexample Rate": _example_counterexample_rate,
    "Ganascia": _ganascia,
    "J-Measure": _j_measure,
    "Confidence": _confidence,
}


def medida_gf(nome: str, n11: int, n10: int, n01: int, n00: int) -> float:
    """Valor da MO `nome` para a regra cuja contingência é dada.

    Parameters
    ----------
    nome : um dos nomes em :data:`MEDIDAS_GF` (case-insensitive).
    n11, n10, n01, n00 : contagens da tabela 2×2 (ver módulo).

    Raises
    ------
    ValueError se a medida não fizer parte do conjunto [GF].
    """
    try:
        calculo = CALCULOS_GF[nome]
    except KeyError:
        raise ValueError(f"MO '{nome}' não pertence ao grupo [GF].") from None
    return calculo(n11, n10, n01, n00)


def medidas_gf(n11: int, n10: int, n01: int, n00: int) -> dict[str, float]:
    """Todas as 13 MOs [GF] de uma vez, em {nome: valor} (ordem canônica)."""
    return {nome: medida_gf(nome, n11, n10, n01, n00) for nome in MEDIDAS_GF}
