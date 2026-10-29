"""13 MOs do grupo [GF] (Yang & Cui 2015; Apêndice B, Table 11 do paper).

Fórmulas em função da tabela de contingência da regra A ⇒ B no fold de treino:

- n11: transações cobertas pelo antecedente com a classe correta (P(AB)·N)
- n10: transações cobertas pelo antecedente com a classe errada (P(AB̄)·N)
- n01: transações não cobertas com a classe correta (P(ĀB)·N)
- n00: transações não cobertas com a classe errada (P(ĀB̄)·N)

Convenção para divisão por zero: o resultado da MO é 0.0 (valor neutro na
soma de ranks/TOPSIS do Ticket 6, que usa ranks — divisões por zero ocorrem
em regras "perfeitas" (n10 = 0) e 0.0 mantém as contas finitas).

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

# Nomes canônicos (ordem = lista do paper/self-ref); usado como eixo da matriz
# regras × MOs [GF] no Ticket 6 (TOPSIS determinístico por coluna).
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
    if nome not in MEDIDAS_GF:
        raise ValueError(f"MO '{nome}' não pertence ao grupo [GF].")

    n_a = n11 + n10          # P(A)·N  (cobre antecedente)
    n_b = n11 + n01          # P(B)·N  (classe correta)
    n_bbarra = n10 + n00     # P(B̄)·N
    n = n11 + n10 + n01 + n00

    if nome == "Confidence":
        return _razao(n11, n_a)
    if nome == "Odd Multiplier":  # P(AB)·P(B̄) / (P(B)·P(AB̄))
        return _razao(n11 * n_bbarra, n_b * n10)
    if nome == "Complement Class Support":  # P(AB̄)/P(B̄)
        # CPS mede a força de cobertura da classe COMPLEMENTAR (P(AB̄)/P(B̄)):
        # para ele, MENOR é melhor. O paper, porém, aplica o TOPSIS (Eq. 4-5)
        # com `max{C_j}` como ideal positivo para TODOS os critérios — sem
        # distinguir critérios de custo —, e é esse protocolo que o repo
        # reproduz (ver `src.associative.scores_topsis` e a seção
        # "Limitações conhecidas" do README).
        return _razao(n10, n_bbarra)
    if nome == "Conviction":  # P(A)·P(B̄) / P(AB̄)
        return _razao(n_a * n_bbarra, n * n10)
    if nome == "Yule-Q":
        num = n11 * n00 - n10 * n01
        den = n11 * n00 + n10 * n01
        return _razao(num, den)
    if nome == "Sebag-Schoenauer":  # P(AB)/P(AB̄)
        return _razao(n11, n10)
    if nome == "Yule-Y":
        r1 = math.sqrt(n11 * n00)
        r2 = math.sqrt(n10 * n01)
        return _razao(r1 - r2, r1 + r2)
    if nome == "Odds Ratio":  # P(AB)·P(ĀB̄) / (P(AB̄)·P(ĀB))
        return _razao(n11 * n00, n10 * n01)
    if nome == "Confirm Causal":  # P(AB) + P(ĀB̄) − 2·P(AB̄)
        return (n11 + n00 - 2 * n10) / n if n > 0 else 0.0
    if nome == "Confirmed Confidence Causal":  # ½(P(B|A)+P(Ā|B̄)) − P(B̄|A)
        p_b_a = _razao(n11, n_a)
        p_a_bbarra = _razao(n00, n_bbarra)
        p_bbarra_a = _razao(n10, n_a)
        return 0.5 * (p_b_a + p_a_bbarra) - p_bbarra_a
    if nome == "Example and Counterexample Rate":  # 1 − P(AB̄)/P(AB)
        return 1.0 - _razao(n10, n11)
    if nome == "Ganascia":  # 2·P(B|A) − 1
        return 2.0 * _razao(n11, n_a) - 1.0
    if nome == "J-Measure":  # P(AB)log2(P(B|A)/P(B)) + P(AB̄)log2(P(B̄|A)/P(B̄))
        valor = 0.0
        p_ab = n11 / n if n > 0 else 0.0
        p_abbarra = n10 / n if n > 0 else 0.0
        if p_ab > 0:
            valor += p_ab * math.log2(_razao(n11, n_a) / _razao(n_b, n))
        if p_abbarra > 0:
            valor += p_abbarra * math.log2(_razao(n10, n_a) / _razao(n_bbarra, n))
        return valor
    raise AssertionError("inalcançável")  # pragma: no cover


def medidas_gf(n11: int, n10: int, n01: int, n00: int) -> dict[str, float]:
    """Todas as 13 MOs [GF] de uma vez, em {nome: valor} (ordem canônica)."""
    return {nome: medida_gf(nome, n11, n10, n01, n00) for nome in MEDIDAS_GF}
