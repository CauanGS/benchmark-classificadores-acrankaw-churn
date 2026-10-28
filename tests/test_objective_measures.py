"""Testes das 13 MOs do grupo [GF] sobre tabela de contingência manual conhecida.

Contingência de referência (regra A ⇒ B): N=20, n11=10, n10=2, n01=1, n00=7.
Valores esperados calculados à mão (fonte independente: Apêndice B do paper
Dall'Agnol & Carvalho 2024, Table 11 + Tan/Brin p/ Conviction + Tabela 2.2 TU
thesis p/ Ganascia/ECR).

Terminologia: n11 = cobre antecedente & classe correta; n10 = cobre & errada;
n01 = não cobre & correta; n00 = não cobre & errada.
"""

from __future__ import annotations

import math

import pytest

from src.objective_measures import medida_gf

N = 20
CASO = (10, 2, 1, 7)


def _val(nome: str, caso: tuple = CASO) -> float:
    return medida_gf(nome, *caso)


# ---- Confidence (P(B|A) = n11/(n11+n10)) ----
def test_confidence():
    assert _val("Confidence") == pytest.approx(10 / 12)


# ---- Odd Multiplier: P(AB)P(B̄)/(P(B)P(AB̄)) ----
def test_odd_multiplier():
    esperado = 10 * (2 + 7) / ((10 + 1) * 2)
    assert _val("Odd Multiplier") == pytest.approx(esperado)  # 90/22


# ---- Complement Class Support: P(AB̄)/P(Β̄) = n10/(n10+n00) ----
def test_complement_class_support():
    assert _val("Complement Class Support") == pytest.approx(2 / 9)


# ---- Conviction: P(A)P(B̄)/P(AB̄) ----
def test_conviction():
    esperado = (10 + 2) * (2 + 7) / (20 * 2)
    assert _val("Conviction") == pytest.approx(esperado)  # 108/40 = 2.7


# ---- Yule-Q ----
def test_yule_q():
    esperado = (10 * 7 - 2 * 1) / (10 * 7 + 2 * 1)
    assert _val("Yule-Q") == pytest.approx(esperado)  # 68/72


# ---- Sebag-Schoenauer: n11/n10 ----
def test_sebag_schoenauer():
    assert _val("Sebag-Schoenauer") == pytest.approx(5.0)


# ---- Yule-Y ----
def test_yule_y():
    r1, r2 = math.sqrt(10 * 7), math.sqrt(2 * 1)
    assert _val("Yule-Y") == pytest.approx((r1 - r2) / (r1 + r2))


# ---- Odds Ratio ----
def test_odds_ratio():
    assert _val("Odds Ratio") == pytest.approx(10 * 7 / (2 * 1))


# ---- Confirm Causal: (n11 + n00 - 2 n10)/N ----
def test_confirm_causal():
    assert _val("Confirm Causal") == pytest.approx((10 + 7 - 4) / 20)  # 0.65


# ---- Confirmed Confidence Causal: ½(P(B|A)+P(Ā|B̄)) - P(B̄|A) ----
def test_confirmed_confidence_causal():
    p_b_a = 10 / 12
    p_a_bbarra = 7 / 9
    p_bbarra_a = 2 / 12
    esperado = 0.5 * (p_b_a + p_a_bbarra) - p_bbarra_a
    assert _val("Confirmed Confidence Causal") == pytest.approx(esperado)


# ---- Example and Counterexample Rate: 1 - n10/n11 ----
def test_example_counterexample_rate():
    assert _val("Example and Counterexample Rate") == pytest.approx(0.8)


# ---- Ganascia: 2·P(B|A) - 1 ----
def test_ganascia():
    assert _val("Ganascia") == pytest.approx(2 * 10 / 12 - 1)


# ---- J-Measure ----
def test_j_measure():
    p_ab, p_abbarra = 10 / 20, 2 / 20
    p_b, p_bbarra = 11 / 20, 9 / 20
    esperado = (
        p_ab * math.log2((10 / 12) / p_b) + p_abbarra * math.log2((2 / 12) / p_bbarra)
    )
    assert _val("J-Measure") == pytest.approx(esperado)


# ---- Propriedade: nomes canônicos das 13 MOs [GF] ----
def test_nomes_das_13_mos():
    esperado = {
        "Odd Multiplier",
        "Complement Class Support",
        "Conviction",
        "Yule-Q",
        "Sebag-Schoenauer",
        "Yule-Y",
        "Odds Ratio",
        "Confirm Causal",
        "Confirmed Confidence Causal",
        "Example and Counterexample Rate",
        "Ganascia",
        "J-Measure",
        "Confidence",
    }
    from src.objective_measures import MEDIDAS_GF

    assert set(MEDIDAS_GF) == esperado


def test_medida_desconhecida():
    with pytest.raises(ValueError):
        medida_gf("Kick", 1, 1, 1, 1)


# ---- Divisão por zero => 0.0 (exceto termos nulos de log) ----
@pytest.mark.parametrize(
    "nome",
    [
        "Odd Multiplier",
        "Complement Class Support",
        "Conviction",
        "Yule-Q",
        "Sebag-Schoenauer",
        "Yule-Y",
        "Odds Ratio",
        "Confirm Causal",
        "Confirmed Confidence Causal",
        "Example and Counterexample Rate",
        "Ganascia",
        "J-Measure",
        "Confidence",
    ],
)
def test_sem_divisao_por_zero(nome):
    # regra perfeita (n10=0) e regra vazia (n11=0) não podem lançar nem
    # produzir inf/nan — NaN encerraria o ranking por precedência.
    for caso in [(10, 0, 1, 9), (0, 4, 3, 13), (0, 0, 5, 15)]:
        v = medida_gf(nome, *caso)
        assert math.isfinite(v), (nome, caso, v)
