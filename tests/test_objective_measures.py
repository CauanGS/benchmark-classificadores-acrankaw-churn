"""Testes das 13 MOs do grupo [GF] sobre tabela de contingência manual conhecida.

Contingência de referência (regra A ⇒ B): N=20, n11=10, n10=2, n01=1, n00=7.
Valores esperados calculados à mão (fonte independente: Apêndice B do paper
Dall'Agnol & Carvalho 2024, Table 11 + Tan/Brin p/ Conviction + Tabela 2.2 TU
thesis p/ Ganascia/ECR).

Terminologia: n11 = cobre antecedente & classe correta; n10 = cobre & errada;
n01 = não cobre & correta; n00 = não cobre & errada.

A segunda parte do arquivo traz os testes **property-based** (hypothesis):
invariantes que as fórmulas do paper garantem para qualquer tabela de
contingência. Onde o paper não garante nada (p.ex. "Example and Counterexample
Rate" não é limitada inferiormente), o teste cobre apenas o que a fórmula
assegura.

Observação sobre *Leverage*: a MO **não faz parte do grupo [GF]** (paper,
Seção IV-B-2 e Table 11 ID 26: Leverage pertence aos grupos G< e C1), logo
não está implementada em `src/objective_measures.py` e não é testada aqui.
A monotonia exigida para Leverage não é, portanto, testável neste escopo; o
que se testa é a monotonia da Confidence (P(B|A)) e das demais MOs do [GF]
cuja fórmula a documenta.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

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


def test_mapa_de_calculo_das_mos_tem_a_ordem_canonica():
    """As 13 MOs [GF] são despachadas por um mapa nome→função (sem cascata de
    `if`), na mesma ordem canônica de `MEDIDAS_GF`."""
    from src.objective_measures import CALCULOS_GF, MEDIDAS_GF

    assert list(CALCULOS_GF) == list(MEDIDAS_GF)
    assert all(callable(calculo) for calculo in CALCULOS_GF.values())


def test_cada_mo_do_mapa_devolve_o_valor_canonico():
    from src.objective_measures import CALCULOS_GF

    for nome, calculo in CALCULOS_GF.items():
        assert calculo(*CASO) == pytest.approx(medida_gf(nome, *CASO)), nome


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


# =====================================================================
# Testes property-based (hypothesis) — ticket #5
#
# Cada invariante abaixo é garantido pela FÓRMULA da MO (paper, Apêndice B /
# Table 11 e fontes originais citadas em src/objective_measures.py). Não são
# inventados: onde a fórmula não impõe limite (ex.: "Example and Counterexample
# Rate" = 1 − P(AB̄)/P(AB) pode ir a −∞), o teste cobre só o que ela garante.
# =====================================================================

CONTAGENS = st.integers(min_value=0, max_value=1000)


@st.composite
def tabela_contingencia(draw):
    """Tabela 2×2 qualquer: (n11, n10, n01, n00) com contagens não negativas."""
    return (
        draw(CONTAGENS),
        draw(CONTAGENS),
        draw(CONTAGENS),
        draw(CONTAGENS),
    )


SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# MOs do [GF] com limite superior/inferior garantido pela fórmula.
LIMITES = {
    # confiança e suporte da classe complementar: proporções em [0, 1]
    "Confidence": (0.0, 1.0),
    "Complement Class Support": (0.0, 1.0),
    # medidas centradas: P(B|A) ∈ [0,1] ⇒ 2P(B|A) − 1 ∈ [−1,1]
    "Ganascia": (-1.0, 1.0),
    # quocientes de Odds: |Q| ≤ 1
    "Yule-Q": (-1.0, 1.0),
    "Yule-Y": (-1.0, 1.0),
    # ½(P(B|A)+P(Ā|B̄)) − P(B̄|A): soma de três termos em [0,1] ⇒ [−1,1]
    "Confirmed Confidence Causal": (-1.0, 1.0),
    # (P(AB)+P(ĀB̄)−2P(AB̄)): mínimo −2 (tudo coberto e errado), máximo 1
    "Confirm Causal": (-2.0, 1.0),
    # J-Measure = informação mútua I(B;A) com B binário ⇒ [0, H(B)] ⊆ [0,1]
    "J-Measure": (0.0, 1.0),
    # razões de contagem: não negativas (convenção 0.0 na divisão por zero)
    "Odd Multiplier": (0.0, math.inf),
    "Conviction": (0.0, math.inf),
    "Sebag-Schoenauer": (0.0, math.inf),
    "Odds Ratio": (0.0, math.inf),
    # 1 − P(AB̄)/P(AB): limitado SUPERIORMENTE por 1 (inferior = −∞)
    "Example and Counterexample Rate": (-math.inf, 1.0),
}


@SETTINGS
@given(tabela_contingencia())
def test_mos_sao_finitas_para_qualquer_tabela(caso):
    """Nenhuma MO pode estourar inf/nan (a convenção do repo devolve 0.0)."""
    for nome in LIMITES:
        valor = medida_gf(nome, *caso)
        assert math.isfinite(valor), (nome, caso, valor)


@SETTINGS
@given(tabela_contingencia())
def test_mos_respeitam_os_limites_da_formula(caso):
    for nome, (minimo, maximo) in LIMITES.items():
        valor = medida_gf(nome, *caso)
        # tolerância de ponto flutuante nas bordas (ex.: J-Measure = −1e−17)
        assert valor >= minimo - 1e-9, (nome, caso, valor, minimo)
        assert valor <= maximo + 1e-9, (nome, caso, valor, maximo)


@SETTINGS
@given(CONTAGENS, CONTAGENS, CONTAGENS, CONTAGENS)
def test_confidence_e_monotona_em_n11_e_n10(n11, n10, n01, n00):
    """Confidence = P(B|A) = n11/(n11+n10): cresce com n11 e decresce com n10.

    (Leverage — Table 11 ID 26 — pertence aos grupos G</C1 do paper, não ao
    [GF]; por isso não é testada aqui.)
    """
    base = medida_gf("Confidence", n11, n10, n01, n00)
    com_mais_certo = medida_gf("Confidence", n11 + 1, n10, n01, n00)
    com_mais_erro = medida_gf("Confidence", n11, n10 + 1, n01, n00)
    assert com_mais_certo >= base
    assert com_mais_erro <= base


@SETTINGS
@given(CONTAGENS, CONTAGENS, CONTAGENS, CONTAGENS)
def test_confidence_cresce_com_n11_a_da_linha_cheia(n11, n10, n01, n00):
    """Acréscimo de acertos (n11) nunca reduz a confiança; acréscimo de erros
    (n10) nunca a aumenta — e a confiança fica em [0,1]."""
    com_mais_certo = medida_gf("Confidence", n11 + 1, n10, n01, n00)
    com_mais_erro = medida_gf("Confidence", n11, n10 + 1, n01, n00)
    assert 0.0 <= com_mais_certo <= 1.0
    assert 0.0 <= com_mais_erro <= 1.0


@SETTINGS
@given(CONTAGENS, CONTAGENS, CONTAGENS, CONTAGENS)
def test_j_measure_e_informacao_mutua_limitada_a_1(n11, n10, n01, n00):
    """J-Measure = I(B;A) com B binário: ≥ 0 e ≤ H(B) ≤ 1."""
    valor = medida_gf("J-Measure", n11, n10, n01, n00)
    assert -1e-9 <= valor <= 1.0 + 1e-9


def test_valores_manuais_na_tabela_de_contingencia_conhecida():
    """Equivalência com os valores manuais da contingência de referência
    (N=20, n11=10, n10=2, n01=1, n00=7) — fonte independente do paper."""
    p_b_a = 10 / 12
    p_bbarra_a = 2 / 12
    p_a_bbarra = 7 / 9
    p_ab, p_abbarra = 10 / 20, 2 / 20
    p_b, p_bbarra = 11 / 20, 9 / 20
    r1, r2 = math.sqrt(10 * 7), math.sqrt(2 * 1)
    esperado = {
        "Odd Multiplier": 10 * 9 / (11 * 2),
        "Complement Class Support": 2 / 9,
        "Conviction": 12 * 9 / (20 * 2),
        "Yule-Q": (10 * 7 - 2 * 1) / (10 * 7 + 2 * 1),
        "Sebag-Schoenauer": 10 / 2,
        "Yule-Y": (r1 - r2) / (r1 + r2),
        "Odds Ratio": 10 * 7 / (2 * 1),
        "Confirm Causal": (10 + 7 - 4) / 20,
        "Confirmed Confidence Causal": 0.5 * (p_b_a + p_a_bbarra) - p_bbarra_a,
        "Example and Counterexample Rate": 1 - 2 / 10,
        "Ganascia": 2 * p_b_a - 1,
        "J-Measure": (
            p_ab * math.log2(p_b_a / p_b) + p_abbarra * math.log2(p_bbarra_a / p_bbarra)
        ),
        "Confidence": p_b_a,
    }
    from src.objective_measures import MEDIDAS_GF, medidas_gf

    assert set(esperado) == set(MEDIDAS_GF)
    obtido = medidas_gf(*CASO)
    for nome, valor in esperado.items():
        assert obtido[nome] == pytest.approx(valor), nome
