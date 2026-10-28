"""Testes do CBA: precedência, poda M1, matriz GF, predição [O] e default."""

from __future__ import annotations

import pandas as pd
import pytest

from src.associative import CBA, matriz_gf, ordenar_precedencia, poda_m1
from src.objective_measures import MEDIDAS_GF


def r(antecedente: frozenset, classe: str, conf: float, sup: float):
    return (antecedente, classe, sup, conf)


# ---- Precedência: confiança ↓, suporte ↓, antecedente menor primeiro ----
def test_precedencia_ordem():
    regras = [
        r(frozenset({"a"}), "Yes", 0.7, 0.3),
        r(frozenset({"b"}), "Yes", 0.9, 0.1),
        r(frozenset({"a", "b", "c"}), "No", 0.9, 0.1),
        r(frozenset({"a", "b"}), "No", 0.7, 0.3),
    ]
    ordem = ordenar_precedencia(regras)
    assert [(x[0], x[2]) for x in ordem] == [
        (frozenset({"b"}), 0.1),
        (frozenset({"a", "b", "c"}), 0.1),
        (frozenset({"a"}), 0.3),
        (frozenset({"a", "b"}), 0.3),
    ]


def test_precedencia_deterministico_com_empates():
    regras = [
        r(frozenset({"x1"}), "Yes", 0.5, 0.2),
        r(frozenset({"x2"}), "No", 0.5, 0.2),
    ]
    assert ordenar_precedencia(regras) == ordenar_precedencia(list(reversed(regras)))


# ---- Poda M1: cenário clássico (regra erra todas => não entra) ----
def test_poda_m1_cenario_classico():
    transacoes = [
        (frozenset({"a1"}), "No"),
        (frozenset({"a2"}), "Yes"),
        (frozenset({"a3"}), "No"),
        (frozenset({"a4"}), "No"),
    ]
    regras = [
        r(frozenset({"a3"}), "Yes", 1.0, 0.25),  # só cobre t3 (No) => erra todas
        r(frozenset({"a1"}), "No", 1.0, 0.25),   # acerta t1 => entra
        r(frozenset({"a2"}), "Yes", 1.0, 0.25),  # acerta t2 => entra
    ]
    mantidas, default = poda_m1(regras, transacoes)
    assert [x[0] for x in mantidas] == [frozenset({"a1"}), frozenset({"a2"})]
    assert default == "No"  # maioria do restante (t3, t4)


def test_poda_m1_deleta_somente_acertos():
    # regra cobre t1 (acerto) e t2 (erro); só t1 é deletada, t2 segue viva
    # e é coberta/atendida por regra seguinte.
    transacoes = [
        (frozenset({"a1", "b1"}), "Yes"),
        (frozenset({"a1"}), "No"),
        (frozenset({"b1"}), "Yes"),
    ]
    # NB: para simplificar o cenário, regras entram SEM erro: r1 acerta t1 e t3.
    regras = [
        r(frozenset({"b1"}), "Yes", 0.8, 0.5),
        r(frozenset({"a1"}), "Yes", 0.6, 0.4),
    ]
    mantidas, default = poda_m1(regras, transacoes)
    # r1 (precedência maior) acerta t1 e t3 => entra; t1,t3 deletadas
    # r2 cobre t1(deletada) e t2... t2 classe No não é acerto => r2 rejeitada
    assert [x[0] for x in mantidas] == [frozenset({"b1"})]
    assert default == "No"  # resto: t2


# ---- Matriz regras × MOs [GF]: contagem por transação ----
def test_matriz_gf_valores_manual():
    # Transações (items, classe):
    # t0 {a} No; t1 {a} No; t2 {a} Yes; t3 {b} Yes; t4 {}  Yes
    transacoes_X = pd.DataFrame(
        {"At": ["a", "a", "a", "b", "-"], "Out": ["-", "-", "-", "-", "-"]}
    )
    y = ["No", "No", "Yes", "Yes", "Yes"]
    regra = (frozenset({"At=a"}), "Yes", 1 / 3, 1 / 3)
    m = matriz_gf([regra], transacoes_X, y)
    # contingência: n11=1, n10=2, n01=2, n00=0
    linha = m.iloc[0]
    assert linha["Confidence"] == pytest.approx(1 / 3)
    assert linha["Sebag-Schoenauer"] == pytest.approx(0.5)
    assert set(m.columns) == set(MEDIDAS_GF)


# ---- Predição ordinal + fallback ----
@pytest.fixture
def cba_treinado():
    X = pd.DataFrame(
        {"A": ["a1", "a1", "a2", "a3", "a3"],
         "B": ["b1", "b1", "b2", "b2", "b2"]}
    )
    y = ["No", "No", "Yes", "No", "No"]
    cba = CBA(sup_min=0.2)
    cba.fit(X, y)
    return cba


def test_cba_predicao_primeira_regra_que_casa(cba_treinado):
    # {A=a1} em treino: t0,t1 (No) e a2,Yes… conf de {A=a1}⇒No = 2/2 = 1.0
    # instância {A=a1} => primeira regra casando (ranking por precedência)
    assert list(cba_treinado.predict(pd.DataFrame({"A": ["a1"], "B": ["b1"]}))) == ["No"]


def test_cba_default_quando_nada_casa(cba_treinado):
    desconhecido = pd.DataFrame({"A": ["zz"], "B": ["zz"]})
    assert list(cba_treinado.predict(desconhecido)) == [cba_treinado.classe_default_]


def test_cba_L_metrica(cba_treinado):
    # L = nº regras pós-poda + 1 default; deve ser inteiro positivo
    assert isinstance(cba_treinado.L, int)
    assert cba_treinado.L >= 1


def test_cba_deterministico():
    X = pd.DataFrame({"A": ["a1", "a1", "a2", "a3", "a2"], "B": ["b1", "b1", "b2", "b2", "b2"]})
    y = ["No", "No", "Yes", "No", "Yes"]
    c1 = CBA(sup_min=0.2).fit(X, y)
    c2 = CBA(sup_min=0.2).fit(X, y)
    X2 = pd.DataFrame({"A": ["a1", "a2", "a3"], "B": ["b1", "b2", "b2"]})
    assert list(c1.predict(X2)) == list(c2.predict(X2))
    assert c1.regras_ == c2.regras_
