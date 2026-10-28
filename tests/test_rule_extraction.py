"""Testes da extração de regras de classe (Apriori + filtro consequente=classe)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.rule_extraction import extrair_regras, suporte_minimo_calibrado


@pytest.fixture
def X_disc_sinal():
    """10 transações discretizadas com regras conhecidas para a classe."""
    genero = ["M", "M", "F", "F", "M", "F", "F", "M", "M", "F"]
    contrato = ["M", "M", "M", "M", "A", "A", "A", "A", "M", "A"]
    servico = ["Fibra"] * 3 + ["Fibra", "Fibra", "Fibra", "Fibra", "Fibra", "Fibra", "Fibra"]
    y = ["Yes", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No"]
    X = pd.DataFrame({"Genero": genero, "Contrato": contrato})
    return X, y


def _regra(regras, antecedente, consequente):
    for a, c, sup, conf in regras:
        if a == antecedente and c == consequente:
            return (sup, conf)
    raise AssertionError(f"regra {antecedente}=>{consequente} ausente")


def test_consuporte_e_confianca_conhecidos(X_disc_sinal):
    X, y = X_disc_sinal
    regras = extrair_regras(X, y, sup_min=0.10)
    # {Genero=M} ⇒ Yes: 3/10 suporte; conf 3/5 = 0.6
    sup, conf = _regra(regras, frozenset({"Genero=M"}), "Yes")
    assert sup == pytest.approx(0.3)
    assert conf == pytest.approx(0.6)
    # {Genero=F} ⇒ No: 5/10; conf 5/5 = 1.0
    sup, conf = _regra(regras, frozenset({"Genero=F"}), "No")
    assert sup == pytest.approx(0.5)
    assert conf == pytest.approx(1.0)
    # {Contrato=M} ⇒ Yes: 3/10; conf 3/5 = 0.6
    sup, conf = _regra(regras, frozenset({"Contrato=M"}), "Yes")
    assert sup == pytest.approx(0.3)
    assert conf == pytest.approx(0.6)


def test_todas_as_regras_tem_consequente_classe(X_disc_sinal):
    X, y = X_disc_sinal
    for antecedente, consequente, *_ in extrair_regras(X, y, sup_min=0.1):
        assert consequente in ("Yes", "No")
        assert 1 <= len(antecedente) <= 4
        # itens são sempre "coluna=valor" sem a coluna da classe
        assert all("=" in it and not it.startswith("Churn=") for it in antecedente)


def test_conf_min_zero_gera_regras_fracas(X_disc_sinal):
    X, y = X_disc_sinal
    regras = extrair_regras(X, y, sup_min=0.1)
    # {Genero=M} ⇒ Yes tem conf 0.6; {Genero=M}⇒No conf 0.4 deve existir também
    sup, conf = _regra(regras, frozenset({"Genero=M"}), "No")
    assert sup == pytest.approx(0.2)
    assert conf == pytest.approx(0.4)


def test_antecedente_maximo_4(X_disc_sinal):
    X, y = X_disc_sinal
    X3 = pd.concat([X] * 1, axis=1)  # apenas 2 colunas aqui; ampliável
    for antecedente, *rest in extrair_regras(X3, y, sup_min=0.1):
        assert len(antecedente) <= 4


def test_determinismo(X_disc_sinal):
    X, y = X_disc_sinal
    a = extrair_regras(X, y, sup_min=0.1)
    b = extrair_regras(X, y, sup_min=0.1)
    assert a == b


def test_vetor_de_regras_ordenacao_instavel_ok(X_disc_sinal):
    X, y = X_disc_sinal
    regras = extrair_regras(X, y, sup_min=0.1)
    assert all(len(r[0]) >= 1 for r in regras)


# ---- Calibração do sup_min ----
def test_calibrador_decai_ate_abaixo_do_limite():
    chamadas = {"n": 0}

    def contador(mun):
        chamadas["n"] += 1
        return 10_001 * 0 + 15000  # sempre acima do limite: decai

    sup = suporte_minimo_calibrado(n_max_regras=10_000, contagem=contador)
    # começa em 0.10 e decai ×0.5 a cada tentativa acima do limite
    assert chamadas["n"] == 8
    assert sup == pytest.approx(0.10 * 0.5**7)


def test_calibrador_para_no_primeiro_abaixo_do_limite():
    seq = [10_000, 5]

    def contador(min_sup):
        return seq.pop(0)

    sup = suporte_minimo_calibrado(n_max_regras=10_000, contagem=contador)
    assert sup == pytest.approx(0.10)


def test_extracao_integra_calibracao(X_disc_sinal):
    X, y = X_disc_sinal
    # com limite baixo (1), o calibrador deve decair até o piso e o run terminar
    regras = extrair_regras(X, y, max_regras=1)
    assert len(regras) >= 1
