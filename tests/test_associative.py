"""Testes do CBA: precedência, poda M1, matriz GF, predição [O] e default."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.associative import (
    ACRankA,
    ACRankAW,
    CBA,
    matriz_gf,
    ordenar_precedencia,
    poda_m1,
    pesos_sd,
    ranking_topsis,
    scores_topsis,
)
from src.objective_measures import MEDIDAS_GF
from src.regra import Regra


def r(antecedente: frozenset, classe: str, conf: float, sup: float) -> Regra:
    """Regra de teste — `conf` = confiança, `sup` = suporte."""
    return Regra(antecedente, classe, sup, conf)


# ---- Regra: tipo nomeado do domínio ----
def test_regra_expoe_campos_nomeados():
    regra = Regra(frozenset({"a=1"}), "Yes", 0.3, 0.6)
    assert regra.antecedente == frozenset({"a=1"})
    assert regra.classe == "Yes"
    assert regra.suporte == 0.3
    assert regra.confianca == 0.6
    assert regra.score is None
    assert regra.com_score(0.9).score == 0.9
    assert regra.cobre(frozenset({"a=1", "b=2"}))
    assert not regra.cobre(frozenset({"b=2"}))


def test_regra_com_score_preserva_o_resto_dos_campos():
    regra = Regra(frozenset({"a=1"}), "No", 0.25, 0.75)
    pontuada = regra.com_score(0.5)
    assert (pontuada.antecedente, pontuada.classe) == (regra.antecedente, regra.classe)
    assert (pontuada.suporte, pontuada.confianca) == (regra.suporte, regra.confianca)
    assert regra.score is None  # a original não é mutada


# ---- Precedência: confiança ↓, suporte ↓, antecedente menor primeiro ----
def test_precedencia_ordem():
    regras = [
        r(frozenset({"a"}), "Yes", 0.7, 0.3),
        r(frozenset({"b"}), "Yes", 0.9, 0.1),
        r(frozenset({"a", "b", "c"}), "No", 0.9, 0.1),
        r(frozenset({"a", "b"}), "No", 0.7, 0.3),
    ]
    ordem = ordenar_precedencia(regras)
    assert [(x.antecedente, x.suporte) for x in ordem] == [
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
    assert [x.antecedente for x in mantidas] == [frozenset({"a1"}), frozenset({"a2"})]
    assert default == "No"  # maioria do restante (t3, t4)


def test_poda_m1_deleta_todas_as_transacoes_cobertas():
    # Protocolo do paper (Algorithm 1, linhas 11-14; Seção II-C-2): a regra
    # entra se acerta >=1 transação coberta viva e, ao entrar, TODAS as
    # transações cobertas são deletadas — inclusive as classificadas errado.
    transacoes = [
        (frozenset({"a1"}), "Yes"),   # t0: acerto da regra
        (frozenset({"a1"}), "No"),    # t1: erro da regra (coberta)
        (frozenset({"a1"}), "No"),    # t2: erro da regra (coberta)
        (frozenset({"a1"}), "No"),    # t3: erro da regra (coberta)
        (frozenset({"b1"}), "Yes"),   # t4: não coberta
    ]
    regras = [r(frozenset({"a1"}), "Yes", 0.25, 0.8)]
    mantidas, default = poda_m1(regras, transacoes)
    # a regra cobre a classe errada e ainda assim é mantida (acerta t0)
    assert [x.antecedente for x in mantidas] == [frozenset({"a1"})]
    # t1..t3 (erros) foram deletadas: o default é o do RESTANTE (t4 => "Yes"),
    # não a maioria global do treino ("No": 3 No × 2 Yes).
    assert default == "Yes"


def test_poda_m1_regra_posterior_perde_a_transacao_certa_deletada():
    # t1 é coberta pela r1 (erro) e pela r2 (acerto): com o protocolo do paper
    # ela é deletada pela r1, e a r2 fica sem transação coberta viva => sai.
    transacoes = [
        (frozenset({"a1"}), "Yes"),       # t0: acerto da r1
        (frozenset({"a1", "a2"}), "No"),  # t1: erro da r1, acerto da r2
    ]
    regras = [
        r(frozenset({"a1"}), "Yes", 0.5, 0.5),
        r(frozenset({"a2"}), "No", 1.0, 0.5),
    ]
    mantidas, default = poda_m1(regras, transacoes)
    assert [x.antecedente for x in mantidas] == [frozenset({"a1"})]
    assert default == "No"  # nada restou => maioria global do treino


# ---- Matriz regras × MOs [GF]: contagem por transação ----
def test_matriz_gf_valores_manual():
    # Transações (items, classe):
    # t0 {a} No; t1 {a} No; t2 {a} Yes; t3 {b} Yes; t4 {}  Yes
    transacoes_X = pd.DataFrame(
        {"At": ["a", "a", "a", "b", "-"], "Out": ["-", "-", "-", "-", "-"]}
    )
    y = ["No", "No", "Yes", "Yes", "Yes"]
    regra = Regra(frozenset({"At=a"}), "Yes", 1 / 3, 1 / 3)
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


# ---- AC.RankA: ranking TOPSIS com pesos uniformes (normalização vetorial) ----

def test_ranking_topsis_matriz_conhecida():
    # m = [[3, 4], [1, 2]]; normas das colunas: √10 e √20; pesos uniformes ½.
    # Normalizada: r0 = [3/√10, 4/√20], r1 = [1/√10, 2/√20] ⇒ r0 domina r1.
    # D+r0 = 0, D-r0 = √(0.25/10 + 0.25/20) ⇒ V0 = 1.0; V1 = 0.0.
    matriz = pd.DataFrame({"m1": [3.0, 1.0], "m2": [4.0, 2.0]})
    assert ranking_topsis(matriz) == [0, 1]
    assert scores_topsis(matriz) == pytest.approx([1.0, 0.0])


def test_ranking_topsis_confere_com_pymcdm_vetorial():
    # Verificação da configuração pymcdm 1.2.0 usada na implementação:
    # vector_normalization + types todos +1 (benefício) + pesos uniformes
    # reproduz os valores hand-computados do teste anterior.
    from pymcdm import normalizations
    from pymcdm.methods import TOPSIS

    matriz = np.array([[3.0, 4.0], [1.0, 2.0]])
    topsis = TOPSIS(normalization_function=normalizations.vector_normalization)
    assert topsis(matriz, np.ones(2) / 2, np.ones(2)) == pytest.approx([1.0, 0.0])


def test_ranking_topsis_invariancia_escala():
    # Corolário do AC.RankA original (pesos uniformes): o score TOPSIS depende
    # apenas da posição/valores normalizados — multiplicar qualquer coluna da
    # matriz por constante positiva não muda o ranking. Coluna zerada (norma
    # zero, sem poder de discriminação) inclusa no caso.
    matriz = pd.DataFrame(
        {
            "a": [2.0, 5.0, 1.0, 4.0, 3.0],
            "b": [0.0, 0.0, 0.0, 0.0, 0.0],
            "c": [1.0, 2.0, 3.0, 5.0, 4.0],
            "d": [-1.0, 0.5, -2.0, 1.0, 0.0],
        }
    )
    escalada = matriz.copy()
    escalada["a"] = escalada["a"] * 7.5
    escalada["c"] = escalada["c"] * 0.02
    escalada["d"] = escalada["d"] * 100.0
    assert ranking_topsis(escalada) == ranking_topsis(matriz)


def test_ranking_topsis_empate_ordem_de_extracao():
    # Empate estrutural (linhas idênticas) ⇒ ordem de extração (índice).
    matriz = pd.DataFrame({"m1": [1.0, 1.0, 1.0], "m2": [2.0, 2.0, 2.0]})
    assert ranking_topsis(matriz) == [0, 1, 2]


# ---- AC.RankA: integração do motor (poda M1 + predição [O] do CBA) ----

def test_acranka_empate_estrutural_l_igual_cba():
    # Fixture degenerado: todas as regras têm a MESMA contingência
    # (n11=1, n10=0, n01=3, n00=0) ⇒ empate estrutural no TOPSIS; qualquer
    # ordenação mantém as mesmas regras na poda M1 ⇒ L(AC.RankA) = L(CBA).
    X = pd.DataFrame({"A": ["a1", "a2", "a3", "a4"]})
    y = ["No", "No", "No", "No"]
    ac = ACRankA(sup_min=0.25).fit(X, y)
    cba = CBA(sup_min=0.25).fit(X, y)
    assert ac.L == cba.L
    assert list(ac.predict(X)) == ["No", "No", "No", "No"]


def test_acranka_prediz_rotulos_validos_no_fixture_geral():
    X = pd.DataFrame(
        {"A": ["a1", "a1", "a2", "a3", "a3"], "B": ["b1", "b1", "b2", "b2", "b2"]}
    )
    y = ["No", "No", "Yes", "No", "No"]
    ac = ACRankA(sup_min=0.2).fit(X, y)
    preds = list(ac.predict(X))
    assert set(preds) <= {"No", "Yes"}
    assert ac.L >= 1


def test_acranka_deterministico():
    X = pd.DataFrame({"A": ["a1", "a1", "a2", "a3", "a2"], "B": ["b1", "b1", "b2", "b2", "b2"]})
    y = ["No", "No", "Yes", "No", "Yes"]
    a1 = ACRankA(sup_min=0.2).fit(X, y)
    a2 = ACRankA(sup_min=0.2).fit(X, y)
    X2 = pd.DataFrame({"A": ["a1", "a2", "a3"], "B": ["b1", "b2", "b2"]})
    assert list(a1.predict(X2)) == list(a2.predict(X2))
    assert a1.regras_ == a2.regras_


# ---- AC.RankAW Ponderado: pesos [PM-SD] (Algoritmo 1 do relatório) ----

def test_pesos_sd_exemplo_literal_relatorio():
    # Relatório de iniciação científica (Algoritmo 1, [PM-SD]): desvio padrão
    # por coluna (ddof=1) normalizado pela soma. Exemplo literal do relatório.
    matriz = pd.DataFrame(
        {"m1": [1.0, 2.0, 3.0], "m2": [10.0, 20.0, 30.0], "m3": [100.0, 105.0, 95.0]}
    )
    pesos = pesos_sd(matriz)
    assert pesos == pytest.approx([0.0625, 0.625, 0.3125])


def test_pesos_sd_matriz_variada_soma_1_positivos():
    # Qualquer matriz com colunas não constantes: pesos somam 1 e são
    # estritamente positivos (duas casas de variância diferentes escaladas).
    matriz = pd.DataFrame(
        {
            "a": [2.0, 5.0, 1.0, 4.0, 3.0],
            "b": [1000.0, 750.0, 500.0, 250.0, 100.0],
            "c": [1.0, 2.0, 3.0, 5.0, 4.0],
        }
    )
    pesos = pesos_sd(matriz)
    assert pesos.sum() == pytest.approx(1.0)
    assert (pesos > 0).all()


def test_pesos_sd_matriz_constante_pesos_igualitarios():
    # Matriz constante: std = 0 em toda coluna ⇒ 0/0; converção: pesos
    # igualitários (mesma régua da variante uniforme do AC.RankA).
    matriz = pd.DataFrame({"a": [2.0, 2.0, 2.0], "b": [5.0, 5.0, 5.0]})
    pesos = pesos_sd(matriz)
    assert pesos == pytest.approx([0.5, 0.5])


def test_scores_topsis_consistente_com_pymcdm_com_pesos_sd():
    # Configuração pymcdm 1.2.0 com vetor de pesos [PM-SD] explícito igual à
    # chamada direta do TOPSIS (mesma normalização vetorial pesada e types +1).
    from pymcdm import normalizations
    from pymcdm.methods import TOPSIS

    matriz = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0],
                           "c": [100.0, 105.0, 95.0]})
    w = pesos_sd(matriz)
    topsis = TOPSIS(normalization_function=normalizations.vector_normalization)
    esperado = np.nan_to_num(
        topsis(np.asarray(matriz, float), w, np.ones(matriz.shape[1])), nan=0.0
    )
    assert scores_topsis(matriz, pesos=w) == pytest.approx(esperado)


def test_scores_topsis_trata_o_cps_como_beneficio_como_no_paper():
    """Limitação documentada (decisão D7): o paper NÃO distingue critérios de
    custo — nas Eq. (4)-(5) o ideal positivo é max{C_j} para todo critério j.
    Logo o CPS (P(AB̄)/P(B̄), menor = melhor) entra como benefício: a regra
    com CPS maior recebe o score maior. A alternativa "custo" (types=-1)
    inverteria a ordem — não é adotada, por fidelidade ao paper."""
    from pymcdm import normalizations
    from pymcdm.methods import TOPSIS

    matriz = pd.DataFrame({"cps": [0.0, 1.0], "outra": [1.0, 1.0]})
    scores = scores_topsis(matriz)
    assert scores[1] > scores[0]  # CPS alto = benefício

    como_custo = TOPSIS(normalization_function=normalizations.vector_normalization)(
        np.asarray(matriz, float), np.ones(2) / 2, np.array([-1, 1])
    )
    assert como_custo[0] > como_custo[1]  # com types=-1 a ordem se inverteria


# ---- Guarda de divisão por zero na renormalização dos pesos ativos ----
def test_scores_topsis_pesos_com_soma_zero_cai_em_pesos_uniformes():
    """Pesos ativos com soma 0 (renormalização 0/0) ⇒ pesos uniformes entre
    os critérios ativos — nunca NaN silencioso."""
    matriz = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    scores = scores_topsis(matriz, pesos=np.array([0.0, 0.0]))
    assert np.isfinite(scores).all()
    assert scores == pytest.approx(scores_topsis(matriz))  # = pesos uniformes


def test_scores_topsis_pesos_nao_finitos_cai_em_pesos_uniformes():
    """Pesos com soma não finita (inf/NaN) também caem na guarda: o score
    continua finito e é o do pesos uniformes."""
    matriz = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    scores = scores_topsis(matriz, pesos=np.array([np.inf, 1.0]))
    assert np.isfinite(scores).all()
    assert scores == pytest.approx(scores_topsis(matriz))


def test_scores_topsis_guarda_considera_somente_os_criterios_ativos():
    """Coluna de norma zero (inativa) com peso 0 não quebra a renormalização:
    a guarda só considera os pesos dos critérios ATIVOS."""
    matriz = pd.DataFrame({"a": [1.0, 2.0, 3.0], "zero": [0.0, 0.0, 0.0]})
    scores = scores_topsis(matriz, pesos=np.array([1.0, 0.0]))
    assert np.isfinite(scores).all()
    assert scores == pytest.approx(scores_topsis(matriz[["a"]]))


# ---- AC.RankAW Ponderado: integração [GF][PM-SD][TS] + poda M1 ----

@pytest.fixture
def fixture_variancia_discriminante():
    # Fixture projetado para que o ranking ponderado [PM-SD] diferencia as
    # regras de modo que a poda M1 (protocolo do paper: TODAS as transações
    # cobertas são deletadas) mantenha MENOS regras que na variante uniforme.
    #
    # Com pesos uniformes, o TOPSIS põe {A0=v1, A1=v0}⇒Yes na frente: ela
    # cobre t0,t4 (Yes) e t1 (No, erro) => 3 regras no total. Com [PM-SD], a
    # regra mais geral {A0=v1}⇒Yes vai à frente: cobre t0,t4,t5 (Yes) e t1
    # (No, erro) => 2 regras.
    X = pd.DataFrame(
        {
            "A0": ["v1", "v1", "v0", "v0", "v1", "v1"],
            "A1": ["v0", "v1", "v0", "v1", "v0", "v1"],
        }
    )
    y = ["Yes", "No", "No", "Yes", "Yes", "Yes"]
    return X, y


def test_acrankaw_l_menor_que_acranka(fixture_variancia_discriminante):
    X, y = fixture_variancia_discriminante
    acranka = ACRankA(sup_min=0.25).fit(X, y)
    acrankaw = ACRankAW(sup_min=0.25).fit(X, y)
    assert acrankaw.L < acranka.L
    preds = list(acrankaw.predict(X))
    assert set(preds) <= {"Yes", "No"}
    assert len(preds) == len(y)


def test_acrankaw_deterministico(fixture_variancia_discriminante):
    X, y = fixture_variancia_discriminante
    w1 = ACRankAW(sup_min=0.25).fit(X, y)
    w2 = ACRankAW(sup_min=0.25).fit(X, y)
    X2 = pd.DataFrame({"A0": ["v1", "v0"], "A1": ["v0", "v1"]})
    assert list(w1.predict(X2)) == list(w2.predict(X2))
    assert w1.regras_ == w2.regras_
    assert w1.classe_default_ == w2.classe_default_
