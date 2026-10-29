"""Testes do export de regras de negócio (Fase 3): JSON, SQL, determinismo, CLI."""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.associative import CBA
from src.rule_export import exportar_regras, montar_json, montar_sql, selecionar_modelo


@pytest.fixture
def dataset_sintetico(tmp_path):
    """Dataset sintético pequeno e determinístico: Contract=Mensal ⇒ Churn=Yes.

    40 clientes, 2 classes balanceadas; `tenure` é uma permutação (sinal
    não correlacionado ao churn) e `MonthlyCharges` tem cardinalidade baixa
    (≤ 10 valores, portanto não passa pelo MDLP).
    """
    n = 40
    df = pd.DataFrame(
        {
            "customerID": [f"C{i:03d}" for i in range(n)],
            "tenure": [(i * 7) % n + 1 for i in range(n)],
            "MonthlyCharges": [20 + (i % 5) for i in range(n)],
            "TotalCharges": [
                float(((i * 7) % n + 1) * (20 + (i % 5))) for i in range(n)
            ],
            "SeniorCitizen": [0] * n,
            "Contract": ["Mensal"] * (n // 2) + ["Anual"] * (n - n // 2),
            "Churn": ["Yes"] * (n // 2) + ["No"] * (n - n // 2),
        }
    )
    caminho = tmp_path / "sintetico.csv"
    df.to_csv(caminho, index=False)
    return caminho


def test_export_json_tem_schema_documentado(dataset_sintetico, tmp_path):
    """(1) JSON com schema correto e regra parseável."""
    json_path = tmp_path / "rules_exported.json"
    sql_path = tmp_path / "rules_exported.sql"
    exportar_regras(
        caminho_dataset=str(dataset_sintetico),
        caminho_json=str(json_path),
        caminho_sql=str(sql_path),
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "metadados",
        "regras",
        "class_default",
        "modelos_alternativos",
    }

    meta = payload["metadados"]
    for campo in ("base", "alvo", "seed", "modelo", "L", "metricas"):
        assert campo in meta, f"metadados sem o campo '{campo}'"
    assert meta["alvo"] == "Churn"
    assert meta["seed"] == 42
    assert meta["modelo"] == "ac_rankaw"
    assert isinstance(meta["L"], int)
    assert meta["L"] >= 1
    assert "f1_macro" in meta["metricas"]

    assert isinstance(payload["regras"], list)
    assert payload["regras"], "o export deve conter ao menos uma regra"
    for regra in payload["regras"]:
        assert set(regra) >= {
            "id",
            "antecedente",
            "consequente",
            "suporte",
            "confianca",
            "score",
        }
        assert regra["antecedente"], "antecedente não pode ser vazio"
        for par in regra["antecedente"]:
            assert set(par) == {"atributo", "valor"}
            assert par["atributo"] and par["valor"] != ""
        assert regra["consequente"] in {"Yes", "No"}
        assert 0.0 <= regra["suporte"] <= 1.0
        assert 0.0 <= regra["confianca"] <= 1.0
        assert isinstance(regra["score"], float)

    # regra parseável: os pares atributo-valor recompõem o antecedente
    primeira = payload["regras"][0]
    recomposto = " AND ".join(
        f"{par['atributo']}={par['valor']}" for par in primeira["antecedente"]
    )
    assert recomposto == primeira["antecedente_serializado"]

    # o score de ordenação é decrescente na ordem do ranking exportado
    scores = [regra["score"] for regra in payload["regras"]]
    assert scores == sorted(scores, reverse=True)

    assert payload["class_default"]["classe"] in {"Yes", "No"}


def test_export_inclui_o_ac_ranka_como_modelo_alternativo(dataset_sintetico, tmp_path):
    """Seção secundária `modelos_alternativos`: regras do AC.RankA (baseline
    multicritério de pesos uniformes) para comparação."""
    json_path = tmp_path / "rules_exported.json"
    sql_path = tmp_path / "rules_exported.sql"
    exportar_regras(
        caminho_dataset=str(dataset_sintetico),
        caminho_json=str(json_path),
        caminho_sql=str(sql_path),
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    alternativos = payload["modelos_alternativos"]
    assert "ac_ranka" in alternativos
    bloco = alternativos["ac_ranka"]
    assert bloco["regras"], "o AC.RankA alternativo deve ter regras"
    assert bloco["class_default"]["classe"] in {"Yes", "No"}
    assert bloco["L"] == len(bloco["regras"]) + 1
    scores = [regra["score"] for regra in bloco["regras"]]
    assert scores == sorted(scores, reverse=True)


def test_alvo_do_export_e_o_ac_rankaw_ponderado():
    """A Fase 3 exporta o AC.RankAW Ponderado (spec issue #1, histórias 7-8):
    a escolha é de protocolo, não de desempenho."""
    assert selecionar_modelo() == "ac_rankaw"


def test_export_sql_executa_no_sqlite_e_resolve_a_primeira_regra(
    dataset_sintetico, tmp_path
):
    """(2) SQL executado em SQLite (sqlite3) sem erro e classe da 1ª regra."""
    json_path = tmp_path / "rules_exported.json"
    sql_path = tmp_path / "rules_exported.sql"
    exportar_regras(
        caminho_dataset=str(dataset_sintetico),
        caminho_json=str(json_path),
        caminho_sql=str(sql_path),
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    primeira = payload["regras"][0]
    sql = sql_path.read_text(encoding="utf-8")

    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(sql)  # não pode levantar erro

        colunas = [linha[1] for linha in conn.execute("PRAGMA table_info(regras)")]
        assert {
            "id",
            "antecedente_serializado",
            "classe_predita",
            "suporte",
            "confianca",
        } <= set(colunas)
        assert conn.execute("SELECT COUNT(*) FROM regras").fetchone()[0] == len(
            payload["regras"]
        )

        # o cliente de exemplo casa com a 1ª regra do JSON
        for par in primeira["antecedente"]:
            achados = conn.execute(
                f'SELECT COUNT(*) FROM cliente_exemplo WHERE "{par["atributo"]}" = ?',
                (par["valor"],),
            ).fetchone()[0]
            assert achados == 1, f"cliente de exemplo não casa em {par}"

        # query de decisão (CASE WHEN) e variante ORDER BY score
        assert conn.execute(
            "SELECT classe_predita FROM decisao_cliente"
        ).fetchone()[0] == primeira["consequente"]
        assert conn.execute(
            "SELECT classe_predita FROM decisao_cliente_por_score"
        ).fetchone()[0] == primeira["consequente"]
    finally:
        conn.close()

    binario = shutil.which("sqlite3")
    if binario:
        proc = subprocess.run(
            [binario, ":memory:"], input=sql, capture_output=True, text=True
        )
        assert proc.returncode == 0, proc.stderr
        assert primeira["consequente"] in proc.stdout


def test_export_deterministico(dataset_sintetico, tmp_path):
    """(3) Determinismo: dois exports do mesmo dataset são byte-idênticos."""
    a_json, a_sql = tmp_path / "a.json", tmp_path / "a.sql"
    b_json, b_sql = tmp_path / "b.json", tmp_path / "b.sql"
    for json_path, sql_path in ((a_json, a_sql), (b_json, b_sql)):
        exportar_regras(
            caminho_dataset=str(dataset_sintetico),
            caminho_json=str(json_path),
            caminho_sql=str(sql_path),
        )

    assert a_json.read_bytes() == b_json.read_bytes()
    assert a_sql.read_bytes() == b_sql.read_bytes()


def test_cli_fase3_gera_json_e_sql(dataset_sintetico, tmp_path):
    """(4) CLI --phase 3 termina com exit 0 e gera JSON + SQL."""
    saida = tmp_path / "outputs"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.run_benchmark",
            "--phase",
            "3",
            "--dataset",
            str(dataset_sintetico),
            "--saida",
            str(saida),
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (saida / "rules_exported.json").exists()
    assert (saida / "rules_exported.sql").exists()
    assert "Fase 3" in proc.stdout


# ---- A ordem exportada (`ordem`) é a ordem de predição [O] do classificador ----

# Fixture mínima em que DUAS regras de classes diferentes casam com o mesmo
# cliente: {A=a1}⇒Yes (confiança 1.0, entra 1ª no ranking CBA) e
# {B=b2}⇒No (confiança 1/3, entra 2ª). As duas sobrevivem à poda M1 e a
# predição [O] do classificador é "Yes" — a 1ª regra do ranking vence.
X_DUAS_REGRAS = pd.DataFrame(
    {
        "A": ["a1", "a1", "a2", "a2", "a2"],
        "B": ["b1", "b2", "b2", "b2", "b3"],
    }
)
Y_DUAS_REGRAS = ["Yes", "Yes", "No", "Yes", "No"]


def _metricas_ficticias(modelo) -> dict[str, dict]:
    """Bloco de métricas por modelo no formato do protocolo (só para o export)."""
    return {
        "cba": {
            "f1_macro": 0.5,
            "f1_macro_desvio": 0.1,
            "precisao_No": 0.5,
            "recall_No": 0.5,
            "precisao_Yes": 0.5,
            "recall_Yes": 0.5,
            "L": float(modelo.L),
        }
    }


def test_export_tem_a_coluna_ordem_com_a_posicao_do_ranking(dataset_sintetico, tmp_path):
    """(5) O JSON e o SQL carregam `ordem` = posição da regra no ranking."""
    json_path = tmp_path / "rules_exported.json"
    sql_path = tmp_path / "rules_exported.sql"
    exportar_regras(
        caminho_dataset=str(dataset_sintetico),
        caminho_json=str(json_path),
        caminho_sql=str(sql_path),
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert [r["ordem"] for r in payload["regras"]] == list(
        range(1, len(payload["regras"]) + 1)
    )
    assert [r["ordem"] for r in payload["regras"]] == [r["posicao"] for r in payload["regras"]]

    sql = sql_path.read_text(encoding="utf-8")
    assert "ordem INTEGER NOT NULL" in sql
    assert "ORDER BY r.ordem ASC" in sql


def test_ordem_do_sql_reproduz_a_ordem_de_predicao_do_classificador():
    """(5) A query de decisão do SQL elege a MESMA regra do classificador.

    A ordem do SQL é a coluna `ordem` (posição no ranking do classificador) —
    a 1ª regra que casa vence, como na predição ordinal [O]. O cliente da
    fixture casa com duas regras de classes diferentes: a do ranking (1ª)
    vence, e o SQL precisa devolver a mesma classe.
    """
    modelo = CBA(sup_min=0.2).fit(X_DUAS_REGRAS, Y_DUAS_REGRAS)
    assert len(modelo.regras_) >= 2, "a fixture precisa ter 2+ regras no ranking"
    assert modelo.regras_[0].classe != modelo.regras_[1].classe

    payload = montar_json(
        modelo=modelo,
        metricas_cv=_metricas_ficticias(modelo),
        nome_modelo="cba",
        caminho_dataset="mini.csv",
        seed=42,
        delta_l=0.0,
    )
    sql = montar_sql(payload, X_DUAS_REGRAS, modelo)

    # o cliente casa com as DUAS regras do ranking
    cliente = {"A": "a1", "B": "b2"}
    esperado = modelo.predict(pd.DataFrame([cliente]))[0]
    assert esperado == modelo.regras_[0].classe  # a 1ª regra do ranking vence

    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(sql)
        conn.execute(
            'INSERT INTO cliente_exemplo (cliente_id, "A", "B") VALUES (?, ?, ?)',
            ("C-DUAS-REGRAS", cliente["A"], cliente["B"]),
        )
        por_id = dict(
            conn.execute(
                "SELECT cliente_id, classe_predita FROM decisao_cliente_por_score"
            ).fetchall()
        )
        por_case = dict(
            conn.execute(
                "SELECT cliente_id, classe_predita FROM decisao_cliente"
            ).fetchall()
        )
    finally:
        conn.close()

    assert por_id["C-DUAS-REGRAS"] == esperado
    assert por_case["C-DUAS-REGRAS"] == esperado

    # a ordem exportada é a ordem do classificador: a 1ª regra casada é a
    # que o SQL elege (a de menor `ordem` entre as que casam).
    casam = [
        r
        for r in payload["regras"]
        if set(r["antecedente_serializado"].split(" AND "))
        <= {f"{k}={v}" for k, v in cliente.items()}
    ]
    assert casam, "o cliente da fixture deve casar com alguma regra"
    assert min(r["ordem"] for r in casam) == 1
    assert payload["regras"][0]["consequente"] == esperado
