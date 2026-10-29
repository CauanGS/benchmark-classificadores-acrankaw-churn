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

from src.rule_export import exportar_regras, selecionar_modelo


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
    assert set(payload) == {"metadados", "regras", "class_default"}

    meta = payload["metadados"]
    for campo in ("base", "alvo", "seed", "modelo", "L", "metricas"):
        assert campo in meta, f"metadados sem o campo '{campo}'"
    assert meta["alvo"] == "Churn"
    assert meta["seed"] == 42
    assert meta["modelo"] in {"cba", "ac_ranka", "ac_rankaw"}
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


def test_selecao_de_modelo_segue_a_decisao_documentada():
    """Maior F1-Macro médio; empate ⇒ menor L; empate ⇒ alfabético."""
    metricas = {
        "cba": {"f1_macro": 0.80, "L": 10.0},
        "ac_ranka": {"f1_macro": 0.82, "L": 12.0},
        "ac_rankaw": {"f1_macro": 0.82, "L": 8.0},
    }
    assert selecionar_modelo(metricas) == "ac_rankaw"

    empate_f1 = {
        "cba": {"f1_macro": 0.81, "L": 5.0},
        "ac_ranka": {"f1_macro": 0.81, "L": 5.0},
    }
    assert selecionar_modelo(empate_f1) == "ac_ranka"


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
