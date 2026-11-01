"""Exportação das regras de negócio (JSON + SQL) — Fase 3 do benchmark.

O export transforma o classificador associativo-alvo em dois artefatos
executáveis: `outputs/rules_exported.json` (regras legíveis por sistema) e
`outputs/rules_exported.sql` (script SQLite com as regras e uma query de
decisão que classifica um cliente).

Decisões de protocolo (o contrato do export):

1. **Qual modelo é exportado** — o **AC.RankAW Ponderado** (`[GF][PM-SD][TS]`),
   o método proposto e o modelo do portfólio. A escolha é de **protocolo**
   (spec issue #1, histórias 7-8: as regras empacotadas são as do método
   proposto), não de desempenho: os três classificadores associativos
   (`cba`, `ac_ranka`, `ac_rankaw`) são avaliados no MESMO protocolo do
   benchmark (`StratifiedKFold` de 10 dobras, `shuffle=True`,
   `random_state=42`, discretização MDLP ajustada só na dobra de treino —
   anti-leakage) e as métricas de todos ficam registradas em
   `metadados.metricas.modelos_avaliados` para transparência.
2. **Quais regras são exportadas** — o AC.RankAW é retreinado no dataset
   **completo** (MDLP ajustado no conjunto inteiro) e exporta-se o conjunto
   pós-poda [M1] na **ordem do ranking**: é essa ordem que a predição [O]
   percorre (a PRIMEIRA regra que casa vence).
3. **Modelos alternativos** — a seção secundária `modelos_alternativos` do
   JSON traz as regras do **AC.RankA** (mesmo motor, pesos uniformes no
   TOPSIS) para comparação/auditoria. Ela é **informativa**: o SQL exporta
   apenas o modelo principal (AC.RankAW).
4. **Score de ordenação** — o número que justifica a ordem do ranking:
   score TOPSIS no `ac_ranka` (pesos uniformes) e no `ac_rankaw`
   (pesos [PM-SD]); no `cba`, a confiança — critério primário da
   precedência CBA (confiança ↓, suporte ↓, cardinalidade ↑). Em todos os
   casos o score é decrescente na ordem exportada e a 1ª regra tem o maior
   score; **a ordem do ranking é exportada explicitamente** na coluna
   `ordem` (posição da regra no ranking do classificador), e é ela — não o
   score — que a query de decisão do SQL usa (`ORDER BY r.ordem ASC`).
   Assim a decisão do SQL reproduz a predição [O] do classificador também
   quando duas regras têm o mesmo score (no CBA o desempate da precedência
   é suporte/cardinalidade, não o score).
5. **L** — mesma convenção do benchmark: nº de regras pós-poda **+ 1**
   (a classe default). `delta_L_vs_cba` =
   (L_CBA − L_vencedor) / L_CBA × 100, com o CBA retreinado no mesmo
   dataset completo.
6. **Determinismo** — seed 42 em toda a cadeia e **nenhum timestamp** nos
   artefatos: a mesma entrada produz JSON e SQL byte-idênticos. O
   `metadados.id_exportacao` é um identificador determinístico do conteúdo
   (SHA-256 do payload canônico das regras).
7. **Valores discretizados** — as regras casam com os valores gerados pela
   discretização MDLP (ex.: `tenure=tenure_(-∞, 12]`); o cliente consultado
   no SQL precisa passar pelo mesmo pré-processamento.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.eda import ALVO
from src.preprocessing import MDLPDiscretizer
from src.protocolo import (
    MODELOS_ASSOCIATIVOS,
    N_SPLITS,
    RANDOM_STATE,
    avaliar_associativos,
    carregar_dataset_churn,
    delta_l,
    features_e_alvo,
    preparar_dobras,
)
from src.rule_extraction import MAX_ANTECEDENTE

NOME_BASE = "Telco Customer Churn (Kaggle: blastchar/telco-customer-churn)"
# Modelo exportado na Fase 3: o AC.RankAW Ponderado (spec issue #1,
# histórias 7-8 — as regras empacotadas são as do método proposto).
MODELO_EXPORTADO = "ac_rankaw"
# Seção secundária do JSON: modelos alternativos, para comparação/auditoria.
MODELOS_ALTERNATIVOS = ("ac_ranka",)
CAMPOS_REGRA = (
    "id",
    "ordem",
    "antecedente",
    "antecedente_serializado",
    "consequente",
    "suporte",
    "confianca",
    "score",
)


# ---------------------------------------------------------------- dados


def _discretizar(X: pd.DataFrame, y) -> pd.DataFrame:
    """MDLP ajustado no conjunto passado (mesmo caminho do runner)."""
    discretizador = MDLPDiscretizer().fit(X, y)
    return discretizador.fit_transform(X, y)


# ------------------------------------------------------------- avaliação


def avaliar_modelos_associativos(
    df: pd.DataFrame, seed: int = RANDOM_STATE, n_splits: int = N_SPLITS
) -> dict[str, dict]:
    """Métricas dos 3 classificadores associativos no protocolo do benchmark.

    Devolve, por modelo, as médias das dobras (F1-Macro, precisão e recall
    por classe, L) e as séries por dobra (`f1_macro_folds`, `L_folds`).
    O cálculo é o do protocolo comum (`src/protocolo.py`), compartilhado com
    o runner do benchmark.
    """
    X, y = features_e_alvo(df)
    dobras = preparar_dobras(X, y, seed=seed, n_splits=n_splits)
    _, metricas = avaliar_associativos(dobras, MODELOS_ASSOCIATIVOS)
    return metricas


def selecionar_modelo() -> str:
    """Modelo exportado na Fase 3: **AC.RankAW Ponderado** (`[GF][PM-SD][TS]`).

    A escolha é de **protocolo**, não de desempenho (spec issue #1, histórias
    7-8): o empacotamento de regras de negócio entrega o método proposto.
    As métricas dos três classificadores associativos continuam sendo
    avaliadas e registradas em `metadados.metricas.modelos_avaliados`.
    """
    return MODELO_EXPORTADO


# ------------------------------------------------------------- export JSON


def _pares_do_antecedente(antecedente: frozenset[str]) -> list[dict[str, str]]:
    """Itens "coluna=valor" → lista de pares atributo-valor (ordenada)."""
    return [
        {"atributo": item.split("=", 1)[0], "valor": item.split("=", 1)[1]}
        for item in sorted(antecedente)
    ]


def _regras_exportaveis(modelo) -> list[dict]:
    """Regras pós-poda na ordem do ranking, com score de ordenação.

    `ordem` é a posição da regra no ranking do classificador (1 = primeira
    regra da predição [O]) — é ela que o SQL usa para reproduzir a ordem de
    decisão do classificador.
    """
    regras = []
    for posicao, regra in enumerate(modelo.regras_, start=1):
        pares = _pares_do_antecedente(regra.antecedente)
        score = modelo.scores_.get(regra, regra.score)
        regras.append(
            {
                "id": posicao,
                "posicao": posicao,
                "ordem": posicao,
                "antecedente": pares,
                "antecedente_serializado": " AND ".join(sorted(regra.antecedente)),
                "consequente": regra.classe,
                "suporte": round(float(regra.suporte), 6),
                "confianca": round(float(regra.confianca), 6),
                "score": round(float(score if score is not None else 0.0), 6),
            }
        )
    return regras


def _id_exportacao(payload: dict) -> str:
    """SHA-256 (16 hex) do payload canônico — identificador determinístico."""
    canonico = json.dumps(
        {"regras": payload["regras"], "class_default": payload["class_default"]},
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonico).hexdigest()[:16]


def montar_json(
    modelo,
    metricas_cv: dict[str, dict],
    nome_modelo: str,
    caminho_dataset: str,
    seed: int,
    delta_l: float,
    alternativos: dict[str, dict] | None = None,
) -> dict:
    """Monta o payload do `rules_exported.json`.

    `alternativos`: seção secundária `modelos_alternativos` (modelo → bloco
    com `L`, `class_default` e `regras`); informativa, não entra no SQL.
    """
    regras = _regras_exportaveis(modelo)
    class_default = {
        "classe": modelo.classe_default_,
        "descricao": (
            "Classe aplicada quando nenhuma regra do ranking casa com o cliente "
            "(predição [O] — fallback do CBA)."
        ),
    }
    payload = {
        "metadados": {
            "base": NOME_BASE,
            "arquivo_dataset": Path(caminho_dataset).name,
            "alvo": ALVO,
            "seed": int(seed),
            "modelo": nome_modelo,
            "L": int(modelo.L),
            "n_regras": len(regras),
            "protocolo": {
                "validacao": f"StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={seed})",
                "discretizacao": "MDLP (Fayyad-Irani) ajustada por dobra de treino (anti-leakage)",
                "extracao": (
                    f"Apriori (mlxtend), conf-min=0, antecedente com até "
                    f"{MAX_ANTECEDENTE} itens"
                ),
                "poda": (
                    "M1 (cover-style) — a regra que acerta >=1 transação coberta "
                    "entra no classificador e TODAS as transações cobertas são "
                    "deletadas; classe default = maioria do restante (recalculada "
                    "a cada iteração)"
                ),
                "predicao": "[O] — primeira regra do ranking que casa; fallback classe default",
                "selecao_do_modelo": (
                    "AC.RankAW Ponderado — alvo de protocolo do export (spec issue #1, "
                    "histórias 7-8); as métricas dos demais modelos associativos "
                    "ficam em metricas.modelos_avaliados"
                ),
                "score_de_ordenacao": (
                    "TOPSIS (pesos [PM-SD]) no ac_rankaw, TOPSIS (pesos uniformes) "
                    "no ac_ranka, confiança no cba"
                ),
            },
            "metricas": {
                "f1_macro": round(float(metricas_cv[nome_modelo]["f1_macro"]), 6),
                "f1_macro_desvio": round(float(metricas_cv[nome_modelo]["f1_macro_desvio"]), 6),
                "precisao_No": round(float(metricas_cv[nome_modelo]["precisao_No"]), 6),
                "recall_No": round(float(metricas_cv[nome_modelo]["recall_No"]), 6),
                "precisao_Yes": round(float(metricas_cv[nome_modelo]["precisao_Yes"]), 6),
                "recall_Yes": round(float(metricas_cv[nome_modelo]["recall_Yes"]), 6),
                "L": int(modelo.L),
                "L_medio_dobras": round(float(metricas_cv[nome_modelo]["L"]), 6),
                "delta_L_vs_cba": round(float(delta_l), 6),
                "modelos_avaliados": {
                    nome: {
                        "f1_macro": round(float(m["f1_macro"]), 6),
                        "f1_macro_desvio": round(float(m["f1_macro_desvio"]), 6),
                        "L_medio_dobras": round(float(m["L"]), 6),
                    }
                    for nome, m in metricas_cv.items()
                },
            },
        },
        "regras": regras,
        "class_default": class_default,
        "modelos_alternativos": {
            "descricao": (
                "Modelos alternativos, retreinados no mesmo dataset completo e na "
                "mesma ordem de ranking. Seção informativa: o SQL exporta apenas "
                "o modelo principal (AC.RankAW Ponderado)."
            ),
            **{
                nome: {
                    "L": int(bloco["modelo"].L),
                    "class_default": {"classe": bloco["modelo"].classe_default_},
                    "regras": _regras_exportaveis(bloco["modelo"]),
                }
                for nome, bloco in (alternativos or {}).items()
            },
        },
    }
    payload["metadados"]["id_exportacao"] = _id_exportacao(payload)
    return payload


# -------------------------------------------------------------- export SQL


def _sql_texto(valor) -> str:
    """Literal SQL (texto com aspas simples escapadas / real com 6 casas)."""
    return "'" + str(valor).replace("'", "''") + "'"


def _sql_real(valor: float) -> str:
    return f"{float(valor):.6f}"


def _condicao_antecedente(pares: list[dict[str, str]]) -> str:
    """Condição SQL de um antecedente: `coluna = 'valor' AND ...`."""
    return " AND ".join(
        f'c."{par["atributo"]}" = {_sql_texto(par["valor"])}' for par in pares
    )


def _cliente_exemplo(X_disc: pd.DataFrame, modelo) -> dict[str, str]:
    """Cliente fictício que casa com a 1ª regra do ranking.

    Os atributos da 1ª regra recebem o valor da regra; os demais recebem o
    valor mais frequente do treino (empate resolvido alfabeticamente) —
    determinístico.
    """
    valores: dict[str, str] = {}
    if modelo.regras_:
        for par in _pares_do_antecedente(modelo.regras_[0].antecedente):
            valores[par["atributo"]] = par["valor"]
    for coluna in X_disc.columns:
        if coluna in valores:
            continue
        contagem = X_disc[coluna].astype(str).value_counts()
        maximo = contagem.max()
        valores[coluna] = sorted(contagem[contagem == maximo].index)[0]
    return valores


def montar_sql(payload: dict, X_disc: pd.DataFrame, modelo) -> str:
    """Monta o script SQLite (`rules_exported.sql`)."""
    meta = payload["metadados"]
    regras = payload["regras"]
    class_default = payload["class_default"]["classe"]
    exemplo = _cliente_exemplo(X_disc, modelo)

    linhas: list[str] = []
    linhas.append("-- " + "=" * 68)
    linhas.append("-- Export das regras de negócio — AC.RankAW Churn Benchmark (Fase 3)")
    linhas.append("-- Gerado por: python -m src.run_benchmark --phase 3")
    linhas.append(f"-- Base: {meta['base']}")
    linhas.append(
        f"-- Alvo: {meta['alvo']} | Seed: {meta['seed']} | "
        f"Modelo: {meta['modelo']} | L: {meta['L']} "
        f"({meta['n_regras']} regras + classe default)"
    )
    linhas.append(f"-- F1-Macro médio ({N_SPLITS} dobras): {meta['metricas']['f1_macro']}")
    linhas.append(f"-- id_exportacao: {meta['id_exportacao']}")
    linhas.append("--")
    linhas.append("-- As regras casam com valores DISCRETIZADOS (MDLP Fayyad-Irani):")
    linhas.append("-- o cliente precisa passar pelo mesmo pré-processamento antes da consulta.")
    linhas.append("-- " + "=" * 68)
    linhas.append("")
    linhas.append("DROP TABLE IF EXISTS regras;")
    linhas.append("CREATE TABLE regras (")
    linhas.append("    id INTEGER PRIMARY KEY,")
    linhas.append("    ordem INTEGER NOT NULL,")
    linhas.append("    antecedente_serializado TEXT NOT NULL,")
    linhas.append("    classe_predita TEXT NOT NULL,")
    linhas.append("    suporte REAL NOT NULL,")
    linhas.append("    confianca REAL NOT NULL,")
    linhas.append("    score REAL NOT NULL")
    linhas.append(");")
    linhas.append("")
    linhas.append("BEGIN TRANSACTION;")
    for regra in regras:
        linhas.append(
            "INSERT INTO regras "
            "(id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) "
            f"VALUES ({regra['id']}, "
            f"{regra['ordem']}, "
            f"{_sql_texto(regra['antecedente_serializado'])}, "
            f"{_sql_texto(regra['consequente'])}, "
            f"{_sql_real(regra['suporte'])}, "
            f"{_sql_real(regra['confianca'])}, "
            f"{_sql_real(regra['score'])});"
        )
    linhas.append("COMMIT;")
    linhas.append("")
    linhas.append("-- " + "-" * 68)
    linhas.append("-- Cliente de exemplo: casa com a 1ª regra do ranking exportado.")
    linhas.append("-- " + "-" * 68)
    colunas = list(X_disc.columns)
    linhas.append("DROP TABLE IF EXISTS cliente_exemplo;")
    linhas.append("CREATE TABLE cliente_exemplo (")
    linhas.append("    cliente_id TEXT PRIMARY KEY,")
    for posicao, coluna in enumerate(colunas):
        virgula = "," if posicao < len(colunas) - 1 else ""
        linhas.append(f'    "{coluna}" TEXT{virgula}')
    linhas.append(");")
    linhas.append("")
    valores_cliente = ", ".join(
        [_sql_texto("EXEMPLO-1")] + [_sql_texto(exemplo[c]) for c in colunas]
    )
    quoted = ", ".join(["cliente_id"] + [f'"{c}"' for c in colunas])
    linhas.append(f"INSERT INTO cliente_exemplo ({quoted}) VALUES ({valores_cliente});")
    linhas.append("")
    linhas.append("-- " + "-" * 68)
    linhas.append("-- Query de decisão (predição [O]): a PRIMEIRA regra do ranking que")
    linhas.append("-- casa vence; se nenhuma casa, vale a classe default.")
    linhas.append("-- " + "-" * 68)
    linhas.append("DROP VIEW IF EXISTS decisao_cliente;")
    linhas.append("CREATE VIEW decisao_cliente AS")
    linhas.append("SELECT")
    linhas.append("    c.cliente_id,")
    linhas.append("    CASE")
    for regra in regras:
        linhas.append(
            f"        WHEN {_condicao_antecedente(regra['antecedente'])} "
            f"THEN {_sql_texto(regra['consequente'])}"
        )
    linhas.append(f"        ELSE {_sql_texto(class_default)}")
    linhas.append("    END AS classe_predita")
    linhas.append("FROM cliente_exemplo c;")
    linhas.append("")
    linhas.append("-- Mesma decisão em forma relacional: filtra as regras que casam e")
    linhas.append("-- ordena pela POSIÇÃO NO RANKING do classificador (coluna `ordem`,")
    linhas.append("-- ascendente) — a 1ª regra que casa vence, exatamente como na")
    linhas.append("-- predição [O]. O `score` fica na tabela para auditoria, mas não")
    linhas.append("-- define a ordem (no CBA o desempate da precedência é suporte e")
    linhas.append("-- cardinalidade, não o score).")
    linhas.append("DROP VIEW IF EXISTS decisao_cliente_por_score;")
    linhas.append("CREATE VIEW decisao_cliente_por_score AS")
    linhas.append("SELECT")
    linhas.append("    c.cliente_id,")
    linhas.append("    COALESCE(")
    linhas.append("        (")
    linhas.append("            SELECT r.classe_predita")
    linhas.append("            FROM regras r")
    if regras:
        linhas.append("            WHERE " + "\n               OR ".join(
            f"(r.id = {regra['id']} AND {_condicao_antecedente(regra['antecedente'])})"
            for regra in regras
        ))
    else:
        linhas.append("            WHERE 0")
    linhas.append("            ORDER BY r.ordem ASC")
    linhas.append("            LIMIT 1")
    linhas.append("        ),")
    linhas.append(f"        {_sql_texto(class_default)}")
    linhas.append("    ) AS classe_predita")
    linhas.append("FROM cliente_exemplo c;")
    linhas.append("")
    linhas.append("-- Saída da query de decisão de exemplo:")
    linhas.append("SELECT * FROM decisao_cliente;")
    linhas.append("")
    return "\n".join(linhas)


# ------------------------------------------------------------------ façade


def exportar_regras(
    caminho_dataset: str,
    caminho_json: str,
    caminho_sql: str,
    seed: int = RANDOM_STATE,
) -> dict:
    """Fase 3: exporta as regras do AC.RankAW Ponderado (JSON + SQL).

    Avalia os três classificadores associativos no protocolo do benchmark
    (métricas em `metadados.metricas.modelos_avaliados`), retreina o
    **AC.RankAW Ponderado** (alvo de protocolo) e o **AC.RankA** (seção
    secundária `modelos_alternativos`) no dataset completo, e escreve
    `caminho_json` e `caminho_sql` (diretórios criados se necessário).

    Devolve o payload JSON exportado.
    """
    df = carregar_dataset_churn(caminho_dataset)
    metricas_cv = avaliar_modelos_associativos(df, seed=seed)
    nome_modelo = selecionar_modelo()

    X, y = features_e_alvo(df)
    X_disc = _discretizar(X, y)
    modelo = MODELOS_ASSOCIATIVOS[nome_modelo]().fit(X_disc, y)

    modelo_cba = MODELOS_ASSOCIATIVOS["cba"]().fit(X_disc, y)
    delta_l_vs_cba = delta_l(int(modelo_cba.L), int(modelo.L))

    alternativos = {
        nome: {"modelo": MODELOS_ASSOCIATIVOS[nome]().fit(X_disc, y)}
        for nome in MODELOS_ALTERNATIVOS
    }

    payload = montar_json(
        modelo=modelo,
        metricas_cv=metricas_cv,
        nome_modelo=nome_modelo,
        caminho_dataset=caminho_dataset,
        seed=seed,
        delta_l=delta_l_vs_cba,
        alternativos=alternativos,
    )
    sql = montar_sql(payload, X_disc, modelo)

    arq_json, arq_sql = Path(caminho_json), Path(caminho_sql)
    arq_json.parent.mkdir(parents=True, exist_ok=True)
    arq_sql.parent.mkdir(parents=True, exist_ok=True)
    arq_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    arq_sql.write_text(sql, encoding="utf-8")
    return payload
