"""Exportação das regras de negócio (JSON + SQL) — Fase 3 do benchmark.

O export transforma o classificador associativo vencedor em dois artefatos
executáveis: `outputs/rules_exported.json` (regras legíveis por sistema) e
`outputs/rules_exported.sql` (script SQLite com as regras e uma query de
decisão que classifica um cliente).

Decisões de protocolo (o contrato do export):

1. **Qual modelo é exportado** — os três classificadores associativos do
   portfólio (`cba`, `ac_ranka`, `ac_rankaw`) são avaliados com o MESMO
   protocolo do benchmark (`StratifiedKFold` de 10 dobras, `shuffle=True`,
   `random_state=42`, discretização MDLP ajustada só na dobra de treino —
   anti-leakage) e **vence o de maior F1-Macro médio** nas dobras. Empate
   de F1-Macro é resolvido pelo menor L médio (modelo mais compacto) e,
   persistindo o empate, pelo nome do modelo em ordem alfabética
   (determinismo). A escolha é registrada em `metadados.metricas`.
2. **Quais regras são exportadas** — o modelo vencedor é retreinado no
   dataset **completo** (MDLP ajustado no conjunto inteiro) e exporta-se o
   conjunto pós-poda [M1] na **ordem do ranking**: é essa ordem que a
   predição [O] percorre (a PRIMEIRA regra que casa vence).
3. **Score de ordenação** — o número que justifica a ordem do ranking:
   score TOPSIS no `ac_ranka` (pesos uniformes) e no `ac_rankaw`
   (pesos [PM-SD]); no `cba`, a confiança — critério primário da
   precedência CBA (confiança ↓, suporte ↓, cardinalidade ↑). Em todos os
   casos o score é decrescente na ordem exportada e a 1ª regra tem o maior
   score (o SQL desempata por `id`).
4. **L** — mesma convenção do benchmark: nº de regras pós-poda **+ 1**
   (a classe default). `delta_L_vs_cba` =
   (L_CBA − L_vencedor) / L_CBA × 100, com o CBA retreinado no mesmo
   dataset completo.
5. **Determinismo** — seed 42 em toda a cadeia e **nenhum timestamp** nos
   artefatos: a mesma entrada produz JSON e SQL byte-idênticos. O
   `metadados.id_exportacao` é um identificador determinístico do conteúdo
   (SHA-256 do payload canônico das regras).
6. **Valores discretizados** — as regras casam com os valores gerados pela
   discretização MDLP (ex.: `tenure=tenure_(-∞, 12]`); o cliente consultado
   no SQL precisa passar pelo mesmo pré-processamento.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.associative import ACRankA, ACRankAW, CBA
from src.eda import ALVO, carregar_dataset
from src.evaluation import f1_macro, precision_recall_por_classe
from src.preprocessing import MDLPDiscretizer

RANDOM_STATE = 42
N_SPLITS = 10
ATRIBUTOS_IGNORADOS = ["customerID"]
NOME_BASE = "Telco Customer Churn (Kaggle: blastchar/telco-customer-churn)"
MODELOS_ASSOCIATIVOS = {"cba": CBA, "ac_ranka": ACRankA, "ac_rankaw": ACRankAW}
MENSAGEM_DATASET_AUSENTE = """\
Dataset não encontrado em {caminho}.
Baixe o Telco Customer Churn com:

    python -c "import kagglehub; kagglehub.dataset_download('blastchar/telco-customer-churn')"

e copie o CSV 'WA_Fn-UseC_-Telco-Customer-Churn.csv' para data/raw/.
"""
CAMPOS_REGRA = (
    "id",
    "antecedente",
    "antecedente_serializado",
    "consequente",
    "suporte",
    "confianca",
    "score",
)


# ---------------------------------------------------------------- dados


def _carregar(caminho_dataset: str) -> pd.DataFrame:
    """Dataset de churn já validado (EDA) e sem `TotalCharges` nulo."""
    caminho = Path(caminho_dataset)
    if not caminho.exists():
        raise FileNotFoundError(MENSAGEM_DATASET_AUSENTE.format(caminho=caminho_dataset))
    df = carregar_dataset(caminho_dataset).dropna(subset=["TotalCharges"]).reset_index(drop=True)
    return df


def _X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[ALVO] + ATRIBUTOS_IGNORADOS)
    return X, df[ALVO]


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
    """
    X, y = _X_y(df)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    por_dobra: dict[str, list[dict]] = {nome: [] for nome in MODELOS_ASSOCIATIVOS}

    for idx_treino, idx_teste in skf.split(X, y):
        X_treino, X_teste = X.iloc[idx_treino], X.iloc[idx_teste]
        y_treino, y_teste = y.iloc[idx_treino], y.iloc[idx_teste]
        discretizador = MDLPDiscretizer().fit(X_treino, y_treino)
        X_treino_disc = discretizador.fit_transform(X_treino, y_treino)
        X_teste_disc = discretizador.transform(X_teste)

        for nome, classe in MODELOS_ASSOCIATIVOS.items():
            modelo = classe().fit(X_treino_disc, y_treino)
            y_pred = modelo.predict(X_teste_disc)
            pr = precision_recall_por_classe(list(y_teste), list(y_pred))
            pr_no, pr_yes = pr.get("No"), pr.get("Yes")
            por_dobra[nome].append(
                {
                    "f1_macro": f1_macro(list(y_teste), list(y_pred)),
                    "precisao_No": pr_no[0] if pr_no else 0.0,
                    "recall_No": pr_no[1] if pr_no else 0.0,
                    "precisao_Yes": pr_yes[0] if pr_yes else 0.0,
                    "recall_Yes": pr_yes[1] if pr_yes else 0.0,
                    "L": int(modelo.L),
                }
            )

    metricas: dict[str, dict] = {}
    for nome, dobras in por_dobra.items():
        medias = {
            campo: float(np.mean([d[campo] for d in dobras]))
            for campo in ("f1_macro", "precisao_No", "recall_No", "precisao_Yes", "recall_Yes", "L")
        }
        metricas[nome] = {
            **medias,
            "f1_macro_desvio": float(np.std([d["f1_macro"] for d in dobras], ddof=1)),
            "f1_macro_folds": [round(float(d["f1_macro"]), 6) for d in dobras],
            "L_folds": [int(d["L"]) for d in dobras],
        }
    return metricas


def selecionar_modelo(metricas: dict[str, dict]) -> str:
    """Modelo exportado: maior F1-Macro médio; empate ⇒ menor L; ⇒ alfabético."""
    return min(
        metricas,
        key=lambda nome: (
            -round(float(metricas[nome]["f1_macro"]), 12),
            round(float(metricas[nome]["L"]), 12),
            nome,
        ),
    )


# ------------------------------------------------------------- export JSON


def _pares_do_antecedente(antecedente: frozenset[str]) -> list[dict[str, str]]:
    """Itens "coluna=valor" → lista de pares atributo-valor (ordenada)."""
    return [
        {"atributo": item.split("=", 1)[0], "valor": item.split("=", 1)[1]}
        for item in sorted(antecedente)
    ]


def _regras_exportaveis(modelo) -> list[dict]:
    """Regras pós-poda na ordem do ranking, com score de ordenação."""
    regras = []
    for posicao, regra in enumerate(modelo.regras_, start=1):
        antecedente, classe, suporte, confianca = regra
        pares = _pares_do_antecedente(antecedente)
        regras.append(
            {
                "id": posicao,
                "posicao": posicao,
                "antecedente": pares,
                "antecedente_serializado": " AND ".join(sorted(antecedente)),
                "consequente": classe,
                "suporte": round(float(suporte), 6),
                "confianca": round(float(confianca), 6),
                "score": round(float(modelo.scores_.get(regra, 0.0)), 6),
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
) -> dict:
    """Monta o payload do `rules_exported.json`."""
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
                "extracao": "Apriori (mlxtend), conf-min=0, antecedente com até 4 itens",
                "poda": "M1 (cover-style) — transações acertadas são deletadas",
                "predicao": "[O] — primeira regra do ranking que casa; fallback classe default",
                "selecao_do_modelo": (
                    "maior F1-Macro médio nas dobras; empate ⇒ menor L médio; "
                    "empate ⇒ nome do modelo em ordem alfabética"
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
        for par in _pares_do_antecedente(modelo.regras_[0][0]):
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
    linhas.append(f"-- F1-Macro médio (10 dobras): {meta['metricas']['f1_macro']}")
    linhas.append(f"-- id_exportacao: {meta['id_exportacao']}")
    linhas.append("--")
    linhas.append("-- As regras casam com valores DISCRETIZADOS (MDLP Fayyad-Irani):")
    linhas.append("-- o cliente precisa passar pelo mesmo pré-processamento antes da consulta.")
    linhas.append("-- " + "=" * 68)
    linhas.append("")
    linhas.append("DROP TABLE IF EXISTS regras;")
    linhas.append("CREATE TABLE regras (")
    linhas.append("    id INTEGER PRIMARY KEY,")
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
            "(id, antecedente_serializado, classe_predita, suporte, confianca, score) "
            f"VALUES ({regra['id']}, "
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
    linhas.append("-- ordena por score (decrescente) — o score é o critério de ordenação")
    linhas.append("-- do modelo exportado; a primeira (maior score) vence.")
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
    linhas.append("            ORDER BY r.score DESC, r.id ASC")
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
    """Fase 3: avalia, seleciona e exporta as regras (JSON + SQL).

    Devolve o payload JSON exportado. Escreve `caminho_json` e
    `caminho_sql` (diretórios criados se necessário).
    """
    df = _carregar(caminho_dataset)
    metricas_cv = avaliar_modelos_associativos(df, seed=seed)
    nome_modelo = selecionar_modelo(metricas_cv)

    X, y = _X_y(df)
    X_disc = _discretizar(X, y)
    modelo = MODELOS_ASSOCIATIVOS[nome_modelo]().fit(X_disc, y)

    modelo_cba = MODELOS_ASSOCIATIVOS["cba"]().fit(X_disc, y)
    delta_l = (int(modelo_cba.L) - int(modelo.L)) / int(modelo_cba.L) * 100.0

    payload = montar_json(
        modelo=modelo,
        metricas_cv=metricas_cv,
        nome_modelo=nome_modelo,
        caminho_dataset=caminho_dataset,
        seed=seed,
        delta_l=delta_l,
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
