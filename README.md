# AC.RankAW Churn Benchmark

[![CI de testes](https://github.com/CauanGS/benchmark-classificadores-acrankaw-churn/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/CauanGS/benchmark-classificadores-acrankaw-churn/actions/workflows/ci.yml)

**Benchmark de classificadores interpretáveis aplicando o método `AC.RankAW Ponderado` (`[GF][PM-SD][TS]`) à predição de churn** no *Telco Customer Churn* (Kaggle).

O repositório compara um modelo caixa-preta (`XGBoost + SHAP`) com três classificadores associativos — `CBA`, `AC.RankA` e o proposto `AC.RankAW Ponderado` — e entrega as regras de decisão do modelo vencedor prontas para uso em sistema de negócio (JSON + SQL).

---

## 1. O problema: churn de clientes

Churn é a saída do cliente. No *Telco Customer Churn* (7.032 clientes válidos, 21 atributos) **26,58% dos clientes cancelaram** (1.869 churn × 5.163 não-churn) — um problema de classificação **desbalanceado**, em que um modelo que sempre responde "não vai churn" já fica em F1-Macro ≈ 0,42.

Para o negócio, não basta *acertar*: é preciso **explicar**. Um modelo de gradient boosting diz *quem* vai cancelar, mas a decisão de retenção (oferta, desconto, contato) precisa de uma justificativa auditável — "por que este cliente foi classificado como churn?".

## 2. Classificação associativa: o modelo é a regra

Em vez de ajustar milhões de parâmetros, a classificação associativa **extrai regras `IF ... THEN`** do histórico de clientes:

```text
SE Contract = Month-to-month E InternetService = Fiber optic E OnlineSecurity = No
   E PaymentMethod = Electronic check ENTÃO Churn = Yes
```

O classificador **é** a lista ordenada de regras: cada regra tem suporte (quanto do histórico ela cobre), confiança (acerto quando casa) e um score de ordenação. A auditoria é nativa — não há Explainability pós-hoc, e qualquer analista de negócio consegue ler, ajustar e reimplantar as regras.

O pipeline do paper (Dall'Agnol & Carvalho, IEEE Access 2024) tem quatro etapas: **extração** (Apriori) → **ordenação** (precedência ou TOPSIS) → **poda M1** (descarta as regras que não acertam nenhuma transação) → **predição [O]** (a primeira regra do ranking que casa vence; senão, a classe default).

## 3. Os modelos do portfólio

| Modelo | Categoria | Descrição | Papel no experimento |
| :--- | :--- | :--- | :--- |
| **dummy_no** | Baseline trivial | Sempre responde "No" | Piso de desempenho (classe majoritária) |
| **XGBoost + SHAP** | Caixa-preta | Gradient boosting + explicações SHAP | Baseline de mercado (alto desempenho preditivo) |
| **CBA** | Associativo | Liu et al. (1998): precedência (confiança ↓, suporte ↓, cardinalidade ↑), poda M1, predição [O] | Baseline clássico da literatura de regras |
| **AC.RankA** | Associativo | Mesma extração/poda do CBA, ordenação por **TOPSIS** sobre as 13 MOs `[GF]` com pesos **uniformes** (Dall'Agnol & Carvalho, 2024) | Baseline de agregação multicritério |
| **AC.RankAW Ponderado** | Associativo (proposta) | AC.RankA com pesos **`[PM-SD]`** (desvio padrão por coluna, Algoritmo 1 do relatório) no TOPSIS — `[GF][PM-SD][TS]` | **Proposta do autor (foco do estudo)** |

`L` é o tamanho do modelo (nº de regras pós-poda + 1 classe default) e `ΔL` a taxa de redução de regras frente ao CBA (SDD 5.2):

```text
ΔL = (L_CBA − L_AC.RankAW) / L_CBA × 100%
```

## 4. Pipeline

```text
 [ Base Telco Churn ]
           │
           ▼
 [ Pré-processamento & Discretização ]
           │
     ┌─────┴──────────────────────────────┐
     ▼                                    ▼
 [ Ramo 1: Dados Contínuos/Dummies ]   [ Ramo 2: Dados Categóricos Discretizados ]
     │                                    │
     ▼                                    ▼
 [ XGBoost + SHAP ]                    [ Extração Apriori / Regras RACs ]
                                          │
                                          ├───────────────────┬───────────────────┐
                                          ▼                   ▼                   ▼
                                       [ CBA ]          [ AC.RankA ]   [ AC.RankAW Ponderado ]
                                     (Baseline)          (Padrão)      ([GF][PM-SD][TS])
                                          │                   │                   │
     └────────────────────────────────────┴───────────────────┴───────────────────┘
                                          │
                                          ▼
                              [ Avaliação & Benchmark ]
                         (F1-Macro, Tamanho L, Auditoria)
```

Protocolo: validação cruzada estratificada em **10 dobras** (`shuffle=True`, `random_state=42`), discretização **MDLP (Fayyad–Irani)** ajustada **só na dobra de treino** (anti-leakage) e, no XGBoost, busca *nested* de hiperparâmetros (`RandomizedSearchCV`, 30 iters, CV interna de 5 dobras, `scoring='f1_macro'`).

## 5. Tabela comparativa final (Fase 1)

Números reais do run de referência (`python -m src.run_benchmark --phase 1` → `outputs/benchmark_results.csv`), médias das 10 dobras:

| Modelo | F1-Macro | Precisão (No) | Recall (No) | Precisão (Yes) | Recall (Yes) | L | ΔL (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dummy_no (baseline) | 0.4234 | 0.7342 | 1.0000 | 0.0000 | 0.0000 | 0.0 | — |
| XGBoost + SHAP | 0.7265 | 0.8392 | 0.9028 | 0.6609 | 0.5222 | 1.0 | — |
| CBA | 0.4262 | 0.7347 | 1.0000 | 0.3000 | 0.0027 | 319.1 | — |
| AC.RankA | 0.7196 | 0.8783 | 0.7856 | 0.5427 | 0.6988 | 227.9 | — |
| AC.RankAW Ponderado | 0.7104 | 0.8346 | 0.8844 | 0.6186 | 0.5158 | 237.3 | 25.63 |

Leitura da tabela:

- O `dummy_no` fixa o piso do problema (F1-Macro 0,4234) — sem ele, qualquer número de modelo parece bom.
- O `CBA` fica praticamente no piso (0,4262) e é o **maior** modelo em número de regras (L 319,1): a precedência CBA (confiança ↓) coloca no topo as regras de "Não churn" mais confiantes, e a predição `[O]` (primeira regra que casa) acaba classificando quase tudo como "No" (recall de "Yes" = 0,0027).
- Já os classificadores com ordenação **TOPSIS** (`AC.RankA` e `AC.RankAW`) alcançam **F1-Macro de 0,7196 e 0,7104** — a 0,7 e 1,6 ponto do XGBoost (0,7265) — com um modelo que é uma lista de 227–237 regras auditáveis.
- `L = 1` no XGBoost significa apenas "não é um modelo de regras" (a complexidade dele está nos 100s de árvores, não em L).

## 6. Fase 2 — AC.RankAW Ponderado: menos regras, mesmo desempenho

A Fase 2 aplica o método proposto (`[GF][PM-SD][TS]`: pesos por desvio padrão sobre as 13 MOs do grupo `[GF]`) e mede a redução de regras frente ao CBA. Números reais de `outputs/benchmark_results_phase2.csv` (mesma pipeline da Fase 1, seed 42 — os dois CSVs são byte-idênticos neste run):

| Métrica | CBA | AC.RankA | AC.RankAW Ponderado |
| :--- | ---: | ---: | ---: |
| F1-Macro (média 10 dobras) | 0.4262 | 0.7196 | 0.7104 |
| L (média 10 dobras) | 319.1 | 227.9 | 237.3 |
| ΔL vs CBA (SDD 5.2) | — | — | **25,63%** |

- **ΔL > 0 (critério de sucesso do SDD 5.3):** o AC.RankAW entrega **25,63% menos regras** que o CBA (L 319,1 → 237,3).
- **Equivalência de F1-Macro vs AC.RankA:** 0,7104 (AC.RankAW) contra 0,7196 (AC.RankA) — diferença de **0,0092** absoluto (1,3% relativo), com performances por dobra na mesma faixa (0,6882–0,7279 contra 0,6903–0,7397). O modelo ponderado mantém o desempenho preditivo do AC.RankA padrão, que é exatamente o critério "manter a equivalência em F1-Macro" da Fase 2.
- **Transparência do número:** a redução é medida **contra o CBA**, como define o SDD (5.2); frente ao AC.RankA (mesmo motor, pesos uniformes) o AC.RankAW tem 237,3 contra 227,9 regras (+4,1%), pois a ponderação `[PM-SD]` altera o ranking e, portanto, o conjunto sobrevivido à poda M1.

O trade-off do método é, portanto, o esperado: a ordenação ponderada troca uma fração de desempenho (0,009 de F1-Macro) por um classificador mais enxuto frente ao CBA de referência.

## 7. Fase 3 — export das regras de negócio

`python -m src.run_benchmark --phase 3` avalia os três classificadores associativos no mesmo protocolo do benchmark (métricas registradas em `metadados.metricas.modelos_avaliados`), retreina o **AC.RankAW Ponderado** no dataset completo e grava:

| Artefato | Conteúdo |
| :--- | :--- |
| `outputs/rules_exported.json` | metadados (base, alvo, seed, modelo, `L`, métricas, protocolo) + lista de regras do AC.RankAW (antecedente como pares atributo-valor, consequente, suporte, confiança, score de ordenação) + bloco `class_default` + seção secundária `modelos_alternativos` (regras do AC.RankA, para comparação) |
| `outputs/rules_exported.sql` | script **executável em SQLite**: tabela `regras` (`id`, `antecedente_serializado`, `classe_predita`, `suporte`, `confianca`, `score`), `INSERT`s das regras, tabela de cliente de exemplo e as queries de decisão |

**O modelo exportado é o AC.RankAW Ponderado** (`[GF][PM-SD][TS]`) — a escolha é de **protocolo** (spec issue #1, histórias 7-8: o empacotamento de regras de negócio entrega o método proposto), e não de desempenho: as métricas de `cba`, `ac_ranka` e `ac_rankaw` ficam registradas no JSON para transparência. O AC.RankA aparece apenas como modelo alternativo, na seção informativa `modelos_alternativos` (o SQL exporta só o AC.RankAW).

O SQL traz duas formas equivalentes da mesma decisão de negócio (predição `[O]`):

```sql
-- (a) CASE WHEN: a primeira regra do ranking que casa vence
SELECT cliente_id, CASE WHEN ... THEN 'Yes' ... ELSE 'No' END AS classe_predita
FROM cliente_exemplo;

-- (b) forma relacional: filtra as regras que casam e ordena por score
SELECT r.classe_predita FROM regras r WHERE (...) ORDER BY r.score DESC, r.id ASC LIMIT 1;
```

Uso:

```bash
sqlite3 saida.db < outputs/rules_exported.sql   # cria as tabelas e roda a query de exemplo
```

As regras casam com **valores discretizados** (MDLP): o cliente consultado precisa passar pelo mesmo pré-processamento antes da consulta.

## 8. Exemplo de regras de negócio

Trecho real de `outputs/rules_exported.json` (top-ranked do modelo exportado):

```text
#1  Contract=Two year AND TotalCharges=TotalCharges_(347.9, 3273.68]                 => No
    suporte 0,1013 | confiança 0,9875 | score 0,7605
#2  Contract=Two year AND PaperlessBilling=No AND PhoneService=Yes AND SeniorCitizen=0 => No
    suporte 0,1075 | confiança 0,9857 | score 0,7376
#3  Contract=Two year AND PaperlessBilling=No AND SeniorCitizen=0                    => No
    suporte 0,1193 | confiança 0,9847 | score 0,7258
```

E as regras de churn (`Churn = Yes`) mais bem posicionadas do ranking:

```text
#27 Contract=Month-to-month AND InternetService=Fiber optic AND OnlineSecurity=No
    AND PaymentMethod=Electronic check                                               => Yes
    suporte 0,1018 | confiança 0,6348 | score 0,4462
#30 Contract=Month-to-month AND InternetService=Fiber optic AND OnlineSecurity=No
    AND TechSupport=No                                                               => Yes
    suporte 0,1315 | confiança 0,6070 | score 0,4443
```

Leitura de negócio: **cliente de contrato mensal, fibra óptica, sem segurança online e sem suporte técnico (ou pagando por boleto eletrônico) é o perfil de churn** — e **cliente de contrato de dois anos é cliente retido**. São Segmentos acionáveis por um time de CRM, com a justificativa embutida na própria regra (`suporte` = porte do segmento, `confiança` = taxa de acerto da regra, `score` = prioridade no ranking).

## 9. Como reproduzir

```bash
# 1) ambiente (Python 3.10)
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2) dataset (Kaggle, via kagglehub)
mkdir -p data/raw
python -c "import kagglehub; kagglehub.dataset_download('blastchar/telco-customer-churn')"
cp <dir-do-download>/WA_Fn-UseC_-Telco-Customer-Churn.csv data/raw/

# 3) benchmark (Fase 1: CSV + artefatos SHAP | Fase 2: CSV com ΔL)
python -m src.run_benchmark --phase 1
python -m src.run_benchmark --phase 2

# 4) export das regras de negócio (Fase 3)
python -m src.run_benchmark --phase 3

# 5) testes
python -m pytest tests/ -q
```

Cada fase completa leva ~15 min (o XGBoost faz busca *nested* de 30 iters em cada uma das 10 dobras; a Fase 3 leva ~10 min, sem XGBoost).

## 10. Artefatos e reprodutibilidade

- Toda a cadeia usa **seed 42** (`random_state` do NumPy/scikit-learn, split estratificado, busca de hiperparâmetros) — o export não carrega timestamp e produz JSON/SQL **byte-idênticos** entre execuções (garantido por teste).
- `outputs/` é **gitignored** (dados e artefatos pesados não entram no repo): os CSVs de benchmark, os PNGs de SHAP (`outputs/shap/`) e os exports JSON/SQL são **reproduzíveis pelo comando** acima, a partir do dataset público.
- Decisões de projeto e protocolo: [`docs/DECISIONS.md`](docs/DECISIONS.md) · desenho do experimento: [`software_design_document_sdd_churn_ac_rankaw_ponderado.md`](software_design_document_sdd_churn_ac_rankaw_ponderado.md).
