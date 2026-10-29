# Registro de Decisões do Projeto — AC.RankAW Churn Benchmark

**Versão:** 1.0 · **Data:** 2026-10-28 · **Sessão:** Design/grilling inicial

Este documento consolida as decisões tomadas na sessão de design, a partir do
SDD (`software_design_document_sdd_churn_ac_rankaw_ponderado.md`) e do
relatório de pesquisa (`relatorio-iniciacao-cientifica.pdf`).

## 1. Objetivo

Repositório **público de portfólio** com resultados reproduzíveis do benchmark
de classificadores interpretáveis e da aplicação do `AC.RankAW Ponderado`
(`[GF][PM-SD][TS]`) na predição de churn. **Não é para transformar em artigo.**

## 2. Fontes de verdade do método

| Artefato | Papel |
| :--- | :--- |
| Paper Dall'Agnol & Carvalho (2024), IEEE Access — DOI `10.1109/ACCESS.2024.3419130` | Definição do `AC.RankA`, grupo de MOs `[GF]`, protocolo de extração, poda e predição do CBA, equações do TOPSIS |
| Relatório de iniciação científica (PDF local) | Ponderação `[PM-SD]` (STD weights do PyMCDM, Algoritmo 1), seleção da combinação `[GF][PM-SD][TS]` |
| SDD (MD local) | Estrutura do repositório, milestones, escopo do benchmark |

## 3. Decisões (numeração da sessão de grilling)

| # | Decisão | Escolha |
| :-- | :------ | :------ |
| Q1 | Repositório | Público, portfólio, reproduzível |
| Q2/Q3 | Implementação | **Do zero**, guiada pelo paper + relatório |
| Q4 | Dataset | Telco Customer Churn via `kagglehub` (`blastchar/telco-customer-churn`) → `data/raw/` |
| Q5 | Extração de regras | `mlxtend` (Apriori); poda/predição do CBA implementadas à mão |
| Q6 | Testes estatísticos | **Não** nesta etapa (future todo) |
| Q7 | Arquitetura | Lógica em `src/` testável + notebooks finos |
| Q8 | XGBoost | Busca leve (RandomizedSearch, 30 iters, *nested* 5-fold interno, `scoring='f1_macro'`) |
| Q9 | Discretização | **Fayyad–Irani (MDLP)** replicando o paper — per-fold de treino (anti-leakage), atributos single-valued pós-discretização removidos |
| Q10 | Protocolo de extração | Apriori, `conf-min=0`, `sup-min` calibrado por fold (≤ 10.000 regras, início 10%, decaimento ×0,5), antecedente máx. 4 itens (itemset máx. 5) |
| Q11 | Ponderação/MCDA | `pymcdm==1.2.0` (última compatível com Python 3.10; ≥1.3.x exige 3.11) para `[PM-SD]` (`pymcdm.weights.standard_deviation_weights`, verificado contra o exemplo do relatório) e `[TS]` — fidelidade máxima ao relatório |
| Q12 | Idioma | **Português** no README/docs/gráficos |
| Q13 | Entrega Fase 3 | `outputs/rules_exported.json` **+ SQL** para integração de negócio |
| Q14 | Ambiente | `venv` + `requirements.txt` fixado (Python 3.10), pytest, GitHub Actions CI, seed **42** |
| Q15 | MDLP | Implementação própria testável |
| Q16 | Busca nested | Confirmada |
| Q17 | Artefatos SHAP | Beeswarm + bar + top-10 por classe (PNG em `outputs/shap/`) |
| Q18 | Runner | CLI `python -m src.run_benchmark --phase 1|2` |

## 4. Fatos do paper indispensáveis para a implementação

1. **Grupo `[GF]`** (Yang & Cui, dados desbalanceados; 13 MOs):
   Odd Multiplier, Complement Class Support, Conviction, Yule-Q,
   Sebag–Schoenauer, Yule-Y, Odds Ratio, Confirm Causal, Confirmed
   Confidence Causal, Example and Counterexample Rate, Ganascia, J-Measure,
   Confidence (fórmulas no Apêndice B do paper).
2. **TOPSIS**: normalização vetorial pesada (Eq. 3) sobre a matriz
   regras × MOs; distâncias euclidianas ao ideal positivo/negativo;
   `V_i = D^min_i / (D^max_i + D^min_i)`.
3. **No `AC.RankA` original** os pesos das MOs são uniformes (o relatório
   adiciona a ponderação `[PM-SD]`).
4. **Poda CBA [M1]**: regras ordenadas por precedência; para cada regra,
   transações cobertas são marcadas e **deletadas** de D se a regra
   acerta qualquer uma; classe default (maioria do restante) por iteração.
5. **Predição [O]**: classe da **primeira regra no ranking que casa**,
   fallback para a classe default.

## 5. Reprodutibilidade

- `random_state = 42` (convenção Python/NumPy; paper/relatório não definem seed).
- `requirements.txt` fixado com versões exatas.
- Todos os splits via `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`.

## 6. Correções do code review (eixo Spec) — 2026-10-29

Decisões alteradas após o code review de Spec sobre a branch `int/acrankaw-spec`
(tickets #2–#8). Cada item registra o protocolo aplicável e a fonte.

| # | Tema | Decisão corrigida | Fonte |
| :-- | :--- | :--- | :--- |
| D1 | Poda CBA [M1] | A regra que acerta ≥1 transação coberta viva entra no classificador e **TODAS** as transações cobertas são deletadas (acertadas ou não); a classe default (maioria do restante) é recalculada a cada iteração e a final é a última calculada (ou a maioria global, se nada restou). A implementação anterior deletava só as transações acertadas — desvio do paper. | Paper, Algorithm 1 (linhas 3–15) e Seção II-C-2 "Pruning": *"all cases covered by r are deleted from D (line 13) and a default (majority) class (considering the remainder of D) is associated with the current classifier (line 14)"*; `docs/DECISIONS.md` §4 item 4 (já descrevia o protocolo correto) |
| D2 | Ramo 1 do SDD (§2) — XGBoost | O baseline caixa-preta (`dummy_no` + `xgboost`) passa a treinar no **frame ORIGINAL** (sem `customerID`): numéricas como estão e categóricas em one-hot (dentro do pipeline do XGBoost, para não vazar o fold de teste). A discretização MDLP fica **exclusiva do ramo 2** (CBA, AC.RankA, AC.RankAW). Antes, o XGBoost treinava sobre os bins MDLP — desvio do SDD e do README. | SDD §2 (diagrama "Ramo 1: Dados Contínuos/Dummies" × "Ramo 2: Dados Categóricos Discretizados") e README §4; seam público `src.run_benchmark.frames_do_fold` |
| D3 | CSV de resultados — "média ± desvio" | Os CSVs das fases 1 e 2 passam a ter, para cada modelo, colunas `<métrica>_desvio` com o desvio padrão amostral (ddof=1) das 10 dobras (preenchidas só no bloco "média"; as linhas de dobra ficam vazias). O schema é **idêntico** nos dois CSVs, para não quebrar a leitura. `delta_L_desvio` é o desvio dos ΔL por dobra (dispersão da razão) — grandeza distinta do `delta_L` do bloco "média" (razão das médias de L). | Tickets #4/#5 (exigem "média ± desvio"); SDD 5.1/5.2 |
| D4 | Export da Fase 3 — modelo-alvo | O export principal (JSON + SQL) passa a ser o **AC.RankAW Ponderado** (`[GF][PM-SD][TS]`), por **protocolo** (spec issue #1, histórias 7-8) e não por desempenho: antes era o de maior F1-Macro médio (que era o AC.RankA). As métricas dos três modelos associativos continuam no JSON (`metadados.metricas.modelos_avaliados`) e as regras do **AC.RankA** entram na seção secundária informativa `modelos_alternativos` (o SQL exporta só o AC.RankAW). | Spec issue #1 (histórias 7-8); `src/rule_export.py` (`MODELO_EXPORTADO`) |

