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

O pipeline do paper (Dall'Agnol & Carvalho, IEEE Access 2024) tem quatro etapas: **extração** (Apriori) → **ordenação** (precedência ou TOPSIS) → **poda M1** (descarta as regras que não acertam nenhuma transação; ao entrar no classificador, a regra remove **todas** as transações que ela cobre — acertadas ou não — do restante a classificar, e a classe default é recalculada a cada iteração) → **predição [O]** (a primeira regra do ranking que casa vence; senão, a classe default).

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

### Limitação conhecida — direção do CPS no TOPSIS

O *Complement Class Support* (CPS) vale `P(AB̄)/P(B̄)` (paper, Table 11, ID 5): mede a força de cobertura da classe **complementar**, para a qual **menor é melhor**. O TOPSIS do paper, porém, trata **todas** as MOs como critério de **benefício** — nas Eq. (4) e (5) o ideal positivo é `max{C_j}` e o ideal negativo `min{C_j}` para todo critério `j`, sem qualquer distinção entre custo e benefício ("*All OMs, in MCDA methods, receive the same weight with regard to the W vector*").

Como o objetivo é **fidelidade ao protocolo publicado**, o repositório reproduz o paper: `types = +1` em todas as colunas do TOPSIS (`src/associative.py::scores_topsis`). A consequência é conhecida e está registrada: o TOPSIS recompensa o CPS alto (cobertura forte da classe complementar), quando o comportamento "ideal" da MO seria o oposto. Tratar o CPS como custo exigiria `types = -1` naquela coluna — o `pymcdm` inverteria a coluna normalizada —, o que se afastaria do paper e foi deixado como *trabalho futuro* (ablation), não como correção silenciosa do protocolo.

## 5. Fase 1 — benchmark inicial (baselines)

`python -m src.run_benchmark --phase 1` executa a **Fase 1** (SDD §1.2): compara o baseline caixa-preta (`dummy_no` + `XGBoost + SHAP`, ramo 1 — dados contínuos/dummies) com os classificadores associativos de linha de base (`CBA` e `AC.RankA`, ramo 2 — dados discretizados). Grava `outputs/benchmark_results.csv` (linhas das 10 dobras + bloco "média" com **média ± desvio**) e os 6 PNGs de SHAP do fold 0 em `outputs/shap/`.

Números reais do run de referência (`outputs/benchmark_results.csv`), médias das 10 dobras (`média ± desvio padrão amostral`, ddof=1):

| Modelo | F1-Macro | Precisão (No) | Recall (No) | Precisão (Yes) | Recall (Yes) | L |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| dummy_no (baseline) | 0,4234 ± 0,0002 | 0,7342 ± 0,0005 | 1,0000 ± 0,0000 | 0,0000 ± 0,0000 | 0,0000 ± 0,0000 | 0,0 |
| XGBoost + SHAP | 0,7307 ± 0,0096 | 0,8414 ± 0,0037 | 0,9039 ± 0,0130 | 0,6672 ± 0,0289 | 0,5291 ± 0,0143 | 1,0 |
| CBA | 0,4262 ± 0,0047 | 0,7347 ± 0,0009 | 1,0000 ± 0,0000 | 0,3000 ± 0,4830 | 0,0027 ± 0,0045 | 179,2 ± 5,16 |
| AC.RankA | 0,7190 ± 0,0127 | 0,8777 ± 0,0087 | 0,7856 ± 0,0262 | 0,5422 ± 0,0259 | 0,6972 ± 0,0298 | 74,0 ± 5,85 |

Leitura da tabela:

- O `dummy_no` fixa o piso do problema — **0,4234** de F1-Macro — sem ele, qualquer número de modelo parece bom.
- O `CBA` é o **maior** modelo em número de regras (**L = 179,2**, ~2,4× o do AC.RankA) e praticamente não identifica churn (`recall_Yes = 0,0027`): a precedência CBA (confiança ↓) põe no topo as regras de "Não churn" mais confiantes, e a predição `[O]` acaba classificando quase tudo como "No". O desvio alto da precisão em `Yes` (0,4830) reflete que em algumas dobras nenhuma regra de churn sobreviveu à poda.
- Já os classificadores com ordenação **TOPSIS** (`AC.RankA`) alcançam F1-Macro na faixa do XGBoost (**0,7190** vs **0,7307**), com um modelo que é uma lista de regras auditável e **58,7% menor** que o CBA (74,0 vs 179,2 regras).
- `L = 1` no XGBoost significa apenas "não é um modelo de regras" (a complexidade dele está nos 100s de árvores, não em L).

## 6. Fase 2 — AC.RankAW Ponderado: menos regras, mesmo desempenho

`python -m src.run_benchmark --phase 2` executa a **Fase 2** (SDD §1.2): aplica o método proposto (`[GF][PM-SD][TS]`: pesos por desvio padrão sobre as 13 MOs do grupo `[GF]`) e mede a redução de regras (ΔL). A fase reavalia as referências do ΔL (`CBA` e `AC.RankA`) no mesmo run e **não** reexecuta a busca *nested* do XGBoost (baseline da Fase 1). Grava `outputs/benchmark_results_phase2.csv`.

Números reais do run de referência (`outputs/benchmark_results_phase2.csv`):

| Métrica | CBA | AC.RankA | AC.RankAW Ponderado |
| :--- | ---: | ---: | ---: |
| F1-Macro (média 10 dobras) | 0,4262 ± 0,0047 | 0,7190 ± 0,0127 | 0,7104 ± 0,0123 |
| Precisão (Yes) | 0,3000 ± 0,4830 | 0,5422 ± 0,0259 | 0,6186 ± 0,0277 |
| Recall (Yes) | 0,0027 ± 0,0045 | 0,6972 ± 0,0298 | 0,5158 ± 0,0255 |
| L (média 10 dobras) | 179,2 ± 5,16 | 74,0 ± 5,85 | 75,1 ± 4,86 |
| ΔL vs CBA (SDD 5.2) | — | — | **+58,09%** |
| ΔL vs AC.RankA | — | — | **−1,49%** |

Leitura da tabela:

- **ΔL > 0 (critério de sucesso do SDD 5.3):** o AC.RankAW entrega **58,09% menos regras** que o CBA (referência do SDD 5.2) — 75,1 contra 179,2 regras em média.
- **Equivalência de F1-Macro vs AC.RankA:** o modelo ponderado mantém o desempenho preditivo do AC.RankA padrão (0,7104 vs 0,7190): a diferença de **0,0086** é menor que o desvio-padrão de qualquer um dos dois modelos (0,0123 e 0,0127), ou seja, equivalente dentro da variabilidade das dobras. O que muda é o **perfil do erro**: o ponderado sobe a precisão em `Yes` (0,5422 → 0,6186) e o recall em `No` (0,7856 → 0,8844), em troca de menos recall em `Yes` (0,6972 → 0,5158) — entre os clientes que ele sinaliza como churn, erra menos; mas deixa mais churners escaparem como retidos.
- **Transparência do número:** a redução é medida **contra o CBA**, como define o SDD (5.2); a coluna `delta_L_vs_acranka` quantifica o efeito da ponderação `[PM-SD]` frente ao AC.RankA (mesmo motor, pesos uniformes) e é **negativa** (−1,49%): a ponderação muda o ranking e, portanto, o conjunto que sobrevive à poda M1, mas **não** reduz o modelo frente ao AC.RankA uniforme — neste run ela o torna 1,1 regra maior em média. O ganho do AC.RankAW está em compactar o CBA, não em suplantar o AC.RankA em tamanho.


## 7. Fase 3 — export das regras de negócio

`python -m src.run_benchmark --phase 3` avalia os três classificadores associativos no mesmo protocolo do benchmark (métricas registradas em `metadados.metricas.modelos_avaliados`), retreina o **AC.RankAW Ponderado** no dataset completo e grava:

| Artefato | Conteúdo |
| :--- | :--- |
| `outputs/rules_exported.json` | metadados (base, alvo, seed, modelo, `L`, métricas, protocolo) + lista de regras do AC.RankAW (antecedente como pares atributo-valor, consequente, suporte, confiança, score de ordenação) + bloco `class_default` + seção secundária `modelos_alternativos` (regras do AC.RankA, para comparação) |
| `outputs/rules_exported.sql` | script **executável em SQLite**: tabela `regras` (`id`, `antecedente_serializado`, `classe_predita`, `suporte`, `confianca`, `score`), `INSERT`s das regras, tabela de cliente de exemplo e as queries de decisão |

**O modelo exportado é o AC.RankAW Ponderado** (`[GF][PM-SD][TS]`) — a escolha é de **protocolo** (spec issue #1, histórias 7-8: o empacotamento de regras de negócio entrega o método proposto), e não de desempenho: as métricas de `cba`, `ac_ranka` e `ac_rankaw` ficam registradas no JSON para transparência. O AC.RankA aparece apenas como modelo alternativo, na seção informativa `modelos_alternativos` (o SQL exporta só o AC.RankAW).

No run de referência, o export gravou **L = 73** (72 regras + classe default `No`), com F1-Macro médio de **0,7104** e `id_exportacao` `4c88b5081a57d9e6` (determinístico, sem timestamp).

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
    suporte 0,1013 | confiança 0,9875 | score 0,9917
#2  Contract=Two year AND PaperlessBilling=No AND PhoneService=Yes AND SeniorCitizen=0 => No
    suporte 0,1075 | confiança 0,9857 | score 0,8689
#3  Contract=Two year AND PaperlessBilling=No AND SeniorCitizen=0                    => No
    suporte 0,1193 | confiança 0,9847 | score 0,8175
```

E a regra de churn (`Churn = Yes`) do ranking exportado:

```text
#70 Contract=Month-to-month                                                               => Yes
    suporte 0,2354 | confiança 0,4271 | score 0,1039
```

Leitura de negócio: **cliente de contrato mensal é o perfil de churn** — e **cliente de contrato de dois anos é cliente retido**. São segmentos acionáveis por um time de CRM, com a justificativa embutida na própria regra (`suporte` = porte do segmento, `confiança` = taxa de acerto da regra, `score` = prioridade no ranking).

Estrutura do modelo exportado (L = 73): **71 regras `No` e 1 regra `Yes`** (`Contract=Month-to-month`), com classe default `No`. Isso é consequência direta da poda M1 fiel ao paper: cada regra que acerta ao menos uma transação remove **todas** as transações que cobre do restante a classificar (Algorithm 1, linhas 3–15), de modo que as regras de alta confiança sobre contratos longos e valores baixos "consomem" o restante e sobra no topo do ranking uma regra ampla de churn. O comportamento resultante é o que a validação cruzada mede: `recall_Yes = 0,5158` e `precisão_Yes = 0,6186` (§6).

## 9. Como reproduzir

```bash
# 1) ambiente (Python 3.10)
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2) dataset (Kaggle, via kagglehub)
mkdir -p data/raw
python -c "import kagglehub; kagglehub.dataset_download('blastchar/telco-customer-churn')"
cp <dir-do-download>/WA_Fn-UseC_-Telco-Customer-Churn.csv data/raw/

# 3) benchmark (Fase 1: baselines + SHAP | Fase 2: AC.RankAW + ΔL)
python -m src.run_benchmark --phase 1     # -> outputs/benchmark_results.csv (+ outputs/shap/)
python -m src.run_benchmark --phase 2     # -> outputs/benchmark_results_phase2.csv (sem XGBoost)

# 4) export das regras de negócio (Fase 3)
python -m src.run_benchmark --phase 3     # -> outputs/rules_exported.{json,sql}

# 5) testes
python -m pytest tests/ -q
```

A Fase 1 leva ~10 min (o XGBoost faz busca *nested* de 30 iters em cada uma das 10 dobras); a Fase 2 é rápida (só classificadores associativos); a Fase 3 leva ~10 min (sem XGBoost).

## 10. Artefatos e reprodutibilidade

- Toda a cadeia usa **seed 42** (`random_state` do NumPy/scikit-learn, split estratificado, busca de hiperparâmetros) — o export não carrega timestamp e produz JSON/SQL **byte-idênticos** entre execuções (garantido por teste).
- `outputs/` é **versionado**: os CSVs de benchmark das fases 1 e 2, os 6 PNGs de SHAP (`outputs/shap/`) e os exports JSON/SQL do **run de referência** ficam no repositório, para que o resultado real do portfólio possa ser inspecionado sem reexecutar o benchmark. Só o dataset bruto (`data/`) fica ignorado — o run é reproduzível pelo comando acima, a partir do dataset público.
- Decisões de projeto e protocolo: [`docs/DECISIONS.md`](docs/DECISIONS.md) · desenho do experimento: [`software_design_document_sdd_churn_ac_rankaw_ponderado.md`](software_design_document_sdd_churn_ac_rankaw_ponderado.md).
