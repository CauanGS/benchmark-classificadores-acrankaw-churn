# Software Design Document (SDD)
## Projeto: Benchmark de Classificadores Interpretáveis e Aplicação do $AC.RankAW\text{ Ponderado}$ na Predição de Churn

**Autor:** Cauan Gabriel de Souza  
**Orientação Originária:** Profa. Dra. Veronica Oliveira de Carvalho (Unesp/IGCE/DEMAC)  
**Domínio de Aplicação:** Predição de Churn de Clientes (Kaggle - Telco Customer Churn)  
**Versão:** 1.0  

---

### 1. Visão Geral e Objetivos do Projeto

#### 1.1 Objetivo Geral
Construir um pipeline experimental de Machine Learning para demonstrar o valor da **Classificação Associativa Interpretável** e validar o impacto do método **$AC.RankAW\text{ Ponderado}$** ($[GF][PM\text{-}SD][TS]$) em relação a modelos caixa-preta ($XGBoost + SHAP$) e classificadores associativos baselines ($CBA$ e $AC.RankA$ padrão).

#### 1.2 Objetivos Específicos
1. **Fase 1 (Benchmark Inicial):** Comparar o modelo caixa-preta $XGBoost + SHAP$ com o algoritmo $CBA$ e o $AC.RankA$ padrão, evidenciando que os classificadores associativos entregam alta interpretabilidade nativa com desempenho preditivo competitivo.
2. **Fase 2 (Aplicação do $AC.RankAW\text{ Ponderado}$):** Aplicar o método proposto ($AC.RankAW\text{ Ponderado}$) para demonstrar a redução significativa na quantidade de regras ($L$) geradas, mantendo o nível de $F1\text{-Macro}$.
3. **Fase 3 (Empacotamento & Portfólio):** Exportar as regras de decisão otimizadas em formato interpretável (JSON/SQL) pronto para integração em sistemas de regras de negócio.

---

### 2. Arquitetura da Solução e Pipeline Experimental

O pipeline é dividido em quatro estágios sequenciais, desde a ingestão dos dados até a geração do relatório comparativo final:

```
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

---

### 3. Especificação dos Modelos Comparados

| Sigla / Nome | Categoria | Descrição do Método | Função no Experimento |
| :--- | :--- | :--- | :--- |
| **$XGBoost + SHAP$** | Black-Box Expliquei | Gradient Boosting com explicações via $SHAP$ values. | Baseline de mercado de alto desempenho preditivo. |
| **$CBA$** | Classificador Associativo | *Classification Based on Association* (Liu et al., 1998) ordenado por suporte e confiança. | Baseline clássico da literatura de regras. |
| **$AC.RankA$** | Classificador Associativo | Método de agregação multicritério original (Dall’Agnol e Carvalho, 2024) com pesos uniformes. | Baseline de agregação sem ponderação. |
| **$AC.RankAW\text{ Ponderado}$** | Classificador Associativo Proposto | Método proposto ($[GF][PM\text{-}SD][TS]$) com atribuição de pesos por Desvio Padrão sobre o grupo $GF$ e ordenação TOPSIS. | **Proposta do Autor (Foco do Estudo).** |

---

### 4. Pipeline de Dados e Pré-Processamento

#### 4.1 Base de Dados
* **Dataset:** *Telco Customer Churn* (Kaggle).
* **Atributo Alvo (Classe $C$):** `Churn` (`Yes` / `No`).

#### 4.2 Discretização (Requisito para Classificadores Associativos)
Classificadores associativos exigem atributos categóricos. As variáveis numéricas contínuas serão discretizadas em faixas de intervalos significativos:
* `tenure` (Tempo de contrato em meses) $\rightarrow$ Binning ordinal: `[0-12]`, `[13-24]`, `[25-48]`, `[49-72]`.
* `MonthlyCharges` (Valor mensal) $\rightarrow$ Discretização por quantis (Baixo, Médio, Alto).
* `TotalCharges` (Valor total) $\rightarrow$ Discretização por quantis.

#### 4.3 Protocolo de Validação
* **Estratégia:** Validação Cruzada Estratificada em 10 dobras ($10\text{-fold cross-validation}$).
* **Garantia:** Preservação da proporção das classes `Churn` em cada dobra.

---

### 5. Métricas de Avaliação e Critérios de Aceitação

#### 5.1 Métricas de Desempenho Preditivo
1. **$F1\text{-Macro}$:** Média aritmética do $F1\text{-Score}$ entre as classes `Churn = Yes` e `Churn = No`.
   $$F1\text{-Macro} = \frac{F1_{\text{Yes}} + F1_{\text{No}}}{2}$$
2. **Precisão e Revocação (Recall)** por classe.

#### 5.2 Métricas de Interpretabilidade e Complexidade
1. **Tamanho do Modelo ($L$):** Número total de Regras de Associação Classificativas (RACs) que compõem o classificador final pós-poda.
2. **Taxa de Redução de Regras ($\Delta L$):**
   $$\Delta L = \frac{L_{CBA} - L_{AC.RankAW}}{L_{CBA}} \times 100\%$$

#### 5.3 Critérios de Sucesso do Experimento
* **Fase 1:** Confirmar que o $AC.RankA$ atinge nível de $F1\text{-Macro}$ estatisticamente equivalente ou próximo ao $XGBoost$, com a vantagem de ser 100% auditável por regras $IF \dots THEN$.
* **Fase 2:** Provar que o $AC.RankAW\text{ Ponderado}$ atinge $\Delta L > 0$ (redução significativa do número de regras $L$) mantendo a equivalência em $F1\text{-Macro}$ em relação ao $AC.RankA$ padrão.

---

### 6. Estrutura do Repositório de Implementação

```text
acrankaw-churn-benchmark/
├── data/
│   ├── raw/                  # Dataset original do Kaggle
│   └── processed/            # Dataset discretizado para associativos
├── src/
│   ├── preprocessing.py      # Pipeline de discretização e binning
│   ├── xgboost_model.py      # Treino do XGBoost e extração de SHAP
│   ├── acrankaw.py           # Implementação da classe AC.RankAW Ponderado
│   └── evaluation.py         # Métricas (F1-Macro, Length L) e testes
├── notebooks/
│   ├── 01_eda_and_discretization.ipynb
│   ├── 02_phase1_xgboost_vs_cba_acranka.ipynb
│   └── 03_phase2_acrankaw_ponderado_experiments.ipynb
├── outputs/
│   ├── rules_exported.json   # Regras finais de negócio exportadas
│   └── benchmark_results.csv # Tabela final de comparação
├── README.md                 # Documentação e apresentação do portfólio
└── requirements.txt
```

---

### 7. Plano de Execução (Milestones)

1. **Milestone 1:** Download do dataset, análise exploratória (EDA) e pipeline de discretização de variáveis contínuas.
2. **Milestone 2:** Treinamento do $XGBoost$, extração da matriz $SHAP$ e avaliação do baseline caixa-preta.
3. **Milestone 3:** Treinamento do $CBA$ e do $AC.RankA$ padrão. Comparação da Fase 1.
4. **Milestone 4:** Integração da classe do $AC.RankAW\text{ Ponderado}$ ($[GF][PM\text{-}SD][TS]$) e execução dos testes de redução de regras.
5. **Milestone 5:** Consolidação da tabela comparativa final e geração do relatório de regras de negócio.