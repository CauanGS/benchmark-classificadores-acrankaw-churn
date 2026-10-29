-- ====================================================================
-- Export das regras de negócio — AC.RankAW Churn Benchmark (Fase 3)
-- Gerado por: python -m src.run_benchmark --phase 3
-- Base: Telco Customer Churn (Kaggle: blastchar/telco-customer-churn)
-- Alvo: Churn | Seed: 42 | Modelo: ac_rankaw | L: 73 (72 regras + classe default)
-- F1-Macro médio (10 dobras): 0.710354
-- id_exportacao: 4c88b5081a57d9e6
--
-- As regras casam com valores DISCRETIZADOS (MDLP Fayyad-Irani):
-- o cliente precisa passar pelo mesmo pré-processamento antes da consulta.
-- ====================================================================

DROP TABLE IF EXISTS regras;
CREATE TABLE regras (
    id INTEGER PRIMARY KEY,
    antecedente_serializado TEXT NOT NULL,
    classe_predita TEXT NOT NULL,
    suporte REAL NOT NULL,
    confianca REAL NOT NULL,
    score REAL NOT NULL
);

BEGIN TRANSACTION;
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (1, 'Contract=Two year AND TotalCharges=TotalCharges_(347.9, 3273.68]', 'No', 0.101251, 0.987517, 0.991672);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (2, 'Contract=Two year AND PaperlessBilling=No AND PhoneService=Yes AND SeniorCitizen=0', 'No', 0.107509, 0.985658, 0.868929);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (3, 'Contract=Two year AND PaperlessBilling=No AND SeniorCitizen=0', 'No', 0.119312, 0.984742, 0.817475);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (4, 'Contract=Two year AND PaperlessBilling=No AND PhoneService=Yes', 'No', 0.113908, 0.984029, 0.779587);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (5, 'Contract=Two year AND PaperlessBilling=No', 'No', 0.126138, 0.983370, 0.749681);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (6, 'DeviceProtection=No internet service AND InternetService=No AND MonthlyCharges=MonthlyCharges_(-∞, 29.375] AND TotalCharges=TotalCharges_(347.9, 3273.68]', 'No', 0.128413, 0.982590, 0.715665);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (7, 'MonthlyCharges=MonthlyCharges_(-∞, 29.375] AND TotalCharges=TotalCharges_(347.9, 3273.68]', 'No', 0.131684, 0.979894, 0.617975);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (8, 'Contract=Two year AND Dependents=Yes AND SeniorCitizen=0', 'No', 0.106371, 0.979058, 0.589927);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (9, 'Contract=Two year AND Dependents=Yes', 'No', 0.108362, 0.976923, 0.534103);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (10, 'Contract=Two year AND Partner=Yes AND PhoneService=Yes AND SeniorCitizen=0', 'No', 0.136945, 0.976673, 0.531014);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (11, 'Contract=Two year AND Partner=Yes AND SeniorCitizen=0', 'No', 0.150028, 0.975948, 0.515857);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (12, 'Contract=Two year AND SeniorCitizen=0 AND gender=Female', 'No', 0.107082, 0.975389, 0.499728);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (13, 'Contract=Two year AND Partner=Yes AND PhoneService=Yes', 'No', 0.149460, 0.974050, 0.476941);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (14, 'Contract=Two year AND Partner=Yes', 'No', 0.163965, 0.973818, 0.473984);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (15, 'Contract=Two year AND gender=Female', 'No', 0.116325, 0.973810, 0.469450);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (16, 'Contract=Two year AND SeniorCitizen=0', 'No', 0.213026, 0.972727, 0.459606);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (17, 'Contract=Two year', 'No', 0.232793, 0.971513, 0.441687);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (18, 'DeviceProtection=No internet service AND InternetService=No AND MonthlyCharges=MonthlyCharges_(-∞, 29.375] AND Partner=Yes', 'No', 0.100114, 0.960437, 0.305063);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (19, 'MonthlyCharges=MonthlyCharges_(-∞, 29.375] AND Partner=Yes', 'No', 0.102531, 0.949934, 0.238465);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (20, 'DeviceProtection=No internet service AND InternetService=No AND PaperlessBilling=No AND SeniorCitizen=0', 'No', 0.138367, 0.934678, 0.181146);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (21, 'DeviceProtection=No internet service AND InternetService=No AND MonthlyCharges=MonthlyCharges_(-∞, 29.375] AND PaperlessBilling=No', 'No', 0.142776, 0.933953, 0.179176);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (22, 'MonthlyCharges=MonthlyCharges_(-∞, 29.375] AND PaperlessBilling=No AND SeniorCitizen=0', 'No', 0.142634, 0.927845, 0.163107);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (23, 'OnlineSecurity=Yes AND SeniorCitizen=0 AND TotalCharges=TotalCharges_(3273.68, +∞]', 'No', 0.121729, 0.927411, 0.161520);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (24, 'DeviceProtection=No internet service AND InternetService=No AND MonthlyCharges=MonthlyCharges_(-∞, 29.375] AND SeniorCitizen=0', 'No', 0.193402, 0.926431, 0.161318);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (25, 'Partner=Yes AND TechSupport=Yes AND TotalCharges=TotalCharges_(3273.68, +∞]', 'No', 0.105375, 0.927409, 0.161117);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (26, 'OnlineBackup=Yes AND OnlineSecurity=Yes AND TotalCharges=TotalCharges_(3273.68, +∞]', 'No', 0.103100, 0.925926, 0.157643);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (27, 'DeviceProtection=Yes AND OnlineSecurity=Yes AND TotalCharges=TotalCharges_(3273.68, +∞]', 'No', 0.104807, 0.925879, 0.157577);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (28, 'OnlineSecurity=Yes AND Partner=Yes AND TotalCharges=TotalCharges_(3273.68, +∞]', 'No', 0.104807, 0.924718, 0.154997);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (29, 'OnlineBackup=Yes AND TechSupport=Yes AND TotalCharges=TotalCharges_(3273.68, +∞]', 'No', 0.106229, 0.923362, 0.152119);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (30, 'SeniorCitizen=0 AND TechSupport=Yes AND TotalCharges=TotalCharges_(3273.68, +∞]', 'No', 0.125142, 0.919540, 0.144858);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (31, 'SeniorCitizen=0 AND tenure=tenure_(59.5, 70.5]', 'No', 0.104096, 0.918444, 0.142350);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (32, 'InternetService=DSL AND OnlineBackup=Yes AND PhoneService=Yes', 'No', 0.103953, 0.918342, 0.142157);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (33, 'OnlineSecurity=Yes AND TotalCharges=TotalCharges_(3273.68, +∞]', 'No', 0.144056, 0.915913, 0.138629);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (34, 'DeviceProtection=Yes AND TechSupport=Yes AND TotalCharges=TotalCharges_(3273.68, +∞]', 'No', 0.110495, 0.916274, 0.138555);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (35, 'MonthlyCharges=MonthlyCharges_(-∞, 29.375] AND SeniorCitizen=0', 'No', 0.201081, 0.914027, 0.136804);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (36, 'InternetService=DSL AND OnlineSecurity=Yes AND PhoneService=Yes AND SeniorCitizen=0', 'No', 0.107082, 0.914945, 0.136191);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (37, 'InternetService=DSL AND Partner=Yes AND PhoneService=Yes AND SeniorCitizen=0', 'No', 0.101820, 0.914432, 0.135224);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (38, 'InternetService=DSL AND OnlineSecurity=Yes AND SeniorCitizen=0', 'No', 0.140358, 0.913043, 0.133681);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (39, 'OnlineSecurity=Yes AND SeniorCitizen=0 AND TechSupport=Yes', 'No', 0.126280, 0.912641, 0.132740);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (40, 'MultipleLines=No AND PaperlessBilling=No AND SeniorCitizen=0 AND TotalCharges=TotalCharges_(347.9, 3273.68]', 'No', 0.103953, 0.910336, 0.128705);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (41, 'OnlineSecurity=Yes AND TechSupport=Yes', 'No', 0.141780, 0.909672, 0.128400);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (42, 'InternetService=DSL AND Partner=Yes AND PhoneService=Yes', 'No', 0.110637, 0.909942, 0.128222);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (43, 'Dependents=Yes AND PaperlessBilling=No AND Partner=Yes AND SeniorCitizen=0', 'No', 0.107935, 0.908982, 0.126738);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (44, 'tenure=tenure_(59.5, 70.5]', 'No', 0.122867, 0.908517, 0.126317);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (45, 'InternetService=DSL AND OnlineSecurity=Yes AND PhoneService=Yes', 'No', 0.115472, 0.907263, 0.124369);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (46, 'InternetService=DSL AND SeniorCitizen=0 AND TechSupport=Yes', 'No', 0.139790, 0.905991, 0.123013);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (47, 'InternetService=DSL AND OnlineSecurity=Yes', 'No', 0.151308, 0.904762, 0.121539);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (48, 'DeviceProtection=Yes AND OnlineSecurity=Yes AND SeniorCitizen=0', 'No', 0.123862, 0.903527, 0.119392);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (49, 'PaperlessBilling=No AND Partner=Yes AND PhoneService=Yes AND SeniorCitizen=0', 'No', 0.147184, 0.903141, 0.119295);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (50, 'Dependents=Yes AND PaperlessBilling=No AND Partner=Yes', 'No', 0.109215, 0.903529, 0.119156);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (51, 'InternetService=DSL AND TechSupport=Yes', 'No', 0.150739, 0.902896, 0.119040);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (52, 'MultipleLines=No AND PaperlessBilling=No AND PhoneService=Yes AND TotalCharges=TotalCharges_(347.9, 3273.68]', 'No', 0.109926, 0.901984, 0.117171);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (53, 'InternetService=DSL AND OnlineBackup=Yes AND SeniorCitizen=0', 'No', 0.122156, 0.901364, 0.116581);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (54, 'DeviceProtection=Yes AND InternetService=DSL AND SeniorCitizen=0', 'No', 0.120592, 0.900212, 0.115125);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (55, 'Dependents=Yes AND PaperlessBilling=No AND PhoneService=Yes AND SeniorCitizen=0', 'No', 0.115472, 0.900222, 0.115057);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (56, 'OnlineBackup=Yes AND OnlineSecurity=Yes AND SeniorCitizen=0', 'No', 0.122298, 0.899582, 0.114383);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (57, 'PaperlessBilling=No AND Partner=Yes AND SeniorCitizen=0', 'No', 0.163111, 0.898903, 0.114256);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (58, 'PaperlessBilling=No AND SeniorCitizen=0 AND TotalCharges=TotalCharges_(347.9, 3273.68]', 'No', 0.171359, 0.897914, 0.113228);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (59, 'DeviceProtection=Yes AND OnlineSecurity=Yes', 'No', 0.141496, 0.898014, 0.112820);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (60, 'OnlineSecurity=Yes AND Partner=Yes AND SeniorCitizen=0', 'No', 0.130830, 0.897561, 0.112117);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (61, 'OnlineBackup=Yes AND SeniorCitizen=0 AND TechSupport=Yes', 'No', 0.125995, 0.896761, 0.111119);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (62, 'OnlineBackup=Yes AND OnlineSecurity=Yes', 'No', 0.142776, 0.894034, 0.108334);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (63, 'Contract=One year AND SeniorCitizen=0', 'No', 0.162827, 0.893136, 0.107688);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (64, 'Partner=Yes AND PaymentMethod=Credit card (automatic)', 'No', 0.108362, 0.893318, 0.107064);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (65, 'InternetService=DSL AND OnlineBackup=Yes', 'No', 0.137372, 0.892791, 0.106916);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (66, 'Partner=Yes AND SeniorCitizen=0 AND TechSupport=Yes', 'No', 0.128982, 0.892717, 0.106714);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (67, 'InternetService=DSL AND Partner=Yes AND SeniorCitizen=0', 'No', 0.132110, 0.892411, 0.106437);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (68, 'PaperlessBilling=No AND Partner=Yes AND PhoneService=Yes', 'No', 0.159841, 0.891356, 0.105760);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (69, 'OnlineBackup=Yes AND TechSupport=Yes', 'No', 0.145478, 0.890339, 0.104495);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (70, 'Contract=Month-to-month', 'Yes', 0.235353, 0.427097, 0.103911);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (71, 'DeviceProtection=Yes AND InternetService=DSL', 'No', 0.134243, 0.889727, 0.103715);
INSERT INTO regras (id, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (72, 'Contract=One year', 'No', 0.185722, 0.887228, 0.102056);
COMMIT;

-- --------------------------------------------------------------------
-- Cliente de exemplo: casa com a 1ª regra do ranking exportado.
-- --------------------------------------------------------------------
DROP TABLE IF EXISTS cliente_exemplo;
CREATE TABLE cliente_exemplo (
    cliente_id TEXT PRIMARY KEY,
    "gender" TEXT,
    "SeniorCitizen" TEXT,
    "Partner" TEXT,
    "Dependents" TEXT,
    "PhoneService" TEXT,
    "MultipleLines" TEXT,
    "InternetService" TEXT,
    "OnlineSecurity" TEXT,
    "OnlineBackup" TEXT,
    "DeviceProtection" TEXT,
    "TechSupport" TEXT,
    "StreamingTV" TEXT,
    "StreamingMovies" TEXT,
    "Contract" TEXT,
    "PaperlessBilling" TEXT,
    "PaymentMethod" TEXT,
    "tenure" TEXT,
    "MonthlyCharges" TEXT,
    "TotalCharges" TEXT
);

INSERT INTO cliente_exemplo (cliente_id, "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "tenure", "MonthlyCharges", "TotalCharges") VALUES ('EXEMPLO-1', 'Male', '0', 'No', 'No', 'Yes', 'No', 'Fiber optic', 'No', 'No', 'No', 'No', 'No', 'No', 'Two year', 'Yes', 'Electronic check', 'tenure_(17.5, 43.5]', 'MonthlyCharges_(68.825, 106.775]', 'TotalCharges_(347.9, 3273.68]');

-- --------------------------------------------------------------------
-- Query de decisão (predição [O]): a PRIMEIRA regra do ranking que
-- casa vence; se nenhuma casa, vale a classe default.
-- --------------------------------------------------------------------
DROP VIEW IF EXISTS decisao_cliente;
CREATE VIEW decisao_cliente AS
SELECT
    c.cliente_id,
    CASE
        WHEN c."Contract" = 'Two year' AND c."TotalCharges" = 'TotalCharges_(347.9, 3273.68]' THEN 'No'
        WHEN c."Contract" = 'Two year' AND c."PaperlessBilling" = 'No' AND c."PhoneService" = 'Yes' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."Contract" = 'Two year' AND c."PaperlessBilling" = 'No' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."Contract" = 'Two year' AND c."PaperlessBilling" = 'No' AND c."PhoneService" = 'Yes' THEN 'No'
        WHEN c."Contract" = 'Two year' AND c."PaperlessBilling" = 'No' THEN 'No'
        WHEN c."DeviceProtection" = 'No internet service' AND c."InternetService" = 'No' AND c."MonthlyCharges" = 'MonthlyCharges_(-∞, 29.375]' AND c."TotalCharges" = 'TotalCharges_(347.9, 3273.68]' THEN 'No'
        WHEN c."MonthlyCharges" = 'MonthlyCharges_(-∞, 29.375]' AND c."TotalCharges" = 'TotalCharges_(347.9, 3273.68]' THEN 'No'
        WHEN c."Contract" = 'Two year' AND c."Dependents" = 'Yes' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."Contract" = 'Two year' AND c."Dependents" = 'Yes' THEN 'No'
        WHEN c."Contract" = 'Two year' AND c."Partner" = 'Yes' AND c."PhoneService" = 'Yes' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."Contract" = 'Two year' AND c."Partner" = 'Yes' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."Contract" = 'Two year' AND c."SeniorCitizen" = '0' AND c."gender" = 'Female' THEN 'No'
        WHEN c."Contract" = 'Two year' AND c."Partner" = 'Yes' AND c."PhoneService" = 'Yes' THEN 'No'
        WHEN c."Contract" = 'Two year' AND c."Partner" = 'Yes' THEN 'No'
        WHEN c."Contract" = 'Two year' AND c."gender" = 'Female' THEN 'No'
        WHEN c."Contract" = 'Two year' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."Contract" = 'Two year' THEN 'No'
        WHEN c."DeviceProtection" = 'No internet service' AND c."InternetService" = 'No' AND c."MonthlyCharges" = 'MonthlyCharges_(-∞, 29.375]' AND c."Partner" = 'Yes' THEN 'No'
        WHEN c."MonthlyCharges" = 'MonthlyCharges_(-∞, 29.375]' AND c."Partner" = 'Yes' THEN 'No'
        WHEN c."DeviceProtection" = 'No internet service' AND c."InternetService" = 'No' AND c."PaperlessBilling" = 'No' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."DeviceProtection" = 'No internet service' AND c."InternetService" = 'No' AND c."MonthlyCharges" = 'MonthlyCharges_(-∞, 29.375]' AND c."PaperlessBilling" = 'No' THEN 'No'
        WHEN c."MonthlyCharges" = 'MonthlyCharges_(-∞, 29.375]' AND c."PaperlessBilling" = 'No' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."OnlineSecurity" = 'Yes' AND c."SeniorCitizen" = '0' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]' THEN 'No'
        WHEN c."DeviceProtection" = 'No internet service' AND c."InternetService" = 'No' AND c."MonthlyCharges" = 'MonthlyCharges_(-∞, 29.375]' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."Partner" = 'Yes' AND c."TechSupport" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]' THEN 'No'
        WHEN c."OnlineBackup" = 'Yes' AND c."OnlineSecurity" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]' THEN 'No'
        WHEN c."DeviceProtection" = 'Yes' AND c."OnlineSecurity" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]' THEN 'No'
        WHEN c."OnlineSecurity" = 'Yes' AND c."Partner" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]' THEN 'No'
        WHEN c."OnlineBackup" = 'Yes' AND c."TechSupport" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]' THEN 'No'
        WHEN c."SeniorCitizen" = '0' AND c."TechSupport" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]' THEN 'No'
        WHEN c."SeniorCitizen" = '0' AND c."tenure" = 'tenure_(59.5, 70.5]' THEN 'No'
        WHEN c."InternetService" = 'DSL' AND c."OnlineBackup" = 'Yes' AND c."PhoneService" = 'Yes' THEN 'No'
        WHEN c."OnlineSecurity" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]' THEN 'No'
        WHEN c."DeviceProtection" = 'Yes' AND c."TechSupport" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]' THEN 'No'
        WHEN c."MonthlyCharges" = 'MonthlyCharges_(-∞, 29.375]' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."InternetService" = 'DSL' AND c."OnlineSecurity" = 'Yes' AND c."PhoneService" = 'Yes' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."InternetService" = 'DSL' AND c."Partner" = 'Yes' AND c."PhoneService" = 'Yes' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."InternetService" = 'DSL' AND c."OnlineSecurity" = 'Yes' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."OnlineSecurity" = 'Yes' AND c."SeniorCitizen" = '0' AND c."TechSupport" = 'Yes' THEN 'No'
        WHEN c."MultipleLines" = 'No' AND c."PaperlessBilling" = 'No' AND c."SeniorCitizen" = '0' AND c."TotalCharges" = 'TotalCharges_(347.9, 3273.68]' THEN 'No'
        WHEN c."OnlineSecurity" = 'Yes' AND c."TechSupport" = 'Yes' THEN 'No'
        WHEN c."InternetService" = 'DSL' AND c."Partner" = 'Yes' AND c."PhoneService" = 'Yes' THEN 'No'
        WHEN c."Dependents" = 'Yes' AND c."PaperlessBilling" = 'No' AND c."Partner" = 'Yes' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."tenure" = 'tenure_(59.5, 70.5]' THEN 'No'
        WHEN c."InternetService" = 'DSL' AND c."OnlineSecurity" = 'Yes' AND c."PhoneService" = 'Yes' THEN 'No'
        WHEN c."InternetService" = 'DSL' AND c."SeniorCitizen" = '0' AND c."TechSupport" = 'Yes' THEN 'No'
        WHEN c."InternetService" = 'DSL' AND c."OnlineSecurity" = 'Yes' THEN 'No'
        WHEN c."DeviceProtection" = 'Yes' AND c."OnlineSecurity" = 'Yes' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."PaperlessBilling" = 'No' AND c."Partner" = 'Yes' AND c."PhoneService" = 'Yes' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."Dependents" = 'Yes' AND c."PaperlessBilling" = 'No' AND c."Partner" = 'Yes' THEN 'No'
        WHEN c."InternetService" = 'DSL' AND c."TechSupport" = 'Yes' THEN 'No'
        WHEN c."MultipleLines" = 'No' AND c."PaperlessBilling" = 'No' AND c."PhoneService" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(347.9, 3273.68]' THEN 'No'
        WHEN c."InternetService" = 'DSL' AND c."OnlineBackup" = 'Yes' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."DeviceProtection" = 'Yes' AND c."InternetService" = 'DSL' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."Dependents" = 'Yes' AND c."PaperlessBilling" = 'No' AND c."PhoneService" = 'Yes' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."OnlineBackup" = 'Yes' AND c."OnlineSecurity" = 'Yes' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."PaperlessBilling" = 'No' AND c."Partner" = 'Yes' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."PaperlessBilling" = 'No' AND c."SeniorCitizen" = '0' AND c."TotalCharges" = 'TotalCharges_(347.9, 3273.68]' THEN 'No'
        WHEN c."DeviceProtection" = 'Yes' AND c."OnlineSecurity" = 'Yes' THEN 'No'
        WHEN c."OnlineSecurity" = 'Yes' AND c."Partner" = 'Yes' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."OnlineBackup" = 'Yes' AND c."SeniorCitizen" = '0' AND c."TechSupport" = 'Yes' THEN 'No'
        WHEN c."OnlineBackup" = 'Yes' AND c."OnlineSecurity" = 'Yes' THEN 'No'
        WHEN c."Contract" = 'One year' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."Partner" = 'Yes' AND c."PaymentMethod" = 'Credit card (automatic)' THEN 'No'
        WHEN c."InternetService" = 'DSL' AND c."OnlineBackup" = 'Yes' THEN 'No'
        WHEN c."Partner" = 'Yes' AND c."SeniorCitizen" = '0' AND c."TechSupport" = 'Yes' THEN 'No'
        WHEN c."InternetService" = 'DSL' AND c."Partner" = 'Yes' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."PaperlessBilling" = 'No' AND c."Partner" = 'Yes' AND c."PhoneService" = 'Yes' THEN 'No'
        WHEN c."OnlineBackup" = 'Yes' AND c."TechSupport" = 'Yes' THEN 'No'
        WHEN c."Contract" = 'Month-to-month' THEN 'Yes'
        WHEN c."DeviceProtection" = 'Yes' AND c."InternetService" = 'DSL' THEN 'No'
        WHEN c."Contract" = 'One year' THEN 'No'
        ELSE 'No'
    END AS classe_predita
FROM cliente_exemplo c;

-- Mesma decisão em forma relacional: filtra as regras que casam e
-- ordena por score (decrescente) — o score é o critério de ordenação
-- do modelo exportado; a primeira (maior score) vence.
DROP VIEW IF EXISTS decisao_cliente_por_score;
CREATE VIEW decisao_cliente_por_score AS
SELECT
    c.cliente_id,
    COALESCE(
        (
            SELECT r.classe_predita
            FROM regras r
            WHERE (r.id = 1 AND c."Contract" = 'Two year' AND c."TotalCharges" = 'TotalCharges_(347.9, 3273.68]')
               OR (r.id = 2 AND c."Contract" = 'Two year' AND c."PaperlessBilling" = 'No' AND c."PhoneService" = 'Yes' AND c."SeniorCitizen" = '0')
               OR (r.id = 3 AND c."Contract" = 'Two year' AND c."PaperlessBilling" = 'No' AND c."SeniorCitizen" = '0')
               OR (r.id = 4 AND c."Contract" = 'Two year' AND c."PaperlessBilling" = 'No' AND c."PhoneService" = 'Yes')
               OR (r.id = 5 AND c."Contract" = 'Two year' AND c."PaperlessBilling" = 'No')
               OR (r.id = 6 AND c."DeviceProtection" = 'No internet service' AND c."InternetService" = 'No' AND c."MonthlyCharges" = 'MonthlyCharges_(-∞, 29.375]' AND c."TotalCharges" = 'TotalCharges_(347.9, 3273.68]')
               OR (r.id = 7 AND c."MonthlyCharges" = 'MonthlyCharges_(-∞, 29.375]' AND c."TotalCharges" = 'TotalCharges_(347.9, 3273.68]')
               OR (r.id = 8 AND c."Contract" = 'Two year' AND c."Dependents" = 'Yes' AND c."SeniorCitizen" = '0')
               OR (r.id = 9 AND c."Contract" = 'Two year' AND c."Dependents" = 'Yes')
               OR (r.id = 10 AND c."Contract" = 'Two year' AND c."Partner" = 'Yes' AND c."PhoneService" = 'Yes' AND c."SeniorCitizen" = '0')
               OR (r.id = 11 AND c."Contract" = 'Two year' AND c."Partner" = 'Yes' AND c."SeniorCitizen" = '0')
               OR (r.id = 12 AND c."Contract" = 'Two year' AND c."SeniorCitizen" = '0' AND c."gender" = 'Female')
               OR (r.id = 13 AND c."Contract" = 'Two year' AND c."Partner" = 'Yes' AND c."PhoneService" = 'Yes')
               OR (r.id = 14 AND c."Contract" = 'Two year' AND c."Partner" = 'Yes')
               OR (r.id = 15 AND c."Contract" = 'Two year' AND c."gender" = 'Female')
               OR (r.id = 16 AND c."Contract" = 'Two year' AND c."SeniorCitizen" = '0')
               OR (r.id = 17 AND c."Contract" = 'Two year')
               OR (r.id = 18 AND c."DeviceProtection" = 'No internet service' AND c."InternetService" = 'No' AND c."MonthlyCharges" = 'MonthlyCharges_(-∞, 29.375]' AND c."Partner" = 'Yes')
               OR (r.id = 19 AND c."MonthlyCharges" = 'MonthlyCharges_(-∞, 29.375]' AND c."Partner" = 'Yes')
               OR (r.id = 20 AND c."DeviceProtection" = 'No internet service' AND c."InternetService" = 'No' AND c."PaperlessBilling" = 'No' AND c."SeniorCitizen" = '0')
               OR (r.id = 21 AND c."DeviceProtection" = 'No internet service' AND c."InternetService" = 'No' AND c."MonthlyCharges" = 'MonthlyCharges_(-∞, 29.375]' AND c."PaperlessBilling" = 'No')
               OR (r.id = 22 AND c."MonthlyCharges" = 'MonthlyCharges_(-∞, 29.375]' AND c."PaperlessBilling" = 'No' AND c."SeniorCitizen" = '0')
               OR (r.id = 23 AND c."OnlineSecurity" = 'Yes' AND c."SeniorCitizen" = '0' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]')
               OR (r.id = 24 AND c."DeviceProtection" = 'No internet service' AND c."InternetService" = 'No' AND c."MonthlyCharges" = 'MonthlyCharges_(-∞, 29.375]' AND c."SeniorCitizen" = '0')
               OR (r.id = 25 AND c."Partner" = 'Yes' AND c."TechSupport" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]')
               OR (r.id = 26 AND c."OnlineBackup" = 'Yes' AND c."OnlineSecurity" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]')
               OR (r.id = 27 AND c."DeviceProtection" = 'Yes' AND c."OnlineSecurity" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]')
               OR (r.id = 28 AND c."OnlineSecurity" = 'Yes' AND c."Partner" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]')
               OR (r.id = 29 AND c."OnlineBackup" = 'Yes' AND c."TechSupport" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]')
               OR (r.id = 30 AND c."SeniorCitizen" = '0' AND c."TechSupport" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]')
               OR (r.id = 31 AND c."SeniorCitizen" = '0' AND c."tenure" = 'tenure_(59.5, 70.5]')
               OR (r.id = 32 AND c."InternetService" = 'DSL' AND c."OnlineBackup" = 'Yes' AND c."PhoneService" = 'Yes')
               OR (r.id = 33 AND c."OnlineSecurity" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]')
               OR (r.id = 34 AND c."DeviceProtection" = 'Yes' AND c."TechSupport" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]')
               OR (r.id = 35 AND c."MonthlyCharges" = 'MonthlyCharges_(-∞, 29.375]' AND c."SeniorCitizen" = '0')
               OR (r.id = 36 AND c."InternetService" = 'DSL' AND c."OnlineSecurity" = 'Yes' AND c."PhoneService" = 'Yes' AND c."SeniorCitizen" = '0')
               OR (r.id = 37 AND c."InternetService" = 'DSL' AND c."Partner" = 'Yes' AND c."PhoneService" = 'Yes' AND c."SeniorCitizen" = '0')
               OR (r.id = 38 AND c."InternetService" = 'DSL' AND c."OnlineSecurity" = 'Yes' AND c."SeniorCitizen" = '0')
               OR (r.id = 39 AND c."OnlineSecurity" = 'Yes' AND c."SeniorCitizen" = '0' AND c."TechSupport" = 'Yes')
               OR (r.id = 40 AND c."MultipleLines" = 'No' AND c."PaperlessBilling" = 'No' AND c."SeniorCitizen" = '0' AND c."TotalCharges" = 'TotalCharges_(347.9, 3273.68]')
               OR (r.id = 41 AND c."OnlineSecurity" = 'Yes' AND c."TechSupport" = 'Yes')
               OR (r.id = 42 AND c."InternetService" = 'DSL' AND c."Partner" = 'Yes' AND c."PhoneService" = 'Yes')
               OR (r.id = 43 AND c."Dependents" = 'Yes' AND c."PaperlessBilling" = 'No' AND c."Partner" = 'Yes' AND c."SeniorCitizen" = '0')
               OR (r.id = 44 AND c."tenure" = 'tenure_(59.5, 70.5]')
               OR (r.id = 45 AND c."InternetService" = 'DSL' AND c."OnlineSecurity" = 'Yes' AND c."PhoneService" = 'Yes')
               OR (r.id = 46 AND c."InternetService" = 'DSL' AND c."SeniorCitizen" = '0' AND c."TechSupport" = 'Yes')
               OR (r.id = 47 AND c."InternetService" = 'DSL' AND c."OnlineSecurity" = 'Yes')
               OR (r.id = 48 AND c."DeviceProtection" = 'Yes' AND c."OnlineSecurity" = 'Yes' AND c."SeniorCitizen" = '0')
               OR (r.id = 49 AND c."PaperlessBilling" = 'No' AND c."Partner" = 'Yes' AND c."PhoneService" = 'Yes' AND c."SeniorCitizen" = '0')
               OR (r.id = 50 AND c."Dependents" = 'Yes' AND c."PaperlessBilling" = 'No' AND c."Partner" = 'Yes')
               OR (r.id = 51 AND c."InternetService" = 'DSL' AND c."TechSupport" = 'Yes')
               OR (r.id = 52 AND c."MultipleLines" = 'No' AND c."PaperlessBilling" = 'No' AND c."PhoneService" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(347.9, 3273.68]')
               OR (r.id = 53 AND c."InternetService" = 'DSL' AND c."OnlineBackup" = 'Yes' AND c."SeniorCitizen" = '0')
               OR (r.id = 54 AND c."DeviceProtection" = 'Yes' AND c."InternetService" = 'DSL' AND c."SeniorCitizen" = '0')
               OR (r.id = 55 AND c."Dependents" = 'Yes' AND c."PaperlessBilling" = 'No' AND c."PhoneService" = 'Yes' AND c."SeniorCitizen" = '0')
               OR (r.id = 56 AND c."OnlineBackup" = 'Yes' AND c."OnlineSecurity" = 'Yes' AND c."SeniorCitizen" = '0')
               OR (r.id = 57 AND c."PaperlessBilling" = 'No' AND c."Partner" = 'Yes' AND c."SeniorCitizen" = '0')
               OR (r.id = 58 AND c."PaperlessBilling" = 'No' AND c."SeniorCitizen" = '0' AND c."TotalCharges" = 'TotalCharges_(347.9, 3273.68]')
               OR (r.id = 59 AND c."DeviceProtection" = 'Yes' AND c."OnlineSecurity" = 'Yes')
               OR (r.id = 60 AND c."OnlineSecurity" = 'Yes' AND c."Partner" = 'Yes' AND c."SeniorCitizen" = '0')
               OR (r.id = 61 AND c."OnlineBackup" = 'Yes' AND c."SeniorCitizen" = '0' AND c."TechSupport" = 'Yes')
               OR (r.id = 62 AND c."OnlineBackup" = 'Yes' AND c."OnlineSecurity" = 'Yes')
               OR (r.id = 63 AND c."Contract" = 'One year' AND c."SeniorCitizen" = '0')
               OR (r.id = 64 AND c."Partner" = 'Yes' AND c."PaymentMethod" = 'Credit card (automatic)')
               OR (r.id = 65 AND c."InternetService" = 'DSL' AND c."OnlineBackup" = 'Yes')
               OR (r.id = 66 AND c."Partner" = 'Yes' AND c."SeniorCitizen" = '0' AND c."TechSupport" = 'Yes')
               OR (r.id = 67 AND c."InternetService" = 'DSL' AND c."Partner" = 'Yes' AND c."SeniorCitizen" = '0')
               OR (r.id = 68 AND c."PaperlessBilling" = 'No' AND c."Partner" = 'Yes' AND c."PhoneService" = 'Yes')
               OR (r.id = 69 AND c."OnlineBackup" = 'Yes' AND c."TechSupport" = 'Yes')
               OR (r.id = 70 AND c."Contract" = 'Month-to-month')
               OR (r.id = 71 AND c."DeviceProtection" = 'Yes' AND c."InternetService" = 'DSL')
               OR (r.id = 72 AND c."Contract" = 'One year')
            ORDER BY r.score DESC, r.id ASC
            LIMIT 1
        ),
        'No'
    ) AS classe_predita
FROM cliente_exemplo c;

-- Saída da query de decisão de exemplo:
SELECT * FROM decisao_cliente;
