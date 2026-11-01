-- ====================================================================
-- Export das regras de negócio — AC.RankAW Churn Benchmark (Fase 3)
-- Gerado por: python -m src.run_benchmark --phase 3
-- Base: Telco Customer Churn (Kaggle: blastchar/telco-customer-churn)
-- Alvo: Churn | Seed: 42 | Modelo: ac_rankaw | L: 69 (68 regras + classe default)
-- F1-Macro médio (10 dobras): 0.720388
-- id_exportacao: f9db6fb1e569a68e
--
-- As regras casam com valores DISCRETIZADOS (MDLP Fayyad-Irani):
-- o cliente precisa passar pelo mesmo pré-processamento antes da consulta.
-- ====================================================================

DROP TABLE IF EXISTS regras;
CREATE TABLE regras (
    id INTEGER PRIMARY KEY,
    ordem INTEGER NOT NULL,
    antecedente_serializado TEXT NOT NULL,
    classe_predita TEXT NOT NULL,
    suporte REAL NOT NULL,
    confianca REAL NOT NULL,
    score REAL NOT NULL
);

BEGIN TRANSACTION;
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (1, 1, 'Contract=Two year AND TotalCharges=TotalCharges_(347.9, 3273.68]', 'No', 0.101251, 0.987517, 0.753638);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (2, 2, 'Contract=Two year AND PaperlessBilling=No AND PhoneService=Yes AND SeniorCitizen=0', 'No', 0.107509, 0.985658, 0.736145);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (3, 3, 'Contract=Two year AND PaperlessBilling=No AND SeniorCitizen=0', 'No', 0.119312, 0.984742, 0.728568);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (4, 4, 'Contract=Two year AND PaperlessBilling=No AND PhoneService=Yes', 'No', 0.113908, 0.984029, 0.716920);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (5, 5, 'Contract=Two year AND PaperlessBilling=No', 'No', 0.126138, 0.983370, 0.712486);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (6, 6, 'DeviceProtection=No internet service AND InternetService=No AND MonthlyCharges=MonthlyCharges_(-∞, 29.375] AND TotalCharges=TotalCharges_(347.9, 3273.68]', 'No', 0.128413, 0.982590, 0.702924);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (7, 7, 'MonthlyCharges=MonthlyCharges_(-∞, 29.375] AND TotalCharges=TotalCharges_(347.9, 3273.68]', 'No', 0.131684, 0.979894, 0.671332);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (8, 8, 'Contract=Two year AND Dependents=Yes AND SeniorCitizen=0', 'No', 0.106371, 0.979058, 0.654078);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (9, 9, 'Contract=Two year AND Partner=Yes AND PhoneService=Yes AND SeniorCitizen=0', 'No', 0.136945, 0.976673, 0.641132);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (10, 10, 'Contract=Two year AND Partner=Yes AND SeniorCitizen=0', 'No', 0.150028, 0.975948, 0.638731);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (11, 11, 'Contract=Two year AND Dependents=Yes', 'No', 0.108362, 0.976923, 0.634599);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (12, 12, 'Contract=Two year AND SeniorCitizen=0', 'No', 0.213026, 0.972727, 0.630747);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (13, 13, 'Contract=Two year', 'No', 0.232793, 0.971513, 0.627360);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (14, 14, 'DeviceProtection=No internet service AND InternetService=No AND MonthlyCharges=MonthlyCharges_(-∞, 29.375] AND Partner=Yes', 'No', 0.100114, 0.960437, 0.546073);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (15, 15, 'MonthlyCharges=MonthlyCharges_(-∞, 29.375] AND Partner=Yes', 'No', 0.102531, 0.949934, 0.519592);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (16, 16, 'DeviceProtection=No internet service AND InternetService=No AND PaperlessBilling=No AND SeniorCitizen=0', 'No', 0.138367, 0.934678, 0.499566);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (17, 17, 'DeviceProtection=No internet service AND InternetService=No AND MonthlyCharges=MonthlyCharges_(-∞, 29.375] AND PaperlessBilling=No', 'No', 0.142776, 0.933953, 0.499269);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (18, 18, 'DeviceProtection=No internet service', 'No', 0.200085, 0.925658, 0.497626);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (19, 19, 'MonthlyCharges=MonthlyCharges_(-∞, 29.375] AND PaperlessBilling=No AND SeniorCitizen=0', 'No', 0.142634, 0.927845, 0.490903);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (20, 20, 'OnlineSecurity=Yes AND SeniorCitizen=0 AND TotalCharges=TotalCharges_(3273.68, +∞]', 'No', 0.121729, 0.927411, 0.486965);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (21, 21, 'Partner=Yes AND TechSupport=Yes AND TotalCharges=TotalCharges_(3273.68, +∞]', 'No', 0.105375, 0.927409, 0.484368);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (22, 22, 'MonthlyCharges=MonthlyCharges_(-∞, 29.375] AND SeniorCitizen=0', 'No', 0.201081, 0.914027, 0.483788);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (23, 23, 'DeviceProtection=Yes AND OnlineSecurity=Yes AND TotalCharges=TotalCharges_(3273.68, +∞]', 'No', 0.104807, 0.925879, 0.482418);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (24, 24, 'OnlineBackup=Yes AND OnlineSecurity=Yes AND TotalCharges=TotalCharges_(3273.68, +∞]', 'No', 0.103100, 0.925926, 0.482209);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (25, 25, 'OnlineSecurity=Yes AND Partner=Yes AND TotalCharges=TotalCharges_(3273.68, +∞]', 'No', 0.104807, 0.924718, 0.481038);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (26, 26, 'OnlineBackup=Yes AND TechSupport=Yes AND TotalCharges=TotalCharges_(3273.68, +∞]', 'No', 0.106229, 0.923362, 0.479676);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (27, 27, 'Contract=Month-to-month AND InternetService=Fiber optic AND OnlineSecurity=No AND TechSupport=No', 'Yes', 0.131542, 0.606955, 0.478803);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (28, 28, 'SeniorCitizen=0 AND TechSupport=Yes AND TotalCharges=TotalCharges_(3273.68, +∞]', 'No', 0.125142, 0.919540, 0.478220);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (29, 29, 'OnlineSecurity=Yes AND TotalCharges=TotalCharges_(3273.68, +∞]', 'No', 0.144056, 0.915913, 0.477100);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (30, 30, 'Contract=Month-to-month AND MonthlyCharges=MonthlyCharges_(68.825, 106.775] AND OnlineSecurity=No AND TechSupport=No', 'Yes', 0.130973, 0.602749, 0.476267);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (31, 31, 'Contract=Month-to-month AND InternetService=Fiber optic AND OnlineSecurity=No AND PaymentMethod=Electronic check', 'Yes', 0.101820, 0.634752, 0.476263);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (32, 32, 'Contract=Month-to-month AND MonthlyCharges=MonthlyCharges_(68.825, 106.775] AND OnlineSecurity=No AND PaymentMethod=Electronic check', 'Yes', 0.101109, 0.633690, 0.475332);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (33, 33, 'Contract=Month-to-month AND OnlineSecurity=No AND PaymentMethod=Electronic check AND TechSupport=No', 'Yes', 0.112059, 0.619984, 0.474541);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (34, 34, 'Contract=Month-to-month AND InternetService=Fiber optic AND MonthlyCharges=MonthlyCharges_(68.825, 106.775] AND OnlineSecurity=No', 'Yes', 0.144482, 0.584244, 0.474105);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (35, 35, 'SeniorCitizen=0 AND tenure=tenure_(59.5, 70.5]', 'No', 0.104096, 0.918444, 0.473895);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (36, 36, 'InternetService=DSL AND OnlineBackup=Yes AND PhoneService=Yes', 'No', 0.103953, 0.918342, 0.473764);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (37, 37, 'Contract=Month-to-month AND InternetService=Fiber optic AND OnlineSecurity=No', 'Yes', 0.146615, 0.581172, 0.473667);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (38, 38, 'InternetService=DSL AND OnlineSecurity=Yes AND SeniorCitizen=0', 'No', 0.140358, 0.913043, 0.473521);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (39, 39, 'Contract=Month-to-month AND InternetService=Fiber optic AND PaymentMethod=Electronic check AND TechSupport=No', 'Yes', 0.101820, 0.629174, 0.473468);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (40, 40, 'DeviceProtection=Yes AND TechSupport=Yes AND TotalCharges=TotalCharges_(3273.68, +∞]', 'No', 0.110495, 0.916274, 0.472530);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (41, 41, 'Contract=Month-to-month AND InternetService=Fiber optic AND MonthlyCharges=MonthlyCharges_(68.825, 106.775] AND TechSupport=No', 'Yes', 0.145762, 0.578769, 0.471861);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (42, 42, 'OnlineSecurity=Yes AND SeniorCitizen=0 AND TechSupport=Yes', 'No', 0.126280, 0.912641, 0.471066);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (43, 43, 'OnlineSecurity=Yes AND TechSupport=Yes', 'No', 0.141780, 0.909672, 0.470303);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (44, 44, 'InternetService=DSL AND Partner=Yes AND PhoneService=Yes AND SeniorCitizen=0', 'No', 0.101820, 0.914432, 0.469392);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (45, 45, 'Contract=Month-to-month AND MonthlyCharges=MonthlyCharges_(68.825, 106.775] AND OnlineSecurity=No AND PaperlessBilling=Yes', 'Yes', 0.121587, 0.598321, 0.468679);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (46, 46, 'Contract=Month-to-month AND OnlineSecurity=No AND PaperlessBilling=Yes AND PaymentMethod=Electronic check', 'Yes', 0.102247, 0.618229, 0.468216);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (47, 47, 'Contract=Month-to-month AND PaperlessBilling=Yes AND PaymentMethod=Electronic check AND TechSupport=No', 'Yes', 0.102105, 0.617900, 0.467973);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (48, 48, 'Contract=Month-to-month AND OnlineSecurity=No AND PaperlessBilling=Yes AND TechSupport=No', 'Yes', 0.133959, 0.583643, 0.467800);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (49, 49, 'Contract=Month-to-month AND OnlineSecurity=No AND TechSupport=No', 'Yes', 0.168515, 0.547344, 0.467622);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (50, 50, 'InternetService=DSL AND OnlineSecurity=Yes', 'No', 0.151308, 0.904762, 0.466867);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (51, 51, 'tenure=tenure_(59.5, 70.5]', 'No', 0.122867, 0.908517, 0.466504);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (52, 52, 'InternetService=DSL AND SeniorCitizen=0 AND TechSupport=Yes', 'No', 0.139790, 0.905991, 0.466433);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (53, 53, 'InternetService=DSL AND Partner=Yes AND PhoneService=Yes', 'No', 0.110637, 0.909942, 0.466184);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (54, 54, 'MultipleLines=No AND PaperlessBilling=No AND SeniorCitizen=0 AND TotalCharges=TotalCharges_(347.9, 3273.68]', 'No', 0.103953, 0.910336, 0.465643);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (55, 55, 'InternetService=DSL AND TechSupport=Yes', 'No', 0.150739, 0.902896, 0.465026);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (56, 56, 'Dependents=Yes AND PaperlessBilling=No AND Partner=Yes AND SeniorCitizen=0', 'No', 0.107935, 0.908982, 0.464890);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (57, 57, 'PaperlessBilling=No AND Partner=Yes AND PhoneService=Yes AND SeniorCitizen=0', 'No', 0.147184, 0.903141, 0.464765);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (58, 58, 'PaperlessBilling=No AND SeniorCitizen=0 AND TotalCharges=TotalCharges_(347.9, 3273.68]', 'No', 0.171359, 0.897914, 0.463261);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (59, 59, 'DeviceProtection=Yes AND OnlineSecurity=Yes AND SeniorCitizen=0', 'No', 0.123862, 0.903527, 0.461951);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (60, 60, 'Dependents=Yes AND PaperlessBilling=No AND Partner=Yes', 'No', 0.109215, 0.903529, 0.460005);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (61, 61, 'InternetService=DSL AND OnlineBackup=Yes AND SeniorCitizen=0', 'No', 0.122156, 0.901364, 0.459766);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (62, 62, 'Dependents=No AND OnlineSecurity=No AND PaymentMethod=Electronic check AND TechSupport=No', 'Yes', 0.101536, 0.601516, 0.459438);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (63, 63, 'DeviceProtection=Yes AND OnlineSecurity=Yes', 'No', 0.141496, 0.898014, 0.459338);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (64, 64, 'DeviceProtection=Yes AND InternetService=DSL AND SeniorCitizen=0', 'No', 0.120592, 0.900212, 0.458537);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (65, 65, 'OnlineBackup=Yes AND OnlineSecurity=Yes AND SeniorCitizen=0', 'No', 0.122298, 0.899582, 0.458203);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (66, 66, 'Contract=One year AND SeniorCitizen=0', 'No', 0.162827, 0.893136, 0.457872);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (67, 67, 'PaperlessBilling=No AND Partner=Yes AND PhoneService=Yes', 'No', 0.159841, 0.891356, 0.455954);
INSERT INTO regras (id, ordem, antecedente_serializado, classe_predita, suporte, confianca, score) VALUES (68, 68, 'Contract=One year', 'No', 0.185722, 0.887228, 0.455800);
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
        WHEN c."Contract" = 'Two year' AND c."Partner" = 'Yes' AND c."PhoneService" = 'Yes' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."Contract" = 'Two year' AND c."Partner" = 'Yes' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."Contract" = 'Two year' AND c."Dependents" = 'Yes' THEN 'No'
        WHEN c."Contract" = 'Two year' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."Contract" = 'Two year' THEN 'No'
        WHEN c."DeviceProtection" = 'No internet service' AND c."InternetService" = 'No' AND c."MonthlyCharges" = 'MonthlyCharges_(-∞, 29.375]' AND c."Partner" = 'Yes' THEN 'No'
        WHEN c."MonthlyCharges" = 'MonthlyCharges_(-∞, 29.375]' AND c."Partner" = 'Yes' THEN 'No'
        WHEN c."DeviceProtection" = 'No internet service' AND c."InternetService" = 'No' AND c."PaperlessBilling" = 'No' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."DeviceProtection" = 'No internet service' AND c."InternetService" = 'No' AND c."MonthlyCharges" = 'MonthlyCharges_(-∞, 29.375]' AND c."PaperlessBilling" = 'No' THEN 'No'
        WHEN c."DeviceProtection" = 'No internet service' THEN 'No'
        WHEN c."MonthlyCharges" = 'MonthlyCharges_(-∞, 29.375]' AND c."PaperlessBilling" = 'No' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."OnlineSecurity" = 'Yes' AND c."SeniorCitizen" = '0' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]' THEN 'No'
        WHEN c."Partner" = 'Yes' AND c."TechSupport" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]' THEN 'No'
        WHEN c."MonthlyCharges" = 'MonthlyCharges_(-∞, 29.375]' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."DeviceProtection" = 'Yes' AND c."OnlineSecurity" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]' THEN 'No'
        WHEN c."OnlineBackup" = 'Yes' AND c."OnlineSecurity" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]' THEN 'No'
        WHEN c."OnlineSecurity" = 'Yes' AND c."Partner" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]' THEN 'No'
        WHEN c."OnlineBackup" = 'Yes' AND c."TechSupport" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]' THEN 'No'
        WHEN c."Contract" = 'Month-to-month' AND c."InternetService" = 'Fiber optic' AND c."OnlineSecurity" = 'No' AND c."TechSupport" = 'No' THEN 'Yes'
        WHEN c."SeniorCitizen" = '0' AND c."TechSupport" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]' THEN 'No'
        WHEN c."OnlineSecurity" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]' THEN 'No'
        WHEN c."Contract" = 'Month-to-month' AND c."MonthlyCharges" = 'MonthlyCharges_(68.825, 106.775]' AND c."OnlineSecurity" = 'No' AND c."TechSupport" = 'No' THEN 'Yes'
        WHEN c."Contract" = 'Month-to-month' AND c."InternetService" = 'Fiber optic' AND c."OnlineSecurity" = 'No' AND c."PaymentMethod" = 'Electronic check' THEN 'Yes'
        WHEN c."Contract" = 'Month-to-month' AND c."MonthlyCharges" = 'MonthlyCharges_(68.825, 106.775]' AND c."OnlineSecurity" = 'No' AND c."PaymentMethod" = 'Electronic check' THEN 'Yes'
        WHEN c."Contract" = 'Month-to-month' AND c."OnlineSecurity" = 'No' AND c."PaymentMethod" = 'Electronic check' AND c."TechSupport" = 'No' THEN 'Yes'
        WHEN c."Contract" = 'Month-to-month' AND c."InternetService" = 'Fiber optic' AND c."MonthlyCharges" = 'MonthlyCharges_(68.825, 106.775]' AND c."OnlineSecurity" = 'No' THEN 'Yes'
        WHEN c."SeniorCitizen" = '0' AND c."tenure" = 'tenure_(59.5, 70.5]' THEN 'No'
        WHEN c."InternetService" = 'DSL' AND c."OnlineBackup" = 'Yes' AND c."PhoneService" = 'Yes' THEN 'No'
        WHEN c."Contract" = 'Month-to-month' AND c."InternetService" = 'Fiber optic' AND c."OnlineSecurity" = 'No' THEN 'Yes'
        WHEN c."InternetService" = 'DSL' AND c."OnlineSecurity" = 'Yes' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."Contract" = 'Month-to-month' AND c."InternetService" = 'Fiber optic' AND c."PaymentMethod" = 'Electronic check' AND c."TechSupport" = 'No' THEN 'Yes'
        WHEN c."DeviceProtection" = 'Yes' AND c."TechSupport" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]' THEN 'No'
        WHEN c."Contract" = 'Month-to-month' AND c."InternetService" = 'Fiber optic' AND c."MonthlyCharges" = 'MonthlyCharges_(68.825, 106.775]' AND c."TechSupport" = 'No' THEN 'Yes'
        WHEN c."OnlineSecurity" = 'Yes' AND c."SeniorCitizen" = '0' AND c."TechSupport" = 'Yes' THEN 'No'
        WHEN c."OnlineSecurity" = 'Yes' AND c."TechSupport" = 'Yes' THEN 'No'
        WHEN c."InternetService" = 'DSL' AND c."Partner" = 'Yes' AND c."PhoneService" = 'Yes' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."Contract" = 'Month-to-month' AND c."MonthlyCharges" = 'MonthlyCharges_(68.825, 106.775]' AND c."OnlineSecurity" = 'No' AND c."PaperlessBilling" = 'Yes' THEN 'Yes'
        WHEN c."Contract" = 'Month-to-month' AND c."OnlineSecurity" = 'No' AND c."PaperlessBilling" = 'Yes' AND c."PaymentMethod" = 'Electronic check' THEN 'Yes'
        WHEN c."Contract" = 'Month-to-month' AND c."PaperlessBilling" = 'Yes' AND c."PaymentMethod" = 'Electronic check' AND c."TechSupport" = 'No' THEN 'Yes'
        WHEN c."Contract" = 'Month-to-month' AND c."OnlineSecurity" = 'No' AND c."PaperlessBilling" = 'Yes' AND c."TechSupport" = 'No' THEN 'Yes'
        WHEN c."Contract" = 'Month-to-month' AND c."OnlineSecurity" = 'No' AND c."TechSupport" = 'No' THEN 'Yes'
        WHEN c."InternetService" = 'DSL' AND c."OnlineSecurity" = 'Yes' THEN 'No'
        WHEN c."tenure" = 'tenure_(59.5, 70.5]' THEN 'No'
        WHEN c."InternetService" = 'DSL' AND c."SeniorCitizen" = '0' AND c."TechSupport" = 'Yes' THEN 'No'
        WHEN c."InternetService" = 'DSL' AND c."Partner" = 'Yes' AND c."PhoneService" = 'Yes' THEN 'No'
        WHEN c."MultipleLines" = 'No' AND c."PaperlessBilling" = 'No' AND c."SeniorCitizen" = '0' AND c."TotalCharges" = 'TotalCharges_(347.9, 3273.68]' THEN 'No'
        WHEN c."InternetService" = 'DSL' AND c."TechSupport" = 'Yes' THEN 'No'
        WHEN c."Dependents" = 'Yes' AND c."PaperlessBilling" = 'No' AND c."Partner" = 'Yes' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."PaperlessBilling" = 'No' AND c."Partner" = 'Yes' AND c."PhoneService" = 'Yes' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."PaperlessBilling" = 'No' AND c."SeniorCitizen" = '0' AND c."TotalCharges" = 'TotalCharges_(347.9, 3273.68]' THEN 'No'
        WHEN c."DeviceProtection" = 'Yes' AND c."OnlineSecurity" = 'Yes' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."Dependents" = 'Yes' AND c."PaperlessBilling" = 'No' AND c."Partner" = 'Yes' THEN 'No'
        WHEN c."InternetService" = 'DSL' AND c."OnlineBackup" = 'Yes' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."Dependents" = 'No' AND c."OnlineSecurity" = 'No' AND c."PaymentMethod" = 'Electronic check' AND c."TechSupport" = 'No' THEN 'Yes'
        WHEN c."DeviceProtection" = 'Yes' AND c."OnlineSecurity" = 'Yes' THEN 'No'
        WHEN c."DeviceProtection" = 'Yes' AND c."InternetService" = 'DSL' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."OnlineBackup" = 'Yes' AND c."OnlineSecurity" = 'Yes' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."Contract" = 'One year' AND c."SeniorCitizen" = '0' THEN 'No'
        WHEN c."PaperlessBilling" = 'No' AND c."Partner" = 'Yes' AND c."PhoneService" = 'Yes' THEN 'No'
        WHEN c."Contract" = 'One year' THEN 'No'
        ELSE 'No'
    END AS classe_predita
FROM cliente_exemplo c;

-- Mesma decisão em forma relacional: filtra as regras que casam e
-- ordena pela POSIÇÃO NO RANKING do classificador (coluna `ordem`,
-- ascendente) — a 1ª regra que casa vence, exatamente como na
-- predição [O]. O `score` fica na tabela para auditoria, mas não
-- define a ordem (no CBA o desempate da precedência é suporte e
-- cardinalidade, não o score).
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
               OR (r.id = 9 AND c."Contract" = 'Two year' AND c."Partner" = 'Yes' AND c."PhoneService" = 'Yes' AND c."SeniorCitizen" = '0')
               OR (r.id = 10 AND c."Contract" = 'Two year' AND c."Partner" = 'Yes' AND c."SeniorCitizen" = '0')
               OR (r.id = 11 AND c."Contract" = 'Two year' AND c."Dependents" = 'Yes')
               OR (r.id = 12 AND c."Contract" = 'Two year' AND c."SeniorCitizen" = '0')
               OR (r.id = 13 AND c."Contract" = 'Two year')
               OR (r.id = 14 AND c."DeviceProtection" = 'No internet service' AND c."InternetService" = 'No' AND c."MonthlyCharges" = 'MonthlyCharges_(-∞, 29.375]' AND c."Partner" = 'Yes')
               OR (r.id = 15 AND c."MonthlyCharges" = 'MonthlyCharges_(-∞, 29.375]' AND c."Partner" = 'Yes')
               OR (r.id = 16 AND c."DeviceProtection" = 'No internet service' AND c."InternetService" = 'No' AND c."PaperlessBilling" = 'No' AND c."SeniorCitizen" = '0')
               OR (r.id = 17 AND c."DeviceProtection" = 'No internet service' AND c."InternetService" = 'No' AND c."MonthlyCharges" = 'MonthlyCharges_(-∞, 29.375]' AND c."PaperlessBilling" = 'No')
               OR (r.id = 18 AND c."DeviceProtection" = 'No internet service')
               OR (r.id = 19 AND c."MonthlyCharges" = 'MonthlyCharges_(-∞, 29.375]' AND c."PaperlessBilling" = 'No' AND c."SeniorCitizen" = '0')
               OR (r.id = 20 AND c."OnlineSecurity" = 'Yes' AND c."SeniorCitizen" = '0' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]')
               OR (r.id = 21 AND c."Partner" = 'Yes' AND c."TechSupport" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]')
               OR (r.id = 22 AND c."MonthlyCharges" = 'MonthlyCharges_(-∞, 29.375]' AND c."SeniorCitizen" = '0')
               OR (r.id = 23 AND c."DeviceProtection" = 'Yes' AND c."OnlineSecurity" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]')
               OR (r.id = 24 AND c."OnlineBackup" = 'Yes' AND c."OnlineSecurity" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]')
               OR (r.id = 25 AND c."OnlineSecurity" = 'Yes' AND c."Partner" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]')
               OR (r.id = 26 AND c."OnlineBackup" = 'Yes' AND c."TechSupport" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]')
               OR (r.id = 27 AND c."Contract" = 'Month-to-month' AND c."InternetService" = 'Fiber optic' AND c."OnlineSecurity" = 'No' AND c."TechSupport" = 'No')
               OR (r.id = 28 AND c."SeniorCitizen" = '0' AND c."TechSupport" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]')
               OR (r.id = 29 AND c."OnlineSecurity" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]')
               OR (r.id = 30 AND c."Contract" = 'Month-to-month' AND c."MonthlyCharges" = 'MonthlyCharges_(68.825, 106.775]' AND c."OnlineSecurity" = 'No' AND c."TechSupport" = 'No')
               OR (r.id = 31 AND c."Contract" = 'Month-to-month' AND c."InternetService" = 'Fiber optic' AND c."OnlineSecurity" = 'No' AND c."PaymentMethod" = 'Electronic check')
               OR (r.id = 32 AND c."Contract" = 'Month-to-month' AND c."MonthlyCharges" = 'MonthlyCharges_(68.825, 106.775]' AND c."OnlineSecurity" = 'No' AND c."PaymentMethod" = 'Electronic check')
               OR (r.id = 33 AND c."Contract" = 'Month-to-month' AND c."OnlineSecurity" = 'No' AND c."PaymentMethod" = 'Electronic check' AND c."TechSupport" = 'No')
               OR (r.id = 34 AND c."Contract" = 'Month-to-month' AND c."InternetService" = 'Fiber optic' AND c."MonthlyCharges" = 'MonthlyCharges_(68.825, 106.775]' AND c."OnlineSecurity" = 'No')
               OR (r.id = 35 AND c."SeniorCitizen" = '0' AND c."tenure" = 'tenure_(59.5, 70.5]')
               OR (r.id = 36 AND c."InternetService" = 'DSL' AND c."OnlineBackup" = 'Yes' AND c."PhoneService" = 'Yes')
               OR (r.id = 37 AND c."Contract" = 'Month-to-month' AND c."InternetService" = 'Fiber optic' AND c."OnlineSecurity" = 'No')
               OR (r.id = 38 AND c."InternetService" = 'DSL' AND c."OnlineSecurity" = 'Yes' AND c."SeniorCitizen" = '0')
               OR (r.id = 39 AND c."Contract" = 'Month-to-month' AND c."InternetService" = 'Fiber optic' AND c."PaymentMethod" = 'Electronic check' AND c."TechSupport" = 'No')
               OR (r.id = 40 AND c."DeviceProtection" = 'Yes' AND c."TechSupport" = 'Yes' AND c."TotalCharges" = 'TotalCharges_(3273.68, +∞]')
               OR (r.id = 41 AND c."Contract" = 'Month-to-month' AND c."InternetService" = 'Fiber optic' AND c."MonthlyCharges" = 'MonthlyCharges_(68.825, 106.775]' AND c."TechSupport" = 'No')
               OR (r.id = 42 AND c."OnlineSecurity" = 'Yes' AND c."SeniorCitizen" = '0' AND c."TechSupport" = 'Yes')
               OR (r.id = 43 AND c."OnlineSecurity" = 'Yes' AND c."TechSupport" = 'Yes')
               OR (r.id = 44 AND c."InternetService" = 'DSL' AND c."Partner" = 'Yes' AND c."PhoneService" = 'Yes' AND c."SeniorCitizen" = '0')
               OR (r.id = 45 AND c."Contract" = 'Month-to-month' AND c."MonthlyCharges" = 'MonthlyCharges_(68.825, 106.775]' AND c."OnlineSecurity" = 'No' AND c."PaperlessBilling" = 'Yes')
               OR (r.id = 46 AND c."Contract" = 'Month-to-month' AND c."OnlineSecurity" = 'No' AND c."PaperlessBilling" = 'Yes' AND c."PaymentMethod" = 'Electronic check')
               OR (r.id = 47 AND c."Contract" = 'Month-to-month' AND c."PaperlessBilling" = 'Yes' AND c."PaymentMethod" = 'Electronic check' AND c."TechSupport" = 'No')
               OR (r.id = 48 AND c."Contract" = 'Month-to-month' AND c."OnlineSecurity" = 'No' AND c."PaperlessBilling" = 'Yes' AND c."TechSupport" = 'No')
               OR (r.id = 49 AND c."Contract" = 'Month-to-month' AND c."OnlineSecurity" = 'No' AND c."TechSupport" = 'No')
               OR (r.id = 50 AND c."InternetService" = 'DSL' AND c."OnlineSecurity" = 'Yes')
               OR (r.id = 51 AND c."tenure" = 'tenure_(59.5, 70.5]')
               OR (r.id = 52 AND c."InternetService" = 'DSL' AND c."SeniorCitizen" = '0' AND c."TechSupport" = 'Yes')
               OR (r.id = 53 AND c."InternetService" = 'DSL' AND c."Partner" = 'Yes' AND c."PhoneService" = 'Yes')
               OR (r.id = 54 AND c."MultipleLines" = 'No' AND c."PaperlessBilling" = 'No' AND c."SeniorCitizen" = '0' AND c."TotalCharges" = 'TotalCharges_(347.9, 3273.68]')
               OR (r.id = 55 AND c."InternetService" = 'DSL' AND c."TechSupport" = 'Yes')
               OR (r.id = 56 AND c."Dependents" = 'Yes' AND c."PaperlessBilling" = 'No' AND c."Partner" = 'Yes' AND c."SeniorCitizen" = '0')
               OR (r.id = 57 AND c."PaperlessBilling" = 'No' AND c."Partner" = 'Yes' AND c."PhoneService" = 'Yes' AND c."SeniorCitizen" = '0')
               OR (r.id = 58 AND c."PaperlessBilling" = 'No' AND c."SeniorCitizen" = '0' AND c."TotalCharges" = 'TotalCharges_(347.9, 3273.68]')
               OR (r.id = 59 AND c."DeviceProtection" = 'Yes' AND c."OnlineSecurity" = 'Yes' AND c."SeniorCitizen" = '0')
               OR (r.id = 60 AND c."Dependents" = 'Yes' AND c."PaperlessBilling" = 'No' AND c."Partner" = 'Yes')
               OR (r.id = 61 AND c."InternetService" = 'DSL' AND c."OnlineBackup" = 'Yes' AND c."SeniorCitizen" = '0')
               OR (r.id = 62 AND c."Dependents" = 'No' AND c."OnlineSecurity" = 'No' AND c."PaymentMethod" = 'Electronic check' AND c."TechSupport" = 'No')
               OR (r.id = 63 AND c."DeviceProtection" = 'Yes' AND c."OnlineSecurity" = 'Yes')
               OR (r.id = 64 AND c."DeviceProtection" = 'Yes' AND c."InternetService" = 'DSL' AND c."SeniorCitizen" = '0')
               OR (r.id = 65 AND c."OnlineBackup" = 'Yes' AND c."OnlineSecurity" = 'Yes' AND c."SeniorCitizen" = '0')
               OR (r.id = 66 AND c."Contract" = 'One year' AND c."SeniorCitizen" = '0')
               OR (r.id = 67 AND c."PaperlessBilling" = 'No' AND c."Partner" = 'Yes' AND c."PhoneService" = 'Yes')
               OR (r.id = 68 AND c."Contract" = 'One year')
            ORDER BY r.ordem ASC
            LIMIT 1
        ),
        'No'
    ) AS classe_predita
FROM cliente_exemplo c;

-- Saída da query de decisão de exemplo:
SELECT * FROM decisao_cliente;
