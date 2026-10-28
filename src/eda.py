"""EDA inicial do Telco Customer Churn — valida o dataset e reporta o perfil."""

from __future__ import annotations

import pandas as pd

RAW_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
ALVO = "Churn"


def carregar_dataset(caminho: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(caminho)
    # TotalCharges contém espaços em branco (11 linhas de tenure=0) -> coerção
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(bool).map({True: 1, False: 0})
    return df


def main() -> None:
    df = carregar_dataset()
    print("shape:", df.shape)
    print("\ntipos numéricos:", df.select_dtypes("number").columns.tolist())
    print("\nvalores ausentes:", int(df.isna().sum().sum()))
    nulos_tc = df["TotalCharges"].isna().sum()
    print(f"TotalCharges nulos: {nulos_tc} (tenure=0 confirmado: {(df.loc[df['TotalCharges'].isna(), 'tenure'] == 0).all()})")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

    bal = df[ALVO].value_counts()
    print("\nChurn:", dict(bal), f"| IR = {bal.max() / bal.min():.2f}")
    print("\nCardinalidade por atributo (>10 => MDLP no fold):")
    for col in df.columns:
        if col not in (ALVO, "customerID"):
            print(f"  {col:20s} nunique={df[col].nunique():4d} tipo={df[col].dtype}")

    churn_por_contrato = df.groupby("Contract")[ALVO].apply(lambda s: (s == "Yes").mean())
    print("\nChurn rate por Contract (sinal forte para regras):")
    print(churn_por_contrato.map("{:.1%}".format).to_string())


if __name__ == "__main__":
    main()
