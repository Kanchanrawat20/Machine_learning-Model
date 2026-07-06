import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data():
    """
    Load, clean, preprocess, and encode the diabetes dataset.
    Returns:
        df_raw: Original dataset
        df_clean: Preprocessed dataset
        encoders: Label encoders for categorical columns
    """

    df = pd.read_csv("data/U.S._Chronic_Disease_Indicators.csv")

    df_raw = df.copy()

    df_clean = df[[
        "YearStart",
        "LocationDesc",
        "Topic",
        "DataValue",
        "DataValueType",
        "StratificationCategory1",
        "Stratification1"
    ]].copy()

    # Keep only Diabetes prevalence records
    df_clean = df_clean[
    (df_clean["Topic"] == "Diabetes") &
    (
        (df_clean["DataValueType"] == "Crude Prevalence") |
        (df_clean["DataValueType"] == "Age-adjusted Prevalence")
    )
].copy()

    # Remove missing values
    df_clean.dropna(inplace=True)
    df_clean.reset_index(drop=True, inplace=True)
    # Create target variable
    threshold = df_clean["DataValue"].median()

    df_clean["Prevalence_Level"] = np.where(
        df_clean["DataValue"] >= threshold,
        1,
        0
    )

    # Remove unnecessary columns
    df_clean.drop(
        columns=["Topic", "DataValue", "DataValueType"],
        inplace=True
    )

    # Encode categorical columns
    categorical_columns = [
        "LocationDesc",
        "StratificationCategory1",
        "Stratification1"
    ]

    encoders = {}

    for col in categorical_columns:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])
        encoders[col] = le

    return df_raw, df_clean, encoders