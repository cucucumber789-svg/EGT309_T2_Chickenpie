# Shared preprocessing utilities for gas monitoring models
import sqlite3
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(db_path: Path) -> pd.DataFrame:
    # Connect to SQLite and load the gas_monitoring table
    con = sqlite3.connect(str(db_path))
    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = [row[0] for row in cursor.fetchall()]
    print("Tables in database:", table_names)
    df = pd.read_sql_query("SELECT * FROM gas_monitoring", con)
    con.close()
    return df


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "Activity Level",
    drop_cols: list[str] | None = None,
):
    # Separate DataFrame into feature matrix X and target vector y
    if drop_cols is None:
        drop_cols = [target_col, "Session ID"]
    y = df[target_col]
    X = df.drop(columns=drop_cols)
    return X, y


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    # Standardise numeric features with StandardScaler (fit on train only)
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    return X_train, X_test


def encode_categoricals(X_train: pd.DataFrame, X_test: pd.DataFrame):
    # One-hot encode categorical features and align test columns to train
    categorical_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    return X_train, X_test
