# Data loading, feature/target splitting, and train/test split
from pathlib import Path
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from lib.config import TEST_SIZE, RANDOM_STATE_SPLIT


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
    # Drops the target column and non-predictive identifiers
    if drop_cols is None:
        drop_cols = [target_col, "Session ID"]
    y = df[target_col]
    X = df.drop(columns=drop_cols)
    return X, y


def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE_SPLIT):
    # Stratified train/test split (uses config defaults)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
