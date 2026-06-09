# Feature scaling and categorical encoding
import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    # Standardise numeric features with StandardScaler (fit on train only)
    # Used by: LR, KNN — tree-based models (RF) are scale-invariant
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
