import sqlite3
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
DB_PATH = ROOT / "gas_monitoring.db.example"

con = sqlite3.connect(str(DB_PATH))
cursor = con.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
table_names = [row[0] for row in cursor.fetchall()]
print("Tables in database:", table_names)

df = pd.read_sql_query("SELECT * FROM gas_monitoring", con)
con.close()

# ---------------------------------------------------------------------------
# 2. Data cleaning (in-memory only)
# ---------------------------------------------------------------------------
from clean import clean_gas_monitoring
df = clean_gas_monitoring(df)

# ---------------------------------------------------------------------------
# 3. Feature / target split
# ---------------------------------------------------------------------------
TARGET = "Activity Level"
DROP_COLS = [TARGET, "Session ID"]

y = df[TARGET]
X = df.drop(columns=DROP_COLS)

# Separate numeric and categorical columns for scaling
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

# Scale numeric features (important for logistic regression convergence)
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# One-hot encode categorical features
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# ---------------------------------------------------------------------------
# 4. Train / test split  (stratified, same seed across all models)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=100, stratify=y
)

# ---------------------------------------------------------------------------
# 5. K-Nearest Neighbors
# ---------------------------------------------------------------------------
clf = KNeighborsClassifier(
    n_neighbors=5,
    metric='euclidean',
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n=== K-Nearest Neighbors ===")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"F1 (wtd)  : {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nDone.")