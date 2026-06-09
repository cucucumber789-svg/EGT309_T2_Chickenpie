import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import (
    encode_categoricals,   # one-hot encode + align train/test columns
    load_data,             # load gas_monitoring table from SQLite
    scale_features,        # StandardScaler — required by distance-based models
    split_features_target, # separate X (features) and y (Activity Level)
)

from clean import clean_gas_monitoring  # cap outliers, impute missing, standardise labels

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
import joblib

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
DB_PATH = ROOT / "gas_monitoring.db.example"
df = load_data(DB_PATH)

# ---------------------------------------------------------------------------
# 2. Data cleaning (in-memory only)
# ---------------------------------------------------------------------------
df = clean_gas_monitoring(df)

# ---------------------------------------------------------------------------
# 3. Feature / target split
# ---------------------------------------------------------------------------
X, y = split_features_target(df)

# ---------------------------------------------------------------------------
# 4. Train / test split  (stratified, same seed across all models)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=100, stratify=y
)

X_train, X_test = scale_features(X_train, X_test)
X_train, X_test = encode_categoricals(X_train, X_test)

# ---------------------------------------------------------------------------
# 5. Multinomial Logistic Regression
# ---------------------------------------------------------------------------
clf = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced',
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n=== Multinomial Logistic Regression ===")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"F1 (wtd)  : {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"F1 (macro): {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ---------------------------------------------------------------------------
# 6. Hyperparameter tuning — regularization sweep
# ---------------------------------------------------------------------------
C_VALUES = [0.001, 0.01, 0.1, 1, 10, 100]
f1_list = []

for C in C_VALUES:
    clf_tmp = LogisticRegression(
        C=C,
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
    )
    clf_tmp.fit(X_train, y_train)
    y_pred_tmp = clf_tmp.predict(X_test)
    f1_macro = f1_score(y_test, y_pred_tmp, average='macro', zero_division=0)
    f1_wtd = f1_score(y_test, y_pred_tmp, average='weighted', zero_division=0)
    f1_list.append({"Strength": C, "f1_macro": round(f1_macro, 4), "f1_weighted": round(f1_wtd, 4)})

results_df = pd.DataFrame(f1_list).set_index("Strength")
print("\n=== Regularization Sweep Results ===")
print(results_df.to_string())

# ---------------------------------------------------------------------------
# 7. Best model — re-train with optimal strength
# ---------------------------------------------------------------------------
best_C = float(results_df["f1_macro"].idxmax())
print(f"\nBest regularization strength: {best_C}")

clf_best = LogisticRegression(
    C=best_C,
    max_iter=1000,
    random_state=42,
    class_weight='balanced',
)
clf_best.fit(X_train, y_train)
y_pred_best = clf_best.predict(X_test)

print("\n=== Optimized Multinomial Logistic Regression ===")
print(f"Accuracy  : {accuracy_score(y_test, y_pred_best):.4f}")
print(f"Precision : {precision_score(y_test, y_pred_best, average='weighted', zero_division=0):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred_best, average='weighted', zero_division=0):.4f}")
print(f"F1 (wtd)  : {f1_score(y_test, y_pred_best, average='weighted', zero_division=0):.4f}")
print(f"F1 (macro): {f1_score(y_test, y_pred_best, average='macro', zero_division=0):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, zero_division=0))

# ---------------------------------------------------------------------------
# 8. Baseline vs Tuned Comparison
# ---------------------------------------------------------------------------
print("\n=== Baseline vs Tuned Comparison ===")
print(f"Baseline F1 (macro): {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
print(f"Tuned    F1 (macro): {f1_score(y_test, y_pred_best, average='macro', zero_division=0):.4f}")
print(f"Baseline F1 (wtd)  : {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"Tuned    F1 (wtd)  : {f1_score(y_test, y_pred_best, average='weighted', zero_division=0):.4f}")

# ---------------------------------------------------------------------------
# 9. Optional — save tuned model
# ---------------------------------------------------------------------------
save_dir = ROOT / "src" / "models"
save_dir.mkdir(parents=True, exist_ok=True)

answer = input("Save tuned model? (y/n): ").strip().lower()
if answer == "y":
    path = save_dir / "model_lr_tuned.joblib"
    joblib.dump(clf_best, path)
    print(f"Model saved to {path}")

print("\nDone.")
