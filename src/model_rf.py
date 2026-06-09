import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lib.config import RANDOM_STATE, RF_CLASS_WEIGHT, RF_N_ESTIMATORS, RF_N_JOBS, RF_N_TREES_RANGE, ZERO_DIVISION
from lib.load_data import load_data, split_data, split_features_target
from lib.preprocess import encode_categoricals
from lib.clean import clean_gas_monitoring
# Note: scale_features intentionally omitted — tree-based models are scale-invariant

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
import joblib

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
DB_PATH = ROOT / "gas_monitoring.db.example"
df = load_data(DB_PATH)

# ---------------------------------------------------------------------------
# 2. Data cleaning (in-memory only — rules discovered in eda.ipynb)
# ---------------------------------------------------------------------------
df = clean_gas_monitoring(df)

# ---------------------------------------------------------------------------
# 3. Feature / target split
# ---------------------------------------------------------------------------
X, y = split_features_target(df)

# ---------------------------------------------------------------------------
# 4. Train / test split  (stratified, same seed across all models)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = split_data(X, y)

X_train, X_test = encode_categoricals(X_train, X_test)

# ---------------------------------------------------------------------------
# 5. Baseline Random Forest
# ---------------------------------------------------------------------------
clf_default = RandomForestClassifier(
    n_estimators=RF_N_ESTIMATORS,       # standard starting point
    class_weight=RF_CLASS_WEIGHT,  # guards against class imbalance
    random_state=RANDOM_STATE,
    n_jobs=RF_N_JOBS,
)
clf_default.fit(X_train, y_train)

y_pred_default = clf_default.predict(X_test)

print("\n=== Baseline Random Forest ===")
print(f"Accuracy  : {accuracy_score(y_test, y_pred_default):.4f}")
print(f"F1 (wtd)  : {f1_score(y_test, y_pred_default, average='weighted', zero_division=ZERO_DIVISION):.4f}")
print(f"F1 (macro): {f1_score(y_test, y_pred_default, average='macro', zero_division=ZERO_DIVISION):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_default, zero_division=ZERO_DIVISION))

# ---------------------------------------------------------------------------
# 6. Hyperparameter tuning — n_estimators sweep
# ---------------------------------------------------------------------------
f1_list = []

for n_trees in RF_N_TREES_RANGE:
    clf_tmp = RandomForestClassifier(
        n_estimators=n_trees,
        class_weight=RF_CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=RF_N_JOBS,
    )
    clf_tmp.fit(X_train, y_train)
    y_pred_tmp = clf_tmp.predict(X_test)
    f1 = f1_score(y_test, y_pred_tmp, average="macro", zero_division=ZERO_DIVISION)
    f1_list.append({"n_trees": n_trees, "f1_macro": round(f1, 4)})

results_df = pd.DataFrame(f1_list).set_index("n_trees")
print("\n=== n_estimators Sweep Results (macro F1) ===")
print(results_df.to_string())

# ---------------------------------------------------------------------------
# 7. Best model — re-train with optimal n_estimators
# ---------------------------------------------------------------------------
best_n = int(results_df["f1_macro"].idxmax())
print(f"\nBest n_estimators: {best_n}")

clf_best = RandomForestClassifier(
    n_estimators=best_n,
    class_weight=RF_CLASS_WEIGHT,
    random_state=RANDOM_STATE,
    n_jobs=RF_N_JOBS,
)
clf_best.fit(X_train, y_train)
y_pred_best = clf_best.predict(X_test)

print("\n=== Optimized Random Forest ===")
print(f"Accuracy  : {accuracy_score(y_test, y_pred_best):.4f}")
print(f"Precision : {precision_score(y_test, y_pred_best, average='weighted', zero_division=ZERO_DIVISION):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred_best, average='weighted', zero_division=ZERO_DIVISION):.4f}")
print(f"F1 (wtd)  : {f1_score(y_test, y_pred_best, average='weighted', zero_division=ZERO_DIVISION):.4f}")
print(f"F1 (macro): {f1_score(y_test, y_pred_best, average='macro', zero_division=ZERO_DIVISION):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, zero_division=ZERO_DIVISION))

# ---------------------------------------------------------------------------
# 8. Baseline vs Tuned Comparison
# ---------------------------------------------------------------------------
print("\n=== Baseline vs Tuned Comparison ===")
print(f"Baseline F1 (macro): {f1_score(y_test, y_pred_default, average='macro', zero_division=ZERO_DIVISION):.4f}")
print(f"Tuned    F1 (macro): {f1_score(y_test, y_pred_best, average='macro', zero_division=ZERO_DIVISION):.4f}")
print(f"Baseline F1 (wtd)  : {f1_score(y_test, y_pred_default, average='weighted', zero_division=ZERO_DIVISION):.4f}")
print(f"Tuned    F1 (wtd)  : {f1_score(y_test, y_pred_best, average='weighted', zero_division=ZERO_DIVISION):.4f}")

# ---------------------------------------------------------------------------
# 9. Optional — save tuned model
# ---------------------------------------------------------------------------
save_dir = ROOT / "src" / "models"
save_dir.mkdir(parents=True, exist_ok=True)

answer = input("Save tuned model? (y/n): ").strip().lower()
if answer == "y":
    path = save_dir / "model_rf_tuned.joblib"
    joblib.dump(clf_best, path)
    print(f"Model saved to {path}")

print("\nDone.")
