import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lib.config import RANDOM_STATE, RF_CLASS_WEIGHT, RF_CV_SPLITS, RF_N_ESTIMATORS, RF_N_JOBS, RF_PARAM_GRID, RF_SCORING, RF_VERBOSE, ZERO_DIVISION
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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
# 6. Hyperparameter tuning — GridSearchCV over RF params
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=RF_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(
        class_weight=RF_CLASS_WEIGHT,
        random_state=RANDOM_STATE,
    ),
    param_grid=RF_PARAM_GRID,
    scoring=RF_SCORING,
    cv=cv,
    n_jobs=RF_N_JOBS,
    verbose=RF_VERBOSE,
)
grid_search.fit(X_train, y_train)

print("\n=== Random Forest Hyperparameter Tuning Results ===")
print(f"Best Parameters : {grid_search.best_params_}")
print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")

# ---------------------------------------------------------------------------
# 7. Evaluate best model on test set
# ---------------------------------------------------------------------------
best_clf = grid_search.best_estimator_
y_pred_best = best_clf.predict(X_test)

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
    joblib.dump(best_clf, path)
    print(f"Model saved to {path}")

print("\nDone.")
