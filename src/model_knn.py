import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lib.config import (
    KNN_CV_SPLITS,
    KNN_METRIC,
    KNN_N_JOBS,
    KNN_N_NEIGHBORS,
    KNN_PARAM_GRID,
    KNN_SCORING,
    KNN_VERBOSE,
    RANDOM_STATE,
    ZERO_DIVISION,
)
from lib.load_data import load_data, split_data, split_features_target
from lib.preprocess import encode_categoricals, scale_features
from lib.clean import clean_gas_monitoring

from sklearn.neighbors import KNeighborsClassifier
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
X_train, X_test, y_train, y_test = split_data(X, y)

X_train, X_test = scale_features(X_train, X_test)
X_train, X_test = encode_categoricals(X_train, X_test)

# ---------------------------------------------------------------------------
# 5. K-Nearest Neighbors
# ---------------------------------------------------------------------------
clf = KNeighborsClassifier(
    n_neighbors=KNN_N_NEIGHBORS,
    metric=KNN_METRIC,
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n=== K-Nearest Neighbors ===")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred, average='weighted', zero_division=ZERO_DIVISION):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred, average='weighted', zero_division=ZERO_DIVISION):.4f}")
print(f"F1 (wtd)  : {f1_score(y_test, y_pred, average='weighted', zero_division=ZERO_DIVISION):.4f}")
print(f"F1 (macro): {f1_score(y_test, y_pred, average='macro', zero_division=ZERO_DIVISION):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=ZERO_DIVISION))

# ---------------------------------------------------------------------------
# 6. KNN Hyperparameter Tuning (GridSearchCV)
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=KNN_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    estimator  = KNeighborsClassifier(),
    param_grid = KNN_PARAM_GRID,
    scoring    = KNN_SCORING,
    cv         = cv,
    n_jobs     = KNN_N_JOBS,
    verbose    = KNN_VERBOSE,
)

grid_search.fit(X_train, y_train)

print("\n=== KNN Hyperparameter Tuning Results ===")
print(f"Best Parameters : {grid_search.best_params_}")
print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")

# ---------------------------------------------------------------------------
# 7. Evaluate best model on test set
# ---------------------------------------------------------------------------
best_clf = grid_search.best_estimator_
y_pred_tuned = best_clf.predict(X_test)

print("\n=== Tuned KNN — Test Set Performance ===")
print(f"Accuracy  : {accuracy_score(y_test, y_pred_tuned):.4f}")
print(f"Precision : {precision_score(y_test, y_pred_tuned, average='weighted', zero_division=ZERO_DIVISION):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred_tuned, average='weighted', zero_division=ZERO_DIVISION):.4f}")
print(f"F1 (wtd)  : {f1_score(y_test, y_pred_tuned, average='weighted', zero_division=ZERO_DIVISION):.4f}")
print(f"F1 (macro): {f1_score(y_test, y_pred_tuned, average='macro', zero_division=ZERO_DIVISION):.4f}")
print("\nClassification Report (Tuned):")
print(classification_report(y_test, y_pred_tuned, zero_division=ZERO_DIVISION))

# ---------------------------------------------------------------------------
# 8. Baseline vs Tuned Comparison
# ---------------------------------------------------------------------------
print("\n=== Baseline vs Tuned Comparison ===")
print(f"Baseline F1 (macro): {f1_score(y_test, y_pred, average='macro', zero_division=ZERO_DIVISION):.4f}")
print(f"Tuned    F1 (macro): {f1_score(y_test, y_pred_tuned, average='macro', zero_division=ZERO_DIVISION):.4f}")
print(f"Baseline F1 (wtd)  : {f1_score(y_test, y_pred, average='weighted', zero_division=ZERO_DIVISION):.4f}")
print(f"Tuned    F1 (wtd)  : {f1_score(y_test, y_pred_tuned, average='weighted', zero_division=ZERO_DIVISION):.4f}")

# ---------------------------------------------------------------------------
# 9. Optional — save tuned model
# ---------------------------------------------------------------------------
save_dir = ROOT / "src" / "models"
save_dir.mkdir(parents=True, exist_ok=True)

answer = input("Save tuned model? (y/n): ").strip().lower()
if answer == "y":
    path = save_dir / "model_knn_tuned.joblib"
    joblib.dump(best_clf, path)
    print(f"Model saved to {path}")

print("\nDone.")