import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import (
    encode_categoricals,   # one-hot encode + align train/test columns
    load_data,             # load gas_monitoring table from SQLite
    scale_features,        # StandardScaler — required by distance-based models
    split_features_target, # separate X (features) and y (Activity Level)
)

from clean import clean_gas_monitoring  # cap outliers, impute missing, standardise labels

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=100, stratify=y
)

X_train, X_test = scale_features(X_train, X_test)
X_train, X_test = encode_categoricals(X_train, X_test)

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
print(f"F1 (macro): {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ---------------------------------------------------------------------------
# 6. KNN Hyperparameter Tuning (GridSearchCV)
# ---------------------------------------------------------------------------
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'metric'     : ['euclidean', 'manhattan', 'minkowski'],
    'weights'    : ['uniform', 'distance'],
    'p'          : [1, 2],   # only used when metric='minkowski'
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator  = KNeighborsClassifier(),
    param_grid = param_grid,
    scoring    = 'f1_macro',
    cv         = cv,
    n_jobs     = -1,
    verbose    = 1,
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
print(f"Precision : {precision_score(y_test, y_pred_tuned, average='weighted', zero_division=0):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred_tuned, average='weighted', zero_division=0):.4f}")
print(f"F1 (wtd)  : {f1_score(y_test, y_pred_tuned, average='weighted', zero_division=0):.4f}")
print(f"F1 (macro): {f1_score(y_test, y_pred_tuned, average='macro', zero_division=0):.4f}")
print("\nClassification Report (Tuned):")
print(classification_report(y_test, y_pred_tuned, zero_division=0))

# ---------------------------------------------------------------------------
# 8. Baseline vs Tuned Comparison
# ---------------------------------------------------------------------------
print("\n=== Baseline vs Tuned Comparison ===")
print(f"Baseline F1 (macro): {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
print(f"Tuned    F1 (macro): {f1_score(y_test, y_pred_tuned, average='macro', zero_division=0):.4f}")
print(f"Baseline F1 (wtd)  : {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"Tuned    F1 (wtd)  : {f1_score(y_test, y_pred_tuned, average='weighted', zero_division=0):.4f}")

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