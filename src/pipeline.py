import os
import sqlite3
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
DB_PATH = os.path.join("data", "gas_monitoring.db")  # relative path per project spec

con = sqlite3.connect(DB_PATH)
cursor = con.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
table_names = [row[0] for row in cursor.fetchall()]
print("Tables in database:", table_names)

df = pd.read_sql_query("SELECT * FROM gas_monitoring", con)
con.close()

# ---------------------------------------------------------------------------
# 2. Feature / target split
# ---------------------------------------------------------------------------
# 'Session ID' is an identifier with no predictive value — dropped.
# 'Activity Level' is the classification target.
TARGET = "Activity Level"
DROP_COLS = [TARGET, "Session ID"]

y = df[TARGET]
X = df.drop(columns=DROP_COLS)

# One-hot encode categorical features (Time of Day, HVAC Mode, Ambient Light, etc.)
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# ---------------------------------------------------------------------------
# 3. Train / test split  (stratified to preserve class distribution)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=100, stratify=y
)

# ---------------------------------------------------------------------------
# 4. Baseline Random Forest
# ---------------------------------------------------------------------------
clf_default = RandomForestClassifier(
    n_estimators=100,       # standard starting point
    class_weight="balanced",  # guards against class imbalance
    random_state=42,
    n_jobs=-1,
)
clf_default.fit(X_train, y_train)

y_pred_default = clf_default.predict(X_test)

print("\n=== Baseline Random Forest ===")
print(f"Accuracy  : {accuracy_score(y_test, y_pred_default):.4f}")
print(f"F1 (wtd)  : {f1_score(y_test, y_pred_default, average='weighted', zero_division=0):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_default, zero_division=0))

# ---------------------------------------------------------------------------
# 5. Hyperparameter tuning — n_estimators sweep
#    Range covers meaningful values; very small forests (< 10) are unreliable.
# ---------------------------------------------------------------------------
N_TREES_RANGE = [10, 50, 100, 150, 200, 250, 300]
f1_list = []

for n_trees in N_TREES_RANGE:
    clf_tmp = RandomForestClassifier(
        n_estimators=n_trees,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf_tmp.fit(X_train, y_train)
    y_pred_tmp = clf_tmp.predict(X_test)
    f1 = f1_score(y_test, y_pred_tmp, average="weighted", zero_division=0)
    f1_list.append({"n_trees": n_trees, "f1_weighted": round(f1, 4)})

results_df = pd.DataFrame(f1_list).set_index("n_trees")
print("\n=== n_estimators Sweep Results ===")
print(results_df.to_string())

# ---------------------------------------------------------------------------
# 6. Best model — re-train with optimal n_estimators
# ---------------------------------------------------------------------------
best_n = int(results_df["f1_weighted"].idxmax())
print(f"\nBest n_estimators: {best_n}")

clf_best = RandomForestClassifier(
    n_estimators=best_n,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
clf_best.fit(X_train, y_train)
y_pred_best = clf_best.predict(X_test)

print("\n=== Optimized Random Forest ===")
print(f"Accuracy  : {accuracy_score(y_test, y_pred_best):.4f}")
print(f"Precision : {precision_score(y_test, y_pred_best, average='weighted', zero_division=0):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred_best, average='weighted', zero_division=0):.4f}")
print(f"F1 (wtd)  : {f1_score(y_test, y_pred_best, average='weighted', zero_division=0):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, zero_division=0))

# ---------------------------------------------------------------------------
# 7. Confusion matrix plot
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax)
classes = clf_best.classes_
tick_marks = np.arange(len(classes))
ax.set_xticks(tick_marks)
ax.set_xticklabels(classes, rotation=45, ha="right")
ax.set_yticks(tick_marks)
ax.set_yticklabels(classes)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_ylabel("True Label")
ax.set_xlabel("Predicted Label")
ax.set_title("Confusion Matrix — Optimized Random Forest")
plt.tight_layout()
plt.savefig("confusion_matrix_rf.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# 8. Feature importance plot
# ---------------------------------------------------------------------------
feature_imp = (
    pd.Series(clf_best.feature_importances_, index=X.columns)
    .sort_values(ascending=False)
    .head(15)  # top 15 for readability
)

fig, ax = plt.subplots(figsize=(10, 6))
feature_imp.plot(kind="bar", ax=ax, color="steelblue")
ax.set_ylabel("Mean Decrease in Impurity (Relative Importance)")
ax.set_title("Top-15 Feature Importances — Optimized Random Forest")
plt.tight_layout()
plt.savefig("feature_importance_rf.png", dpi=150)
plt.show()

print("\nDone. Plots saved to confusion_matrix_rf.png and feature_importance_rf.png")
