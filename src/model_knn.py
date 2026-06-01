# ---------------------------------------------------------------------------
# k-Nearest Neighbors classifier
#
# TODO: Implement k-NN model following the same pattern as model_rf.py:
#   1. Load   — sqlite3.connect(DB_PATH), pd.read_sql_query
#   2. Clean  — from clean import clean_gas_monitoring
#   3. Split  — train_test_split(..., random_state=100, stratify=y)
#   4. Scale  — StandardScaler on numeric columns
#   5. Train  — KNeighborsClassifier(...).fit(X_train, y_train)
#   6. Evaluate — accuracy, precision, recall, F1, classification_report
# ---------------------------------------------------------------------------
