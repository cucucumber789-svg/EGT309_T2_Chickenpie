import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import tree
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

con = sqlite3.connect('../gas_monitoring_cleaned.db')

cursor = con.cursor()
        # Query the sqlite_master table to get table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        # Fetch all results
table_names = [row[0] for row in cursor.fetchall()]

query = "SELECT * FROM gas_monitoring"
df = pd.read_sql_query(query, con)

X = df.drop(['Session ID'], axis=1)

# Define the target variable y (e.g., 'Activity Level')
y = df['Activity Level']

# Define the features X by dropping the target variable and 'Session ID'
# 'Session ID' is dropped as it's an identifier and not a predictive feature.
X = df.drop(['Activity Level', 'Session ID'], axis=1)

# Identify categorical columns in X
categorical_cols = X.select_dtypes(include=['object']).columns

# Apply one-hot encoding to the categorical features in X
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

clf = RandomForestClassifier()   #with default parameters
clf.fit(X_train,y_train)

#plot trees
print('The total number of trees are', len(clf.estimators_))

count = 0
for tree_in_forest in clf.estimators_:
  if count<5: #plot 5 only to save time
    plot_tree(tree_in_forest, filled=True)
    plt.show()
    count += 1

# Optimization of Parameters: number of trees
clf = RandomForestClassifier()
f1_list = []

# Iterate through a range of numbers of trees
for n_trees in range(2,10):
    # Use this to set the number of trees
    clf.set_params(n_estimators=n_trees)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    f1_list.append(pd.Series({'n_trees': n_trees, 'f1': f1}))

pd.concat(f1_list, axis=1).T.set_index('n_trees')

#feature importance
feature_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
fig = plt.figure()
ax = feature_imp.plot(kind='bar')
ax.set(ylabel='Relative Importance');