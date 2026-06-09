# Configuration hub — edit values here to change model behaviour without touching model scripts

# Train / test split
TEST_SIZE = 0.3           # Fraction of data held out for testing (0.0–1.0)
RANDOM_STATE_SPLIT = 100  # Seed for reproducible train/test split

# General model seed
RANDOM_STATE = 42         # Seed used by all classifiers for reproducible results
ZERO_DIVISION = 0         # Fallback value when metric divides by zero (0 avoids warnings)

# Logistic Regression
LR_C_VALUES = [0.001, 0.01, 0.1, 1, 10, 100]  # Regularisation strengths to sweep
LR_MAX_ITER = 1000        # Max solver iterations (increase if convergence warnings appear)
LR_CLASS_WEIGHT = "balanced"  # Auto-adjusts weights for imbalanced classes

# K-Nearest Neighbors
KNN_N_NEIGHBORS = 5       # Default neighbours for baseline model
KNN_METRIC = "euclidean"  # Default distance metric for baseline model
KNN_PARAM_GRID = {        # Hyperparameter combinations GridSearchCV will try
    "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
    "metric": ["euclidean", "manhattan", "minkowski"],
    "weights": ["uniform", "distance"],
    "p": [1, 2],
}
KNN_CV_SPLITS = 5         # Number of folds for cross-validation (higher = more stable but slower)
KNN_SCORING = "f1_macro"  # Metric used to select the best parameter combo
KNN_N_JOBS = -1           # Parallel workers (-1 = use all CPU cores)
KNN_VERBOSE = 1           # Progress output during GridSearchCV (0 = silent, 1 = brief, 2+ = detailed)

# Random Forest
RF_N_ESTIMATORS = 100     # Default tree count for baseline model
RF_CLASS_WEIGHT = "balanced"  # Auto-adjusts weights for imbalanced classes
RF_N_JOBS = -1            # Parallel workers (-1 = use all CPU cores)
RF_PARAM_GRID = {         # Hyperparameter combinations GridSearchCV will try
    "n_estimators": [100, 200],
    "max_depth": [None, 20],           # None = unlimited depth; 20 = cap tree growth
    "min_samples_split": [2, 5],       # Minimum samples to split a node (higher = simpler trees)
    "min_samples_leaf": [1, 2],        # Minimum samples per leaf (higher = simpler trees)
    "max_features": ["sqrt"],          # Features considered per split ("sqrt" = sqrt(n_features))
}
RF_CV_SPLITS = 5          # Number of folds for cross-validation (higher = more stable but slower)
RF_SCORING = "f1_macro"   # Metric used to select the best parameter combo
RF_VERBOSE = 1            # Progress output during GridSearchCV (0 = silent, 1 = brief, 2+ = detailed)
