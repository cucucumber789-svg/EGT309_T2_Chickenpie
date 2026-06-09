# Configuration hub — edit values here to change model behaviour without touching model scripts

# Train / test split
TEST_SIZE = 0.3
RANDOM_STATE_SPLIT = 100

# General model seed
RANDOM_STATE = 42
ZERO_DIVISION = 0

# Logistic Regression
LR_C_VALUES = [0.001, 0.01, 0.1, 1, 10, 100]
LR_MAX_ITER = 1000
LR_CLASS_WEIGHT = "balanced"

# K-Nearest Neighbors
KNN_N_NEIGHBORS = 5
KNN_METRIC = "euclidean"
KNN_PARAM_GRID = {
    "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
    "metric": ["euclidean", "manhattan", "minkowski"],
    "weights": ["uniform", "distance"],
    "p": [1, 2],
}
KNN_CV_SPLITS = 5
KNN_SCORING = "f1_macro"
KNN_N_JOBS = -1
KNN_VERBOSE = 1

# Random Forest
RF_N_ESTIMATORS = 100
RF_CLASS_WEIGHT = "balanced"
RF_N_JOBS = -1
RF_N_TREES_RANGE = [10, 50, 100, 150, 200, 250, 300]
