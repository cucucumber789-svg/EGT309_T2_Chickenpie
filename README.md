# EGT309_T2_Chickenpie

## Overview

This project performs exploratory data analysis (EDA) on indoor air quality sensor data collected from an HVAC-monitored environment, then trains classifiers to predict activity level from sensor readings. The notebook examines readings from multiple gas sensors (CO2, CO, Metal Oxide), temperature, humidity, and environmental context (HVAC mode, light level, activity level). Three models are implemented in `src/` — Random Forest (`model_rf.py`), Logistic Regression (`model_lr.py`), and k-Nearest Neighbors (`model_knn.py`) — each reading the raw `.example` database, applying the same cleaning rules discovered during EDA (`src/clean.py`), and outputting console-only results with no plot artifacts.

## Members 

Wei Guan - Random Forest
Shun Wei - Logistic Regression
Derek - KNN Model

## Dataset

- **File**: `gas_monitoring.db.example` (SQLite, 10,000 rows, 14 columns)
- The database is committed directly because the notebook requires the exact dataset for analysis — generating synthetic data or converting to SQL dumps would compromise reproducibility.
- The notebook automatically copies `gas_monitoring.db.example` → `gas_monitoring.db` on first run (the `.db` file is gitignored to prevent accidental modifications from being tracked).
- **Features**:
  - Time of Day, Temperature, Humidity
  - CO2_InfraredSensor, CO2_ElectroChemicalSensor
  - MetalOxideSensor (Units 1–4)
  - CO_GasSensor, Session ID
  - HVAC Operation Mode, Ambient Light Level, Activity Level

## Setup

### VS Code

1. Clone the repository
2. Install the **Python** and **Jupyter** extensions from the VS Code marketplace
3. Open the project folder in VS Code
4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Open `eda.ipynb` and click **Run All** (or run cells individually)

### Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `eda.ipynb` (File → Upload notebook)
3. Upload `gas_monitoring.db.example` to the Colab runtime (click the folder icon on the left → upload icon)
4. Run all cells (Runtime → Run all)

## Notebook Summary

1. **Data Loading** — Copies `gas_monitoring.db.example` → `gas_monitoring.db`, then loads the full dataset
2. **Initial Analysis** — Distribution and summary statistics for each sensor column, anomaly detection, and missing value analysis
3. **Data Cleaning**:
   - **Temperature**: replaces unrealistic values (>60°C) with NaN, then median imputation
   - **Humidity**: replaces physically impossible values (< 0 or > 100%) with NaN, then median imputation
   - **CO2_InfraredSensor**: clips negative values (sensor calibration error, min -17.47 ppm) to NaN, then median imputation
   - **CO2_ElectroChemicalSensor**: no missing or anomalous values found
   - **CO_GasSensor**: mode imputation for missing values (8.3% missing), cast to integer — values range 0–4 (not binary)
   - **MetalOxideSensor_Unit2**: median imputation (14% missing; other units have 0 missing)
   - **HVAC Operation Mode**: standardised 23 inconsistent variant spellings into 6 canonical `lowercase_underscore` categories
   - **Activity Level**: standardised 6 variant spellings into 3 canonical forms (`Low Activity`, `Moderate Activity`, `High Activity`)
   - **Ambient Light Level**: 10.5% missing — diagnosed missing pattern (MCAR across Time of Day / HVAC modes), imputed with mode (`very_bright`) to avoid dropping 10% of rows
   - **Time of Day**: 4 consistent categories (`morning`, `afternoon`, `evening`, `night`), no missing or casing issues
   - **Session ID**: unique row identifier (10000/10000 unique), no predictive value — dropped during modeling
4. **Feature vs. Target Analysis** — Examines how each feature relates to the `Activity Level` target:
   - Numeric sensor distributions by activity class (box plots)
   - Full correlation matrix across all sensors + target
   - Categorical cross-tabulations (HVAC mode, Time of Day, Ambient Light, CO_GasSensor vs. Activity Level)
   - Key finding: Metal Oxide sensors and CO2_ElectroChemicalSensor have the strongest predictive signal; Temperature and Humidity have near-zero correlation with the target

## File Structure & How It Works

The project has two independent tracks that share cleaning rules but never depend on each other:

```
gas_monitoring.db.example  (canonical raw data, never modified)
        │
        ├── eda.ipynb
        │     │  Reads raw data, applies inline cleaning cells,
        │     │  explores distributions, discovers cleaning rules.
        │     │  Self-contained — no imports from src/.
        │
        └── src/
              ├── clean.py         # Cleaning rules discovered during EDA (shared utility)
              │                      Called by model files only.
              │
              ├── model_rf.py      # Random Forest (n_estimators sweep)
              ├── model_lr.py      # Logistic Regression (scaled, regularization sweep)
              └── model_knn.py     # k-Nearest Neighbors (GridSearchCV tuning)
├── requirements.txt
└── README.md
```

**Key point:** The EDA is research — it discovers the cleaning rules. `src/clean.py` is another file that reimplements those rules. Each model imports only `clean.py`; none of them touch the notebook.

> **Gitignore notes:**
> - `*.db` — Local database copies (e.g. `gas_monitoring.db`) are regenerated by the notebook from `gas_monitoring.db.example`. The `.example` suffix is the committed canonical source; tracking local `.db` files would cause accidental drift between developers.
> - `__pycache__/` — Python bytecode cache, auto-generated on import and machine-specific (includes Python version in filename). Excluded to keep `src/` clutter-free and avoid diffs across different Python versions.

## Requirements

- **Python 3.x**
- **pandas** — Data loading, manipulation, and one-hot encoding
- **numpy** — Numerical operations (NaN handling, median imputation)
- **scikit-learn** — ML models (RF, LR, KNN), preprocessing (StandardScaler), metrics, cross-validation
- **seaborn** — Statistical visualisations (used in EDA notebook only)
- **matplotlib** — Plotting (used in EDA notebook only)

## Metrics used

For an eldercare early warning system, F1-Score is selected as the primary evaluation metric over standard classification accuracy due to inherent class imbalances in smart-home sensor logs. Because elderly residents spend a disproportionate amount of time resting, the dataset is naturally heavily skewed toward "Low Activity" instances. Relying on standard accuracy would reward a naive model that consistently predicts the majority class while failing entirely to catch critical movement transitions. F1-Score mitigates this bias by calculating the precision and recall for each activity class independently and taking their unweighted average. This ensures that "Low", "Moderate", and "High" activity states are treated with equal importance, directly penalizing the pipeline if it fails to accurately detect less frequent but potentially life-saving activity shifts.

## Terms

### Metrics (all models)

**F1-Score** — The harmonic mean of precision and recall. Unlike accuracy (which counts correct vs incorrect), F1 balances false positives and false negatives. This is our primary metric because it punishes models that guess the majority class and ignore rare but critical events. A high F1-Score means both precision and recall are strong — few false alarms and few missed detections.

**Precision** — Of all the times the model predicted a class (e.g. "High Activity"), how many were actually correct? High precision means few false alarms.

**Recall** — Of all the actual instances of a class, how many did the model catch? High recall means few missed events.

**Weighted F1 (wtd)** — Averages each class's F1 weighted by the number of samples in that class. This gives a representative view of overall performance since larger classes contribute more — if you randomly sample a prediction, this score reflects expected accuracy. However, because Low Activity makes up 57.7% of the dataset, a strong score on Low can mask poor performance on Moderate or High Activity. This is not ideal for eldercare where detecting rare events matters most. Weighted F1 is reported alongside macro F1 to give a complete picture of both per-sample and per-class performance. A high weighted F1 means the model performs well on the majority class (Low Activity), but can still miss rare events if macro F1 is low.

**Macro F1** — While weighted F1 reflects per-sample accuracy, macro F1 tells you whether the model works for ALL classes. It averages each class's F1 equally regardless of sample count, so a low F1 on High Activity hurts the score just as much as a low F1 on Low Activity. This is the stricter metric used to select the best model, because detecting rare activity shifts matters most. Together with weighted F1, macro ensures the model performs well on common cases without ignoring the critical rare ones. A high macro F1 means the model performs well on ALL classes equally — harder to achieve but safer for eldercare, since a rare missed event is penalised as much as a common one.

### Preprocessing (LR, KNN)

**StandardScaler** — A preprocessing step that rescales numeric features to mean 0 and variance 1. Required by LR so L2 regularisation treats all features fairly; required by KNN so large-scale features don't dominate distance calculations.

### Shared Configuration (RF, LR)

**class_weight="balanced"** — Automatically adjusts weights so minority classes (Moderate, High Activity) have higher importance during training, preventing the model from ignoring rare but critical events.

### Random Forest

**Ensemble** — A technique that combines multiple weak models (decision trees) to produce a stronger prediction. Random Forest builds hundreds of trees on random subsets of data and averages their outputs, reducing overfitting compared to a single tree.

**n_estimators** — The number of decision trees in the Random Forest. More trees generally improve performance but increase training time. RF sweeps this from 10 to 300 to find the optimal value.

### Logistic Regression

**Multinomial** — A classification strategy that treats all classes simultaneously using a softmax function, producing a probability for each class that sums to 1.

**L2 regularization** — The default penalty in logistic regression. It adds the sum of squared coefficients to the loss function, penalising large coefficients and preventing overfitting. In scikit-learn 1.8, L2 is auto-selected when using the lbfgs solver — even though `penalty='l2'` is no longer written explicitly in code.

**Regularization strength (C)** — Controls how much the model is allowed to grow its coefficients by controlling the L2 penalty. Small C means strong regularization (simpler model, less overfitting); large C means weak regularization (more complex, can overfit). This is the primary tuning knob for LR.

**L-BFGS** — The default optimisation algorithm used by Logistic Regression. It iteratively adjusts coefficients to minimise the loss function (error + L2 penalty). Unlike tree-based models (RF) or distance-based models (KNN), LR needs an optimiser to find its coefficients.

### K-Nearest Neighbors

**GridSearchCV** — An automated search that tries every combination of hyperparameters (e.g., different neighbor counts and distance metrics for KNN) using cross-validation, then picks the combination with the best score.

## Random Forest

An ensemble of decision trees tolerant of non-linear sensor interactions and resistant to noisy data (see Terms — Ensemble, n_estimators). Uses `class_weight="balanced"` (see Terms — class_weight) for class imbalance. Sweeps n_estimators 10 to 300, selecting the best value via macro F1-score.

## Logistic Regression

A linear classifier valued for interpretability — coefficients trace which sensors drive predictions (see Terms — Multinomial, L2 regularization, Regularization strength (C), L-BFGS). StandardScaler ensures fair L2 penalisation (see Terms — StandardScaler). Uses `class_weight="balanced"` (see Terms — class_weight). Sweeps C 0.001 to 100 via macro F1-score, suppressing noisy features (Temperature, Humidity) while preserving sensor signal.

## K-Nearest Neighbors

A distance-based classifier that captures non-linear thresholds and sensor clusters (see Terms — GridSearchCV). StandardScaler prevents large-scale sensors from dominating distance (see Terms — StandardScaler). Tunes neighbor counts, distance metrics, and weight configurations via GridSearchCV with 5-fold CV and macro F1 scoring.

## Training

Run any model with a single command after installing dependencies:

```bash
pip install -r requirements.txt
python src/model_rf.py   # Random Forest
python src/model_lr.py   # Logistic Regression
python src/model_knn.py  # k-Nearest Neighbors
```

Each model performs these steps:

| Step | What happens |
|------|-------------|
| 1. Load | Reads `gas_monitoring.db.example` (10,000 rows, 14 columns) |
| 2. Clean | Calls `clean_gas_monitoring()` from `src/clean.py` — caps outliers, imputes missing values, standardises labels |
| 3. Split | Separates features from target (`Activity Level`), one-hot encodes categoricals, stratifies 70/30 train/test |
| 4. Scale | Applies `StandardScaler` to numeric features (except Random Forest, which is scale-invariant) |
| 5. Train | Fits the classifier with tuning: RF sweeps `n_estimators` 10–300; LR sweeps regularization strength; KNN uses GridSearchCV over neighbors, metrics, and weights |
| 6. Evaluate | Prints accuracy, precision, recall, F1, and full classification report |

All output is printed to the console
