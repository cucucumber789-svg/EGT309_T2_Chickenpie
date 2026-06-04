# git_practical

## Overview

This project performs exploratory data analysis (EDA) on indoor air quality sensor data collected from an HVAC-monitored environment, then trains classifiers to predict activity level from sensor readings. The notebook examines readings from multiple gas sensors (CO2, CO, Metal Oxide), temperature, humidity, and environmental context (HVAC mode, light level, activity level). Three models are implemented in `src/` — Random Forest (`model_rf.py`), Logistic Regression (`model_lr.py`), and k-Nearest Neighbors (`model_knn.py`) — each reading the raw `.example` database, applying the same cleaning rules discovered during EDA (`src/clean.py`), and outputting console-only results with no plot artifacts.

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

## File Interaction & How It Works

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
              ├── clean.py         # Cleaning rules extracted from EDA findings
              │                      Called by model files only.
              │
              ├── model_rf.py      # Random Forest (n_estimators sweep)
              ├── model_lr.py      # Logistic Regression (scaled, multinomial)
              └── model_knn.py     # k-Nearest Neighbors (placeholder)
```

**Key point:** The EDA is research — it discovers the cleaning rules. `src/clean.py` is an engineering artifact that reimplements those rules. Each model imports only `clean.py`; none of them touch the notebook.

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
| 5. Train | Fits the classifier (RF sweeps `n_estimators` 10–300; LR uses multinomial with `max_iter=1000`) |
| 6. Evaluate | Prints accuracy, precision, recall, F1, and full classification report |

All output is printed to the console — no files are created, no popup windows appear.

## Project Structure

├── eda.ipynb                  # Research & visualisation (self-contained, inline cleaning cells)
├── gas_monitoring.db.example  # Canonical raw data (never modified)
├── src/
│   ├── clean.py               # Cleaning rules discovered during EDA (shared utility)
│   ├── model_rf.py            # Random Forest classifier (n_estimators sweep)
│   ├── model_lr.py            # Logistic Regression classifier (scaled, multinomial)
│   └── model_knn.py           # k-Nearest Neighbors classifier (placeholder)
├── requirements.txt
└── README.md

> **Gitignore notes:**
> - `*.db` — Local database copies (e.g. `gas_monitoring.db`) are regenerated by the notebook from `gas_monitoring.db.example`. The `.example` suffix is the committed canonical source; tracking local `.db` files would cause accidental drift between developers.
> - `__pycache__/` — Python bytecode cache, auto-generated on import and machine-specific (includes Python version in filename). Excluded to keep `src/` clutter-free and avoid diffs across different Python versions.

## Requirements

- Python 3.x
- Libraries listed in `requirements.txt`

## Metrics used

For an eldercare early warning system, F1-Score is selected as the primary evaluation metric over standard classification accuracy due to inherent class imbalances in smart-home sensor logs. Because elderly residents spend a disproportionate amount of time resting, the dataset is naturally heavily skewed toward "Low Activity" instances. Relying on standard accuracy would reward a naive model that consistently predicts the majority class while failing entirely to catch critical movement transitions. F1-Score mitigates this bias by calculating the precision and recall for each activity class independently and taking their unweighted average. This ensures that "Low", "Moderate", and "High" activity states are treated with equal importance, directly penalizing the pipeline if it fails to accurately detect less frequent but potentially life-saving activity shifts. 

## Random Forest Model

The model_rf.py script implements a modular machine learning pipeline to train, tune, and evaluate a Random Forest classifier. Following database ingestion, the framework excludes non-predictive features like Session ID, one-hot encodes categorical variables, and splits the data into stratified training and testing subsets using a fixed seed to ensure reproducibility. This ensemble tree-based framework is exceptionally well-suited for the eldercare monitoring problem due to its high tolerance for non-linear feature relationships and complex multi-sensor interactions, such as matching a simultaneous spike in humidity and specific gas levels to indoor human movement. Furthermore, its inherent resistance to overfitting is critical given the heavily noisy, synthetic, and contaminated environmental attributes identified during data profiling. To address the severe class imbalance where resting states dominate the logs, the architecture configures class_weight="balanced" to strictly penalize minority class misclassifications. An automated hyperparameter sweep optimizes n_estimators (from 10 to 300) based on the F1-score, before extracting the optimal tree count to re-train the final champion model and generate comprehensive classification metrics.

## Logistic regression

The model_lr.py script implements a structured machine learning pipeline to train and evaluate a multinomial Logistic Regression classifier. Following database ingestion, the framework drops non-predictive features like Session ID, standardizes numerical features using StandardScaler to ensure optimization convergence, and applies one-hot encoding to categorical variables. The data is then partitioned into stratified training and testing subsets using a fixed random seed to guarantee reproducibility. This linear classification model is highly suited for eldercare monitoring as it offers a computationally efficient, low-latency baseline with exceptional interpretability. By providing explicit coefficients for each sensor input, the model allows developers to trace how specific environmental fluctuations influence activity level predictions, satisfying the need for explainable AI. Performance is finalized using a maximum iteration threshold of 1,000 and evaluated via accuracy, precision, recall, and F1-score metrics.

## KNN model

The model_knn.py script implements a structured machine learning pipeline to train, tune, and evaluate a K-Nearest Neighbors (KNN) classifier. Following database ingestion, the script drops non-predictive variables like Session ID, standardizes numerical features using StandardScaler to prevent larger-scale sensor values from dominating distance metrics, and encodes categorical attributes. The data is split into stratified training and testing subsets using a fixed random seed to ensure reproducibility. This distance-based instance framework is highly suited for eldercare monitoring because it captures non-linear environmental thresholds and distinct sensor clusters without making restrictive assumptions about data distribution. An automated hyperparameter optimization sweep is conducted via GridSearchCV using 5-fold cross-validation to evaluate neighbor counts, distance metrics, and weight configurations. Finally, the optimal parameters are extracted to re-train the final classifier, which is evaluated across accuracy, precision, recall, and F1-score performance dimensions.

=======
>>>>>>> parent of 5041808 (linear regression brief explanation)



