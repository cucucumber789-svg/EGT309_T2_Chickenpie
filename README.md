# git_practical

## Overview

This project performs exploratory data analysis (EDA) on indoor air quality sensor data collected from an HVAC-monitored environment, then trains a Random Forest classifier to predict activity level from sensor readings. The notebook examines readings from multiple gas sensors (CO2, CO, Metal Oxide), temperature, humidity, and environmental context (HVAC mode, light level, activity level). The pipeline (`src/pipeline.py`) reads the raw `.example` database, applies the same cleaning rules discovered during EDA (`src/clean.py`), and trains a stratified Random Forest model — all output is console-only, no plot artifacts.

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
              │                      Called by pipeline.py only.
              │
              └── pipeline.py      # Loads .example → clean.py → trains model
                                     Console-only output, no plot files.
```

**Key point:** The EDA is research — it discovers the cleaning rules. `src/clean.py` is an engineering artifact that reimplements those rules. The pipeline imports only `clean.py`; it never touches the notebook.

## Training

Run the pipeline with a single command after installing dependencies:

```bash
pip install -r requirements.txt
python src/pipeline.py
```

The pipeline performs these steps:

| Step | What happens |
|------|-------------|
| 1. Load | Reads `gas_monitoring.db.example` (10,000 rows, 14 columns) |
| 2. Clean | Calls `clean_gas_monitoring()` from `src/clean.py` — caps outliers, imputes missing values, standardises labels |
| 3. Split | Separates features from target (`Activity Level`), one-hot encodes categoricals, stratifies 70/30 train/test |
| 4. Baseline | Trains a Random Forest (100 trees, balanced class weights) |
| 5. Tune | Sweeps `n_estimators` from 10 to 300, picks the best F1 score |
| 6. Final | Retrains with optimal trees, prints accuracy, precision, recall, F1, and full classification report |

All output is printed to the console — no files are created, no popup windows appear.

## Project Structure

├── eda.ipynb                  # Research & visualisation (self-contained, inline cleaning cells)
├── gas_monitoring.db.example  # Canonical raw data (never modified)
├── src/
│   ├── clean.py               # Cleaning rules discovered during EDA (shared utility)
│   └── pipeline.py            # Loads raw data, applies clean.py, trains classifier
├── requirements.txt
└── README.md

> **Gitignore notes:**
> - `*.db` — Local database copies (e.g. `gas_monitoring.db`) are regenerated by the notebook from `gas_monitoring.db.example`. The `.example` suffix is the committed canonical source; tracking local `.db` files would cause accidental drift between developers.
> - `__pycache__/` — Python bytecode cache, auto-generated on import and machine-specific (includes Python version in filename). Excluded to keep `src/` clutter-free and avoid diffs across different Python versions.

## Requirements

- Python 3.x
- Libraries listed in `requirements.txt`
