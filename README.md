# git_practical

## Overview

This project performs exploratory data analysis (EDA) on indoor air quality sensor data collected from an HVAC-monitored environment. The notebook examines readings from multiple gas sensors (CO2, CO, Metal Oxide), temperature, humidity, and environmental context (HVAC mode, light level, activity level).

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

1. **Data Loading** — Connects to `gas_monitoring.db` and loads the full dataset
2. **Initial Analysis** — Distribution and summary statistics for each sensor column, anomaly detection, and missing value analysis
3. **Data Cleaning**:
   - **Temperature**: replaces unrealistic values (>60°C) with NaN, then median imputation
   - **Humidity**: replaces physically impossible values (< 0 or > 100%) with NaN, then median imputation
   - **CO2_InfraredSensor**: clips negative values (sensor calibration error, min -17.47 ppm) to NaN, then median imputation
   - **CO2_ElectroChemicalSensor**: no missing or anomalous values found
   - **CO_GasSensor**: mode imputation for missing values (8.3% missing), cast to integer
   - **MetalOxideSensor_Unit2**: median imputation (14% missing; other units have 0 missing)
   - **HVAC Operation Mode**: standardised 23 inconsistent variant spellings into 6 canonical `lowercase_underscore` categories
   - **Activity Level**: standardised 6 variant spellings into 3 canonical forms (`Low Activity`, `Moderate Activity`, `High Activity`)
   - **Ambient Light Level**: 10.5% missing — diagnosed missing pattern (MCAR across Time of Day / HVAC modes), imputed with mode (`very_bright`) to avoid dropping 10% of rows
   - **Time of Day**: 3 consistent categories (`morning`, `afternoon`, `night`), no missing or casing issues
   - **Session ID**: unique row identifier (10000/10000 unique), no predictive value — dropped during modeling

## Requirements

- Python 3.x
- Libraries listed in `requirements.txt`
