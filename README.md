# Predictive Maintenance Data Preparation

This project prepares predictive maintenance data for a future machine learning model.

The main goal is to understand the raw CSV files, clean the data, explore it, and create useful features for predicting whether a machine may fail in the next 24 hours.

## Project Files

| File | Description |
| --- | --- |
| `pdm_data_cleaning_eda_feature_engineering.ipynb` | Main beginner-friendly notebook for data cleaning, EDA, and feature engineering |
| `pdm_model_training.ipynb` | Beginner-friendly notebook for training and comparing machine learning models |
| `PdM_telemetry.csv` | Hourly sensor readings for each machine |
| `PdM_errors.csv` | Machine error event records |
| `PdM_failures.csv` | Actual machine component failure records |
| `PdM_machines.csv` | Machine model and age information |
| `PdM_maint.csv` | Machine maintenance history |

## What the Notebook Covers

The data preparation notebook is written in a simple step-by-step style for students who are learning data science.

It covers:

1. Loading all CSV files
2. Understanding each dataset
3. Checking missing values
4. Checking duplicate rows
5. Cleaning and sorting data
6. Basic exploratory data analysis
7. Creating time-based features
8. Creating rolling sensor average features
9. Creating recent error count features
10. Creating maintenance-related features
11. Creating the target column `failure_next_24h`

The model training notebook covers:

1. Recreating the prepared dataset
2. Selecting feature columns and target column
3. Splitting data into train and test sets
4. Training Logistic Regression
5. Training Random Forest
6. Comparing model performance
7. Saving the best model

## Target Column

The target column is:

```text
failure_next_24h
```

It means:

- `1`: the machine fails within the next 24 hours
- `0`: the machine does not fail within the next 24 hours

This target can be used later for machine learning model training.

## How to Run

1. Install the required libraries.
2. Open `pdm_data_cleaning_eda_feature_engineering.ipynb`.
3. Run all cells from top to bottom.
4. This creates `pdm_modeling_data.csv` and `pdm_feature_columns.csv`.
5. Open `pdm_model_training.ipynb`.
6. Run all cells from top to bottom.
7. This trains models and saves the best model as `model.pkl`.

Important: run the data preparation notebook before the model training notebook.

## Required Python Libraries

The notebook uses:

```text
pandas
numpy
matplotlib
scikit-learn
joblib
```

Install all required libraries in one go with:

```bash
pip install -r requirements.txt
```

## Next Step

After model training, the next step can be creating a simple prediction app or prediction script.
