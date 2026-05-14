# Predictive Maintenance System

This project builds a simple predictive maintenance system using machine sensor data.

The main goal is to understand the raw CSV files, clean the data, explore it, create useful features, train machine learning models, and use a simple app to predict whether a machine may fail in the next 24 hours.

## Project Files

| File | Description |
| --- | --- |
| `pdm_data_cleaning_eda_feature_engineering.ipynb` | Main beginner-friendly notebook for data cleaning, EDA, and feature engineering |
| `pdm_model_training.ipynb` | Beginner-friendly notebook for training and comparing machine learning models |
| `app.py` | Streamlit app for predicting machine failure risk |
| `PdM_telemetry.csv` | Hourly sensor readings for each machine |
| `PdM_errors.csv` | Machine error event records |
| `PdM_failures.csv` | Actual machine component failure records |
| `PdM_machines.csv` | Machine model and age information |
| `PdM_maint.csv` | Machine maintenance history |
| `pdm_modeling_data.csv` | Prepared dataset created by the data preparation notebook |
| `pdm_feature_columns.csv` | Feature list created by the data preparation notebook |
| `model.pkl` | Saved best machine learning model |
| `feature_columns.pkl` | Saved feature list used by the prediction app |

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

The Streamlit app covers:

1. Loading the saved model
2. Loading the saved feature columns
3. Taking machine and sensor input from the user
4. Preparing the input in the same format used during training
5. Predicting failure risk for the next 24 hours
6. Showing the result as Low, Medium, or High Risk

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
7. This trains models and saves the best model as `model.pkl` and the training feature list as `feature_columns.pkl`.

Important: run the data preparation notebook before the model training notebook.

## How to Run the App

After running the notebooks and creating the saved model files, run the Streamlit app with:

```bash
streamlit run app.py
```

The app will open in the browser. Enter machine details, sensor readings, recent error count, and maintenance information to get the predicted failure risk.

Note: this app is a simple learning project. During training, rolling average features are created from historical sensor readings. In the app, the current sensor reading is used as a simple substitute for recent average values because the user enters only one set of readings.

## Required Python Libraries

The notebook uses:

```text
pandas
numpy
matplotlib
scikit-learn
joblib
streamlit
```

Install all required libraries in one go with:

```bash
pip install -r requirements.txt
```

## Possible Improvements

Future improvements can include:

1. Adding inputs for specific error types such as `error1`, `error2`, and `error3`
2. Using real historical readings to calculate rolling averages in the app
3. Showing charts for sensor trends
4. Improving the app design
5. Adding more model evaluation details
