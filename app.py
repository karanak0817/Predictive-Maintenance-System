from datetime import datetime

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Predictive Maintenance", layout="centered")


@st.cache_resource
def load_model_files():
    model = joblib.load("model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, feature_columns


def make_input_row(values, feature_columns):
    row = {column: 0 for column in feature_columns}

    row["machineID"] = values["machine_id"]
    row["volt"] = values["volt"]
    row["rotate"] = values["rotate"]
    row["pressure"] = values["pressure"]
    row["vibration"] = values["vibration"]
    row["age"] = values["age"]

    row["hour"] = values["prediction_time"].hour
    row["dayofweek"] = values["prediction_time"].weekday()
    row["month"] = values["prediction_time"].month

    # Simple first version: use current readings as recent average values.
    row["volt_mean_3h"] = values["volt"]
    row["volt_mean_24h"] = values["volt"]
    row["rotate_mean_3h"] = values["rotate"]
    row["rotate_mean_24h"] = values["rotate"]
    row["pressure_mean_3h"] = values["pressure"]
    row["pressure_mean_24h"] = values["pressure"]
    row["vibration_mean_3h"] = values["vibration"]
    row["vibration_mean_24h"] = values["vibration"]

    row["error_count_24h"] = values["error_count_24h"]
    row["days_since_maint"] = values["days_since_maint"]
    row["maint_overdue_30d"] = int(values["days_since_maint"] > 30)

    for error_name in values.get("error_types", []):
        column = f"error_{error_name}"
        if column in row:
            row[column] = 1

    model_column = f"model_{values['machine_model']}"
    if model_column in row:
        row[model_column] = 1

    return pd.DataFrame([row], columns=feature_columns)


def risk_label(probability):
    if probability >= 0.75:
        return "High Risk", "Machine may fail in the next 24 hours. Maintenance check is recommended."
    if probability >= 0.4:
        return "Medium Risk", "Machine needs monitoring. Check sensor readings and recent errors."
    return "Low Risk", "Machine looks stable based on the entered values."


def apply_maintenance_adjustment(probability, days_since_maint):
    adjustment = 0.0

    if days_since_maint > 30:
        adjustment = 0.10
    elif days_since_maint > 20:
        adjustment = 0.05
    elif days_since_maint > 14:
        adjustment = 0.02

    final_probability = min(probability + adjustment, 1.0)
    return final_probability, adjustment


model, feature_columns = load_model_files()

st.markdown(
    """
    <style>
        .stApp {
            background: #f6fbff;
            color: #1f2937;
        }

        .stApp,
        .stApp p,
        .stApp label,
        .stApp span,
        .stApp div {
            color: #1f2937;
        }

        .block-container {
            padding-top: 2rem;
            max-width: 860px;
        }

        header[data-testid="stHeader"] {
            background: #f6fbff !important;
        }

        div[data-testid="stToolbar"] {
            background: #f6fbff !important;
        }

        h1, h2, h3 {
            color: #0f3f5c !important;
            font-weight: 700;
        }

        .main-heading {
            background: #dff4ff;
            border: 1px solid #bae6fd;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
            text-align: left;
        }

        .main-heading h1 {
            margin: 0;
            color: #0f3f5c !important;
            font-size: 2.1rem;
        }

        .main-heading p {
            margin: 0.3rem 0 0 0;
            color: #334155 !important;
        }

        div[data-testid="stForm"] {
            background: #ffffff;
            border: 1px solid #dbeafe;
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }

        div[data-testid="stMetric"] {
            background: #ecfeff;
            border: 1px solid #bae6fd;
            border-radius: 10px;
            padding: 0.8rem;
        }

        .stButton > button,
        .stForm button {
            background: #0284c7;
            color: white !important;
            border-radius: 8px;
            border: none;
            font-weight: 600;
        }

        .stButton > button:hover,
        .stForm button:hover {
            background: #0369a1;
            color: white !important;
        }

        section[data-testid="stSidebar"] {
            background: #e0f2fe;
        }

        section[data-testid="stSidebar"] * {
            color: #1e293b !important;
        }

        input,
        textarea {
            background: #ffffff !important;
            color: #111827 !important;
        }

        div[data-baseweb="input"],
        div[data-baseweb="select"] > div {
            background: #ffffff !important;
            border-color: #bfdbfe !important;
        }

        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div {
            color: #111827 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("User Guide")
    st.write("1. Enter machine details.")
    st.write("2. Add current sensor readings.")
    st.write("3. Add recent error count and maintenance gap.")
    st.write("4. Click Predict Failure Risk.")
    st.info("This app predicts failure risk for the next 24 hours.")

st.markdown(
    """
    <div class="main-heading">
        <h1>Predictive Maintenance System</h1>
        <p>Enter machine details to estimate failure risk for the next 24 hours.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    This system uses machine sensor readings, recent error count, machine age,
    model type, and maintenance history to estimate whether a machine may fail soon.
    It is useful for planning maintenance before a breakdown happens.
    """
)

with st.expander("What do these inputs mean?"):
    st.write("**Voltage:** electrical reading from the machine.")
    st.write("**Rotation:** speed of rotating machine parts.")
    st.write("**Pressure:** operating pressure of the machine.")
    st.write("**Vibration:** vibration level during operation.")
    st.write("**Errors in Last 24 Hours:** recent warning or error events.")
    st.write("**Days Since Last Maintenance:** how long the machine has gone without maintenance.")

st.info("Tip: Higher recent errors, high vibration, and long maintenance gaps can increase failure risk.")

with st.form("prediction_form"):
    st.subheader("Machine Details")
    machine_id = st.number_input("Machine ID", min_value=1, max_value=100, value=1, step=1)
    machine_model = st.selectbox("Machine Model", ["model1", "model2", "model3", "model4"])
    age = st.slider("Machine Age", min_value=0, max_value=30, value=10, step=1)

    st.subheader("Sensor Readings")
    volt = st.number_input("Voltage", min_value=0.0, value=170.0)
    rotate = st.number_input("Rotation", min_value=0.0, value=450.0)
    pressure = st.number_input("Pressure", min_value=0.0, value=100.0)
    vibration = st.number_input("Vibration", min_value=0.0, value=40.0)

    st.subheader("Errors and Maintenance")
    error_count_24h = st.number_input("Errors in Last 24 Hours", min_value=0, value=0, step=1)
    days_since_maint = st.number_input("Days Since Last Maintenance", min_value=0, value=10, step=1)

    submitted = st.form_submit_button("Predict Failure Risk")

st.caption("Reference values from the dataset: Voltage around 170, Rotation around 447, Pressure around 101, Vibration around 40.")

if submitted:
    prediction_datetime = datetime.now()

    values = {
        "machine_id": int(machine_id),
        "machine_model": machine_model,
        "age": int(age),
        "volt": float(volt),
        "rotate": float(rotate),
        "pressure": float(pressure),
        "vibration": float(vibration),
        "error_count_24h": int(error_count_24h),
        "error_types": [],
        "days_since_maint": int(days_since_maint),
        "prediction_time": prediction_datetime,
    }

    input_data = make_input_row(values, feature_columns)
    base_probability = float(model.predict_proba(input_data)[0][1])
    probability, maintenance_adjustment = apply_maintenance_adjustment(
        base_probability,
        int(days_since_maint),
    )
    label, message = risk_label(probability)

    st.subheader("Prediction Result")
    st.metric("Failure Risk in Next 24 Hours", f"{probability:.2%}")
    if maintenance_adjustment > 0:
        st.caption(
            f"Base model risk: {base_probability:.2%}. "
            f"Maintenance safety adjustment added: {maintenance_adjustment:.0%}."
        )
    st.write(f"**Status:** {label}")
    st.write(message)

    if probability >= 0.75:
        st.error("Action: Inspect the machine as soon as possible.")
    elif probability >= 0.4:
        st.warning("Action: Monitor the machine closely and plan a check.")
    else:
        st.success("Action: Continue normal monitoring.")

    st.caption("Risk bands used in this app: Low below 40%, Medium from 40% to 74%, High from 75% and above.")

    summary = pd.DataFrame(
        {
            "Input": [
                "Machine ID",
                "Machine Model",
                "Machine Age",
                "Voltage",
                "Rotation",
                "Pressure",
                "Vibration",
                "Errors in Last 24 Hours",
                "Days Since Last Maintenance",
            ],
            "Value": [
                machine_id,
                machine_model,
                age,
                volt,
                rotate,
                pressure,
                vibration,
                error_count_24h,
                days_since_maint,
            ],
        }
    )

    st.subheader("Entered Details")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    with st.expander("Show model input row"):
        st.dataframe(input_data, use_container_width=True)
