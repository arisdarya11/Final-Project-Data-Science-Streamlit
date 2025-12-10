import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go
import os

# ======================== PATH ABSOLUT ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
encoder_path = os.path.join(BASE_DIR, "encoder.pkl")
explainer_path = os.path.join(BASE_DIR, "explainer.pkl")

# ======================== LOAD MODEL ========================
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
encoder = joblib.load(encoder_path)
explainer_saved = joblib.load(explainer_path)

# ======================== PAGE CONFIG ========================
st.set_page_config(
    page_title="Employee Turnover Prediction",
    page_icon="📉",
    layout="wide"
)

# ======================== DARK MODE ========================
st.write("### 🌓 Dark Mode")
dark_mode = st.toggle("Aktifkan Dark Mode?", value=False)

if dark_mode:
    st.markdown(
        """
        <style>
        body, .stApp {
            background-color: #0d1117 !important;
            color: #ffffff !important;
        }
        .stSelectbox label, .stNumberInput label, .stSlider label {
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# ======================== MODEL COLUMNS ========================
model_columns = [
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "salary",
    "Work_accident",
    "promotion_last_5years"
]


# ======================== UI ========================
st.title("📉 Employee Turnover Prediction Dashboard")

col1, col2 = st.columns(2)

with col1:
    satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
    last_evaluation = st.slider("Last Evaluation Score", 0.0, 1.0, 0.5)
    number_project = st.number_input("Number of Projects", 1, 10, 3)
    average_montly_hours = st.number_input("Average Monthly Hours", 50, 350, 160)

with col2:
    time_spend_company = st.number_input("Years in Company", 1, 20, 3)
    work_accident = st.selectbox("Ever Had Work Accident?", ["No", "Yes"])
    promotion_last_5years = st.selectbox("Promoted in Last 5 Years?", ["No", "Yes"])
    salary = st.selectbox("Salary Level", ["low", "medium", "high"])

work_accident = 1 if work_accident == "Yes" else 0
promotion_last_5years = 1 if promotion_last_5years == "Yes" else 0
salary_encoded = encoder.transform([salary])[0]

# ======================== BUILD INPUT ========================
input_data = pd.DataFrame([[
    satisfaction_level,
    last_evaluation,
    number_project,
    average_montly_hours,
    time_spend_company,
    salary_encoded,
    work_accident,
    promotion_last_5years
]], columns=model_columns)

scaled_input = scaler.transform(input_data)


# ======================== PREDICT ========================
if st.button("🔮 Predict Turnover"):
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.subheader("🔍 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk — Employee Likely to LEAVE\nProbability: **{probability:.2f}**")
        risk_label = "High Risk of Attrition"
    else:
        st.success(f"✅ Low Risk — Employee Likely to STAY\nProbability: **{probability:.2f}**")
        risk_label = "Low Risk of Attrition"

    # ======================== RADAR CHART ========================
    st.subheader("🕸 Risk Radar Chart")

    radar_features = ["Satisfaction", "Evaluation", "Projects", "Monthly Hours", "Tenure"]
    radar_values = [
        satisfaction_level,
        last_evaluation,
        number_project / 10,
        average_montly_hours / 350,
        time_spend_company / 20
    ]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=radar_values,
        theta=radar_features,
        fill="toself"
    ))

    st.plotly_chart(fig_radar, use_container_width=True)

    # ======================== FEATURE IMPORTANCE ========================
    st.subheader("📊 Feature Importance")

    try:
        importances = model.feature_importances_
        fig_imp = go.Figure([go.Bar(x=model_columns, y=importances)])
        st.plotly_chart(fig_imp, use_container_width=True)
    except:
        st.info("Feature importance not available for this model.")

    # ======================== SHAP VALUE PLOT ========================
    st.subheader("🔥 SHAP Local Explanation")

    shap_values = explainer_saved(scaled_input)

    # Bar Plot SHAP
    fig_shap = go.Figure([go.Bar(
        x=np.abs(shap_values.values[0]),
        y=model_columns,
        orientation="h"
    )])
    fig_shap.update_layout(title="SHAP Feature Contribution")
    st.plotly_chart(fig_shap, use_container_width=True)
