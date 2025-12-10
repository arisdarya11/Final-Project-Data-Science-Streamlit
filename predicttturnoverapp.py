import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image

# ======================== PAGE CONFIG ========================
st.set_page_config(
    page_title="Employee Turnover Prediction",
    page_icon="📉",
    layout="wide"
)


# ======================== LOAD FILES ========================
def load_file(filename):
    try:
        return joblib.load(filename)
    except:
        with open(filename, "rb") as f:
            return pickle.load(f)

model = load_file("xgb_attrition_model.pkl")
scaler = load_file("scaler.pkl")
encoder = load_file("encoder.pkl")

# ======================== IMAGE HEADER ========================
st.image("turnover-adalah.jpg", use_column_width=True)

# ======================== TITLE ========================
st.title("📉 Employee Turnover Prediction Dashboard")

# ======================== INPUT SECTION ========================
st.subheader("📌 Employee Data Input")

col1, col2 = st.columns(2)

with col1:
    satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.50)
    last_evaluation = st.slider("Last Evaluation Score", 0.0, 1.0, 0.60)
    number_project = st.number_input("Number of Projects", 1, 10, 3)

with col2:
    average_montly_hours = st.number_input("Average Monthly Hours", 50, 350, 160)
    time_spend_company = st.number_input("Years at Company", 1, 20, 3)

work_accident_label = st.selectbox("Work Accident?", ["No", "Yes"])
promotion_label = st.selectbox("Promotion in Last 5 Years?", ["No", "Yes"])

work_accident = 1 if work_accident_label == "Yes" else 0
promotion_last_5years = 1 if promotion_label == "Yes" else 0

salary = st.selectbox("Salary Level", ["low", "medium", "high"])
salary_encoded = encoder.transform([salary])[0]

# ======================== BUILD INPUT ========================
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

# ======================== PREDICT BUTTON ========================
predict_btn = st.button("🔮 Predict Turnover")

if predict_btn:

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.subheader("🎯 Prediction Result")

    # ======================== BEAUTIFUL RESULT CARD ========================
    if prediction == 1:
        st.markdown(f"""
        <div style="
            padding: 20px;
            border-radius: 12px;
            background-color: #ffdddd;
            border-left: 8px solid #ff4d4d;">
            <h2>⚠️ High Risk of Attrition</h2>
            <h3>Employee likely to <b>LEAVE</b></h3>
            <h3>Probability: <b>{probability:.2f}</b></h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="
            padding: 20px;
            border-radius: 12px;
            background-color: #ddffdd;
            border-left: 8px solid #37c837;">
            <h2>✅ Low Risk of Attrition</h2>
            <h3>Employee likely to <b>STAY</b></h3>
            <h3>Probability: <b>{probability:.2f}</b></h3>
        </div>
        """, unsafe_allow_html=True)

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
        st.info("Feature importance unavailable.")

    # ======================== SHAP EXPLAINABILITY ========================
    st.subheader("🔥 SHAP Explainability")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(scaled_input)
    sample_sv = shap_values[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(sample_sv, show=False)
    st.pyplot(fig)

