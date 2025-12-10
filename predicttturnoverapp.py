import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import shap
import matplotlib.pyplot as plt
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go

# ===========================
# LOAD MODEL & PREPROCESSOR
# ===========================
def load_file(filename):
    try:
        return joblib.load(filename)
    except:
        return pickle.load(open(filename, "rb"))

model = load_file("xgb_attrition_model.pkl")
scaler = load_file("scaler.pkl")
encoder = load_file("encoder.pkl")

# ===========================
# FIXED COLUMN ORDER (PENTING)
# ===========================
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

# ===========================
# THEME SWITCH
# ===========================
dark_mode = st.toggle("🌙 Dark Mode")

if dark_mode:
    st.markdown("""
    <style>
        body { background-color: #0E1117; color: white; }
    </style>
    """, unsafe_allow_html=True)

# ===========================
# APP HEADER
# ===========================
st.title("🧠 Employee Attrition Prediction Dashboard")
st.caption("Modern ML-powered analytics with SHAP Explainability")

st.markdown("---")

# ===========================
# INPUT FORM
# ===========================
st.subheader("📥 Employee Data Input")

col1, col2 = st.columns(2)

with col1:
    satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
    last_evaluation = st.slider("Last Evaluation Score", 0.0, 1.0, 0.5)
    number_project = st.number_input("Number of Projects", 1, 10, 3)
    salary_text = st.selectbox("Salary Level", ["low", "medium", "high"])

with col2:
    average_montly_hours = st.number_input("Average Monthly Hours", 50, 350, 160)
    time_spend_company = st.number_input("Years at Company", 1, 20, 3)
    work_accident = st.selectbox("Work Accident", ["No", "Yes"])
    promotion_last_5years = st.selectbox("Promoted in Last 5 Years", ["No", "Yes"])

# Convert YES/NO to numeric
work_accident = 1 if work_accident == "Yes" else 0
promotion_last_5years = 1 if promotion_last_5years == "Yes" else 0

salary_encoded = encoder.transform([salary_text])[0]

# ===========================
# BUILD DATAFRAME
# ===========================
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

# ===========================
# PREDICT BUTTON
# ===========================
if st.button("🚀 Predict Employee Turnover", use_container_width=True):

    prediction = model.predict(scaled_input)[0]
    probability = float(model.predict_proba(scaled_input)[0][1])

    st.markdown("## 🔍 Prediction Result")

    if prediction == 1:
        st.error(f"**⚠️ Employee Likely to Leave**\n### Probability: **{probability:.2f}**")
    else:
        st.success(f"**✅ Employee Likely to Stay**\n### Probability: **{probability:.2f}**")

    # ===========================
    # RISK RADAR CHART
    # ===========================
    st.subheader("📊 Risk Radar Overview")

    radar_labels = [
        "Satisfaction",
        "Evaluation",
        "Projects",
        "Monthly Hours",
        "Tenure"
    ]

    radar_values = [
        satisfaction_level,
        last_evaluation,
        number_project / 10,
        average_montly_hours / 350,
        time_spend_company / 20
    ]

    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=radar_values + [radar_values[0]],
        theta=radar_labels + [radar_labels[0]],
        fill='toself'
    ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
        height=450
    )

    st.plotly_chart(fig_radar, use_container_width=True)

    # ===========================
    # FEATURE IMPORTANCE (BAR)
    # ===========================
    st.subheader("📊 Global Feature Importance")

    try:
        importance = model.feature_importances_
        df_imp = pd.DataFrame({
            "Feature": model_columns,
            "Importance": importance
        }).sort_values("Importance", ascending=False)

        fig_imp = px.bar(df_imp, x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig_imp)
    except:
        st.info("Feature importance unavailable.")

    # ===========================
    # SHAP EXPLAINABILITY
    # ===========================
    st.subheader("🔥 SHAP Explainability Dashboard")

    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(scaled_input)

    st.write("### Local SHAP Force Plot")
    fig_shap = shap.force_plot(explainer.expected_value, shap_values, input_data, matplotlib=True)
    st.pyplot(fig_shap)

    # ===========================
    # EXPORT PDF
    # ===========================
    st.subheader("📥 Download Prediction Report (PDF)")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Employee Attrition Prediction Report", ln=1)
    pdf.cell(200, 10, txt=f"Prediction: {'Leave' if prediction==1 else 'Stay'}", ln=1)
    pdf.cell(200, 10, txt=f"Probability: {probability:.2f}", ln=1)

    file_path = "prediction_report.pdf"
    pdf.output(file_path)

    with open(file_path, "rb") as f:
        st.download_button("⬇ Download PDF Report", f, file_name=file_path)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Machine Learning")
