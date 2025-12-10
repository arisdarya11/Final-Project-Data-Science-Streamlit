import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import plotly.express as px

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Employee Turnover Prediction",
    page_icon="📊",
    layout="wide"
)

# ---------------------------
# DARK MODE (GLOBAL STYLE)
# ---------------------------
dark_mode_css = """
<style>
    body, .stApp {
        background-color: #121212;
        color: #E0E0E0;
    }
    .stCheckbox, .stRadio, .stSelectbox, label, .stMarkdown, .stTextInput, .stNumberInput {
        color: #E0E0E0 !important;
    }
    .stButton>button {
        background-color: #333333 !important;
        color: #FFFFFF !important;
        border-radius: 8px;
        padding: 8px 20px;
    }
    .stDataFrame, .stTable {
        background-color: #1E1E1E !important;
    }
</style>
"""
st.markdown(dark_mode_css, unsafe_allow_html=True)

# ---------------------------
# LOAD MODEL & SHAP EXPLAINER
# ---------------------------
model = joblib.load("model.pkl")
explainer = joblib.load("explainer.pkl")

# ---------------------------
# USER INPUT FORM
# ---------------------------
st.title("📊 Employee Turnover Prediction Dashboard")

with st.form("prediction_form"):
    satisfaction = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
    evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.6)
    project = st.number_input("Number of Projects", 1, 10, 3)
    hours = st.number_input("Average Monthly Hours", 50, 350, 160)
    time_spend = st.number_input("Years at Company", 1, 15, 3)
    accident = st.selectbox("Work Accident", [0, 1])
    promotion = st.selectbox("Promotion in Last 5 Years", [0, 1])
    salary = st.selectbox("Salary Level", ["low", "medium", "high"])

    submit_btn = st.form_submit_button("Predict")

# Map salary to numeric
salary_map = {"low": 0, "medium": 1, "high": 2}

# ---------------------------
# MAKE PREDICTION
# ---------------------------
if submit_btn:
    input_data = pd.DataFrame([{
        "satisfaction_level": satisfaction,
        "last_evaluation": evaluation,
        "number_project": project,
        "average_montly_hours": hours,
        "time_spend_company": time_spend,
        "Work_accident": accident,
        "promotion_last_5years": promotion,
        "salary": salary_map[salary]
    }])

    st.subheader("🔍 Input Data")
    st.write(input_data)

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("📌 Prediction Result")
    st.metric("Turnover Probability", f"{prob:.2f}")
    st.write("Prediction:", "**Will Leave**" if prediction == 1 else "**Stay**")

    # ---------------------------
    # SHAP WATERFALL FIX
    # ---------------------------
    st.subheader("🔎 SHAP Explanation (Waterfall)")

    shap_values = explainer(input_data)

    # FIX: Use shap.plots.waterfall instead of internal API
    fig = shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    # ---------------------------
    # FEATURE IMPORTANCE
    # ---------------------------
    st.subheader("📈 Global Feature Importance")

    shap_values_global = explainer.shap_values(input_data)
    shap.summary_plot(shap_values_global, input_data, show=False)
    st.pyplot(bbox_inches='tight')
