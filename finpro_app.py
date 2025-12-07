import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Employee Turnover Prediction",
    page_icon="📊",
    layout="centered"
)

# ===================== LOAD MODEL =====================
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("xgb_attrition_model.pkl")
        scaler = joblib.load("scaler.pkl")
        final_columns = joblib.load("final_columns.pkl")
        return model, scaler, final_columns, True
    except:
        return None, None, None, False

model, scaler, final_columns, real_model = load_artifacts()

# ===================== HEADER =====================
st.markdown("""
<h1 style='text-align:center;'>📊 Employee Turnover Prediction</h1>
<p style='text-align:center;color:gray;'>Aplikasi prediksi potensi karyawan resign berbasis Machine Learning</p>
""", unsafe_allow_html=True)

if not real_model:
    st.warning("⚠️ Model .pkl tidak ditemukan — aplikasi berjalan dalam DEMO MODE.")

# ===================== INPUT FORM =====================
with st.container():
    st.markdown("### 🧑‍💼 Data Karyawan")

    col1, col2 = st.columns(2)

    with col1:
        satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
        last_evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.6)
        number_project = st.number_input("Number of Projects", 1, 10, 3)
        avg_hours = st.number_input("Average Monthly Hours", 50, 350, 160)

    with col2:
        years_company = st.number_input("Years at Company", 1, 20, 3)
        work_accident = st.selectbox("Work Accident", [0, 1])
        promotion_last_5years = st.selectbox("Promotion in Last 5 Years", [0, 1])
        salary = st.selectbox("Salary Level", ["low", "medium", "high"])

# ===================== PREDICTION =====================
st.markdown("---")

if st.button("🚀 Predict Turnover", use_container_width=True):
    if not real_model:
        prob = np.random.uniform(0.2, 0.9)
        prediction = 1 if prob > 0.5 else 0
    else:
        # Bangun dataframe sesuai kolom model
        model_input = pd.DataFrame(0, index=[0], columns=final_columns)
        col_map = {
            "satisfaction_level": satisfaction_level,
            "last_evaluation": last_evaluation,
            "number_project": number_project,
            "average_montly_hours": avg_hours,
            "time_spend_company": years_company,
            "Work_accident": work_accident,
            "promotion_last_5years": promotion_last_5years,
            "salary": 0 if salary == "low" else 1 if salary == "medium" else 2
        }

        for k, v in col_map.items():
            if k in model_input.columns:
                model_input.loc[0, k] = v

        final_scaled = scaler.transform(model_input)
        prediction = model.predict(final_scaled)[0]
        prob = model.predict_proba(final_scaled)[0][1]

    # ===================== RESULT UI =====================
    st.markdown("## 📈 Prediction Result")

    if prediction == 1:
        st.markdown("<div style='background:#ffe5e5;padding:20px;border-radius:15px;'>"
                    "<h2 style='color:#d00000;'>⚠️ High Turnover Risk</h2>"
                    "</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='background:#e5ffe5;padding:20px;border-radius:15px;'>"
                    "<h2 style='color:#008000;'>✅ Low Turnover Risk</h2>"
                    "</div>", unsafe_allow_html=True)

    st.markdown("### Probability Score")
    st.progress(prob)
    st.metric("Turnover Probability", f"{prob:.2f}")
