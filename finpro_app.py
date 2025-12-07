import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# ======================================================
# CONFIG
# ======================================================
MODEL_FILE = "xgb_attrition_model.pkl"
ENCODER_FILE = "le_salary.pkl"
SCALER_FILE = "scaler.pkl"

model = None
le_salary = None
scaler = None


# ======================================================
# LOAD MODEL, ENCODER, SCALER (REAL / SIMULATION)
# ======================================================
try:
    model = joblib.load(MODEL_FILE)
    le_salary = joblib.load(ENCODER_FILE)
    scaler = joblib.load(SCALER_FILE)
    st.info("✅ Model, Encoder, dan Scaler berhasil dimuat.")

except FileNotFoundError:
    st.warning("⚠️ File .pkl tidak ditemukan. Membuat model + encoder + scaler simulasi.")

    try:
        # Dummy dataset untuk simulasi
        dummy_X = pd.DataFrame({
            "satisfaction_level": np.random.rand(100),
            "last_evaluation": np.random.rand(100),
            "number_project": np.random.randint(2, 7, 100),
            "average_montly_hours": np.random.randint(100, 300, 100),
            "time_spend_company": np.random.randint(1, 6, 100),
            "Work_accident": np.random.randint(0, 2, 100),
            "promotion_last_5years": np.random.randint(0, 2, 100),
            "salary": np.random.randint(0, 3, 100)
        })
        dummy_y = np.random.randint(0, 2, 100)

        # Simulasi model XGBoost
        model = XGBClassifier(random_state=42)
        model.fit(dummy_X, dummy_y)

        # Simulasi LabelEncoder
        le_salary = LabelEncoder()
        le_salary.fit(["low", "medium", "high"])

        # Simulasi Scaler
        scaler = StandardScaler()
        scaler.fit(dummy_X)

        st.warning("⚠️ Model, encoder, dan scaler yang digunakan adalah SIMULASI.")

    except Exception as e:
        st.error(f"❌ Fatal error saat membuat simulasi model: {e}")
        st.stop()


# ======================================================
# STREAMLIT UI
# ======================================================
st.title("Employee Attrition Prediction App")
st.write("Aplikasi prediksi apakah karyawan berpotensi keluar (**attrition**) berdasarkan data HR.")

satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5, 0.01)
last_evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.6, 0.01)
number_project = st.number_input("Number of Projects", 1, 10, 3)
average_montly_hours = st.number_input("Average Monthly Hours", 50, 350, 150)
time_spend_company = st.number_input("Years in Company", 1, 20, 3)

salary = st.selectbox("Salary Level", ["low", "medium", "high"])
Work_accident = st.selectbox("Work Accident", [0, 1], format_func=lambda x: "Ya (1)" if x else "Tidak (0)")
promotion_last_5years = st.selectbox("Promotion in Last 5 Years", [0, 1], format_func=lambda x: "Ya (1)" if x else "Tidak (0)")


# ======================================================
# PREDICTION BUTTON
# ======================================================
if st.button("Prediksi Attrition"):

    # Validate essential objects
    if model is None or le_salary is None or scaler is None:
        st.error("❌ Model/encoder/scaler belum dimuat dengan benar.")
        st.stop()

    # 1. Build input DataFrame
    input_df = pd.DataFrame({
        "satisfaction_level": [satisfaction_level],
        "last_evaluation": [last_evaluation],
        "number_project": [number_project],
        "average_montly_hours": [average_montly_hours],
        "time_spend_company": [time_spend_company],
        "salary": [salary],
        "Work_accident": [Work_accident],
        "promotion_last_5years": [promotion_last_5years]
    })

    # 2. Encode salary
    try:
        input_df["salary"] = le_salary.transform(input_df["salary"])
    except Exception as e:
        st.error(f"⚠️ Error saat encode salary: {e}")
        st.stop()

    # 3. Ensure column order matches training phase
    expected_cols = [
        'satisfaction_level', 'last_evaluation', 'number_project',
        'average_montly_hours', 'time_spend_company', 'Work_accident',
        'promotion_last_5years', 'salary'
    ]
    input_df = input_df[expected_cols]

    # 4. Scaling
    try:
        X_scaled = scaler.transform(input_df)
    except Exception as e:
        st.error(f"⚠️ Error saat scaling: {e}")
        st.stop()

    # 5. Predict
    try:
        proba = model.predict_proba(X_scaled)[0][1]
        pred = model.predict(X_scaled)[0]

        st.subheader("📊 Hasil Prediksi")
        st.write(f"Probabilitas Attrition: **{proba:.2f}**")

        if pred == 1:
            st.error("🚨 Karyawan **berpotensi keluar** (Attrited)")
        else:
            st.success("👍 Karyawan **tidak berpotensi keluar** (Stays)")

    except Exception as e:
        st.error(f"⚠️ Error saat prediksi: {e}")
