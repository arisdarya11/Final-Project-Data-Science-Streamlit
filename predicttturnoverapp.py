import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from PIL import Image

# ----------------- LOAD MODEL SAFE PATH -----------------
model_path = "./model.pkl"
model = joblib.load(model_path)

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Employee Turnover Prediction",
    page_icon="📊",
    layout="wide"
)

# ----------------- DARK MODE TOGGLE -----------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

col_dark, _ = st.columns([1, 8])
with col_dark:
    if st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode):
        st.session_state.dark_mode = True
    else:
        st.session_state.dark_mode = False

# Apply CSS Theme
if st.session_state.dark_mode:
    st.markdown("""
        <style>
        body, .stApp {
            background-color: #0d1117 !important;
            color: #ffffff !important;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body, .stApp {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        </style>
    """, unsafe_allow_html=True)


# ----------------- TITLE -----------------
st.markdown("<h1 style='text-align:center;'>🔎 Employee Turnover Prediction Dashboard</h1>",
            unsafe_allow_html=True)

st.write("Isi data berikut untuk memprediksi apakah karyawan akan resign atau tidak.")

# ----------------- INPUT FORM -----------------
with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.3)
        number_project = st.slider("Number of Projects", 1, 10, 3)
        work_accident = st.selectbox("Work Accident", [0, 1])

    with col2:
        last_evaluation = st.slider("Last Evaluation Score", 0.0, 1.0, 0.7)
        average_montly_hours = st.number_input("Average Monthly Hours", 50, 350, 160)
        promotion_last_5years = st.selectbox("Promotion in Last 5 Years", [0, 1])

    with col3:
        time_spend_company = st.slider("Years at Company", 1, 15, 3)
        salary = st.selectbox("Salary Level", ["low", "medium", "high"])

salary_map = {"low": 0, "medium": 1, "high": 2}

# ----------------- PREDICTION -----------------

st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("📊 Prediction Result")

if st.button("Predict Turnover"):
    
    # Show header image
    image = Image.open("/mnt/data/turnover-adalah.jpg")
    st.image(image, caption="Employee Turnover Illustration", use_column_width=True)

    input_data = pd.DataFrame({
        "satisfaction_level": [satisfaction_level],
        "last_evaluation": [last_evaluation],
        "number_project": [number_project],
        "average_montly_hours": [average_montly_hours],
        "time_spend_company": [time_spend_company],
        "Work_accident": [work_accident],
        "promotion_last_5years": [promotion_last_5years],
        "salary": [salary_map[salary]]
    })

    prediction = model.predict(input_data)[0]
    pred_proba = model.predict_proba(input_data)[0][1]

    # ----------------- CARD STYLE -----------------
    st.markdown("""
        <style>
            .result-card {
                padding: 25px;
                border-radius: 20px;
                background-color: #161b22;
                color: white;
                box-shadow: 0px 0px 20px rgba(0,0,0,0.45);
                text-align: center;
                transition: 0.3s ease;
            }
            .result-card:hover {
                transform: scale(1.02);
            }
        </style>
    """, unsafe_allow_html=True)

    label = "❌ High Risk — Employee Likely to Leave" if prediction == 1 else "✅ Low Risk — Employee Likely to Stay"
    color = "red" if prediction == 1 else "lightgreen"

    st.markdown(
        f'<div class="result-card"><h2 style="color:{color};">{label}</h2></div>',
        unsafe_allow_html=True
    )

    # ----------------- TWO COLUMN VISUAL -----------------
    left, right = st.columns([1.3, 1])

    with left:
        # GAUGE METER
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(pred_proba * 100, 2),
            title={'text': "Turnover Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if pred_proba > 0.5 else "green"},
                'steps': [
                    {'range': [0, 50], 'color': "#2ecc71"},
                    {'range': [50, 100], 'color': "#e74c3c"}
                ],
            }
        ))

        st.plotly_chart(gauge, use_container_width=True)

    with right:
        # Insight Box
        if prediction == 1:
            st.error("""
            ### ⚠️ Risk Analysis  
            - Probabilitas resign **tinggi**  
            - Perhatikan **workload, satisfaction, dan kompensasi**
            - Rekomendasi: lakukan **engagement check** & **performance feedback**
            """)
        else:
            st.success("""
            ### 🟢 Stability Insight  
            - Karyawan cenderung **bertahan**
            - Tidak ditemukan indikator risiko signifikan  
            - Pertahankan iklim kerja & kompensasi saat ini  
            """)

    # ----------------- EXTRA DETAILS BOX -----------------
    st.info(f"**Turnover Probability Score: {pred_proba:.2f}**")

