import streamlit as st
import pandas as pd
import pickle
import joblib

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Employee Turnover Prediction",
    page_icon="📊",
    layout="centered"
)

# ================= LOAD =================
def load_file(filename):
    try:
        return joblib.load(filename)
    except:
        with open(filename, "rb") as f:
            return pickle.load(f)

model = load_file("xgb_attrition_model.pkl")
scaler = load_file("scaler.pkl")
encoder = load_file("encoder.pkl")

# ===== FIXED COLUMN ORDER (SESUI TRAINING) =====
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

# ================= HEADER =================
st.markdown("""
<h1 style='text-align:center;'>📊 Employee Turnover Prediction</h1>
<p style='text-align:center; color:gray;'>
Modern Machine Learning App for Employee Turnover Analysis powered by XGBoost 
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ================= INPUT FORM (CARD STYLE) =================
with st.container():
    st.markdown("### 🧑‍💼 Employee Profile")

    c1, c2 = st.columns(2)

    with c1:
        satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
        last_evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.5)
        number_project = st.number_input("Number of Projects", 1, 10, 3)
        average_montly_hours = st.number_input("Avg Monthly Hours", 50, 350, 160)

    with c2:
        time_spend_company = st.number_input("Years at Company", 1, 20, 3)
        work_accident = st.selectbox("Work Accident", [0, 1])
        promotion_last_5years = st.selectbox("Promotion in Last 5 Years", [0, 1])
        salary = st.selectbox("Salary Level", ["low", "medium", "high"])

st.markdown("---")

# ================= BUILD INPUT =================
input_data = pd.DataFrame([[ 
    satisfaction_level,
    last_evaluation,
    number_project,
    average_montly_hours,
    time_spend_company,
    encoder.transform([salary])[0],
    work_accident,
    promotion_last_5years
]], columns=model_columns)

# ================= SCALING =================
scaled_input = scaler.transform(input_data)

# ================= PREDICT BUTTON =================
if st.button("🚀 Predict Turnover", use_container_width=True):

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.markdown("## 📈 Prediction Result")

    # Visual Card Result
    if prediction == 1:
        st.markdown(f"""
        <div style="
            background-color:#FFEBEE;
            padding:20px;
            border-radius:15px;
            text-align:center;
            border:2px solid #FFCDD2;
        ">
            <h2 style="color:#C62828;">⚠ High Risk of Turnover</h2>
            <p style="font-size:18px;">Probability: <b>{probability:.2f}</b></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="
            background-color:#E8F5E9;
            padding:20px;
            border-radius:15px;
            text-align:center;
            border:2px solid #C8E6C9;
        ">
            <h2 style="color:#2E7D32;">✅ Low Risk of Turnover</h2>
            <p style="font-size:18px;">Probability: <b>{probability:.2f}</b></p>
        </div>
        """, unsafe_allow_html=True)

    # Progress bar
    st.markdown("### 🔍 Probability Visualization")
    st.progress(float(probability))


st.markdown("---")

# ================= FOOTER =================
st.markdown("""
<div style='text-align:center;color:gray;'>
Built with ❤️ using Streamlit & Machine Learning
</div>
""", unsafe_allow_html=True)


