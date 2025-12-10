# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import io
import base64
import datetime

# plotting
import plotly.express as px
import plotly.graph_objects as go

# Optional imports
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

# ---------------------------
# Utilities
# ---------------------------
def load_file(filename):
    try:
        return joblib.load(filename)
    except Exception:
        with open(filename, "rb") as f:
            return pickle.load(f)

def to_pdf_bytes(summary_text, filename="prediction_report.pdf"):
    """
    Generate PDF bytes with reportlab if available, otherwise return bytes of plain text.
    """
    if HAS_REPORTLAB:
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=letter)
        width, height = letter
        margin = 40
        y = height - margin
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y, "Employee Turnover Prediction Report")
        y -= 30
        c.setFont("Helvetica", 10)
        for line in summary_text.splitlines():
            if y < margin:
                c.showPage()
                y = height - margin
                c.setFont("Helvetica", 10)
            c.drawString(margin, y, line)
            y -= 14
        c.showPage()
        c.save()
        buf.seek(0)
        return buf.read()
    else:
        # fallback: return plain text bytes as .txt inside bytes
        return summary_text.encode("utf-8")

# ---------------------------
# Load models & encoder & scaler
# ---------------------------
@st.cache_data(show_spinner=False)
def load_artifacts():
    model = load_file("xgb_attrition_model.pkl")
    scaler = load_file("scaler.pkl")
    encoder = load_file("encoder.pkl")
    return model, scaler, encoder

try:
    model, scaler, encoder = load_artifacts()
except Exception as e:
    st.error(f"Failed to load model artifacts: {e}")
    st.stop()

# ---------------------------
# App config & CSS (dark mode toggle)
# ---------------------------
st.set_page_config(page_title="Employee Turnover Prediction", page_icon="🧑‍💼", layout="wide")

st.sidebar.title("Settings")
dark_mode = st.sidebar.checkbox("🌙 Dark mode", value=False)

if dark_mode:
    st.markdown(
        """
        <style>
            .reportview-container, .sidebar .sidebar-content {
                background: linear-gradient(180deg, #0f1724 0%, #0b1220 100%);
                color: #e6eef8;
            }
            .stButton>button { background-color: #1f2937; color: #fff; }
            .metric { color: #fff !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
            .reportview-container, .sidebar .sidebar-content {
                background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
                color: #0b1220;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# small helper for safe encoder transform
def safe_encode_salary(enc, salary_val):
    """
    Try to transform salary. If encoder returns array-like, attempt to compress to single scalar if model expects scalar.
    Otherwise return first element / sum as fallback.
    """
    try:
        res = enc.transform([salary_val])
        # if res is 2d array
        arr = np.array(res)
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr[0,0]
        # if multiple columns, return flattened joined string? for safety return arr.ravel()
        # Many older pipelines used LabelEncoder-like output; try to pick first element
        return arr.ravel()
    except Exception:
        # if encoder cannot transform, try mapping
        mapping = {"low":0, "medium":1, "high":2}
        return mapping.get(salary_val, 0)

# ---------------------------
# UI Header
# ---------------------------
st.title("🔍 Employee Turnover Prediction — Advanced Dashboard")
st.markdown("Interactive predictive tool with model explainability, importance charts, PDF export, and modern visuals.")

st.markdown("---")

# ---------------------------
# Input form (two columns)
# ---------------------------
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.50, step=0.01)
        last_evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.60, step=0.01)
        number_project = st.number_input("Number of Projects", min_value=1, max_value=20, value=3)
        average_montly_hours = st.number_input("Average Monthly Hours", min_value=10, max_value=400, value=160)
    with col2:
        time_spend_company = st.number_input("Years at Company", min_value=0, max_value=40, value=3)
        work_accident_choice = st.selectbox("Work Accident", ["No", "Yes"])
        promotion_choice = st.selectbox("Promotion in Last 5 Years", ["No", "Yes"])
        salary = st.selectbox("Salary Level", ["low", "medium", "high"])
    submitted = st.form_submit_button("🔎 Predict & Explain")

# Convert binary choices
work_accident = 1 if work_accident_choice == "Yes" else 0
promotion_last_5years = 1 if promotion_choice == "Yes" else 0

# Prepare salary encoding for model input
enc_sal = safe_encode_salary(encoder, salary)

# Determine model input shape: if enc_sal is array-like (multiple features), need to handle.
# The training pipeline that saved the model likely expected salary as either a single encoded column or multiple.
# We will attempt to build input as earlier used: put encoded_salary directly in column "salary".
# If encoder returns vector, we'll try to reduce to single value using mean (fallback).
if isinstance(enc_sal, np.ndarray):
    # reduce to single scalar feature by taking mean - fallback
    try:
        encoded_salary_value = float(np.mean(enc_sal))
    except Exception:
        encoded_salary_value = float(enc_sal.ravel()[0])
else:
    try:
        encoded_salary_value = float(enc_sal)
    except Exception:
        encoded_salary_value = 0.0

# Build DataFrame input according to the model_columns order
input_df = pd.DataFrame([[
    satisfaction_level,
    last_evaluation,
    number_project,
    average_montly_hours,
    time_spend_company,
    encoded_salary_value,
    work_accident,
    promotion_last_5years
]], columns=model_columns)

# Scale input
try:
    scaled_input = scaler.transform(input_df)
except Exception as e:
    st.error(f"Error when applying scaler: {e}")
    st.stop()

# If form submitted -> predict and show visuals
if submitted:
    # Predict
    try:
        pred = model.predict(scaled_input)[0]
        proba = float(model.predict_proba(scaled_input)[0][1])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Modern result card
    if pred == 1:
        title = "High Attrition Risk"
        color = "#ff4d4f"
        badge = "⚠️"
    else:
        title = "Low Attrition Risk"
        color = "#2ecc71"
        badge = "✅"

    # Result Card (HTML)
    st.markdown(
        f"""
        <div style="border-radius:12px;padding:18px;border:1px solid rgba(0,0,0,0.06);background:linear-gradient(90deg, rgba(255,255,255,0.9), rgba(245,247,250,0.9));">
            <h3 style="color:{color};margin:6px 0 4px 0">{badge} {title}</h3>
            <p style="margin:4px 0 12px 0;color:#555">Model probability score for attrition</p>
            <div style="background:#eee;border-radius:8px;padding:8px;">
                <div style="width:100%;background:#e6e6e6;border-radius:8px;height:14px;overflow:hidden">
                    <div style="width:{proba*100}%;height:14px;background:{color};transition:width 0.7s;"></div>
                </div>
                <div style="display:flex;justify-content:space-between;margin-top:8px">
                    <strong style="color:{color};font-size:18px">{proba:.2f}</strong>
                    <small style="color:#666">Interpretation: closer to 1 = higher turnover risk</small>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")  # spacing

    # Metric cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Satisfaction", f"{satisfaction_level:.2f}")
    m2.metric("Last Evaluation", f"{last_evaluation:.2f}")
    m3.metric("Projects", f"{number_project}")
    m4.metric("Years at Company", f"{time_spend_company}")

    # Feature bar chart (input profile)
    profile = input_df.T.reset_index()
    profile.columns = ["feature", "value"]
    # For display, convert arrays to scalars if any
    profile["display"] = profile["value"].apply(lambda x: float(np.array(x).ravel()[0]) if isinstance(x, (list, np.ndarray)) else float(x))

    fig_profile = px.bar(profile, x="feature", y="display", title="Employee Input Profile", text="display")
    st.plotly_chart(fig_profile, use_container_width=True)

    # Feature importance bar chart from model or SHAP mean abs
    st.subheader("📊 Feature Importance")
    importance_vals = None
    importance_index = None

    # Try model.feature_importances_
    try:
        if hasattr(model, "feature_importances_"):
            importance_vals = model.feature_importances_
            importance_index = model_columns
    except Exception:
        importance_vals = None

    # If SHAP is available, compute shap values and use mean absolute for importance
    shap_values = None
    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(scaled_input)  # for classification returns list or array
            # shap_values might be array-like; take absolute mean across samples
            if isinstance(shap_values, list):
                # for binary classification shap_values[1] corresponds to class 1
                sv = np.abs(shap_values[1]).mean(axis=0)
            else:
                sv = np.abs(shap_values).mean(axis=0)
            importance_vals = sv
            importance_index = model_columns
        except Exception:
            shap_values = None

    if importance_vals is not None:
        imp_df = pd.DataFrame({
            "feature": importance_index,
            "importance": np.array(importance_vals).ravel()
        })
        imp_df = imp_df.sort_values("importance", ascending=False)
        fig_imp = px.bar(imp_df, x="importance", y="feature", orientation="h", title="Feature Importance", text="importance")
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importance not available (model does not expose feature_importances_ and SHAP not available).")

    # SHAP explainability section (force/summary plots) if available
    if HAS_SHAP and shap_values is not None:
        st.subheader("🔬 SHAP Explainability")
        try:
            # SHAP summary (requires matplotlib); convert to plotly by saving to image? Instead show bar of shap for this instance
            # Create a small table of shap values per feature for this instance
            if isinstance(shap_values, list):
                # class 1 as positive class
                sv_inst = shap_values[1][0]
            else:
                sv_inst = shap_values[0]
            shap_df_inst = pd.DataFrame({
                "feature": model_columns,
                "shap_value": sv_inst
            }).sort_values("shap_value", key=lambda s: np.abs(s), ascending=False)
            fig_shap = px.bar(shap_df_inst, x="shap_value", y="feature", orientation="h",
                              title="SHAP values for this prediction", text="shap_value",
                              color="shap_value", color_continuous_scale="RdBu")
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception as e:
            st.warning(f"SHAP visualization failed: {e}")

    elif not HAS_SHAP:
        st.info("Install `shap` package to enable model explainability (SHAP).")

    # Risk Radar Chart
    st.subheader("📡 Risk Radar Chart (Profile Radar)")
    try:
        radar_features = ["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company"]
        radar_vals = [input_df.loc[0, f] if f in input_df.columns else 0 for f in radar_features]
        # normalize numeric values to 0-1 for radar (use simple min-max assumptions)
        norm_vals = []
        for f, v in zip(radar_features, radar_vals):
            if f == "satisfaction_level" or f == "last_evaluation":
                norm_vals.append(float(v))
            elif f == "number_project":
                norm_vals.append(float(v) / 10.0)  # assume max 10
            elif f == "average_montly_hours":
                norm_vals.append(float(v) / 400.0)  # assume max 400
            elif f == "time_spend_company":
                norm_vals.append(float(v) / 40.0)  # assume max 40
            else:
                norm_vals.append(0.0)

        radar_vals_plot = norm_vals + [norm_vals[0]]  # close loop
        radar_cat = radar_features + [radar_features[0]]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_vals_plot,
            theta=radar_cat,
            fill='toself',
            name='Employee Profile',
            marker=dict(color=color)
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=False, title="Risk Radar (normalized)")
        st.plotly_chart(fig_radar, use_container_width=True)
    except Exception as e:
        st.warning(f"Radar chart failed: {e}")

    # ---------------------------
    # Generate downloadable PDF / TXT report
    # ---------------------------
    st.subheader("📥 Download Prediction Report")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_lines = [
        "Employee Turnover Prediction Report",
        f"Generated: {datetime.datetime.now().isoformat()}",
        "",
        f"Prediction: {'Leave' if pred==1 else 'Stay'}",
        f"Probability: {proba:.4f}",
        "",
        "Input features:",
    ]
    for col in input_df.columns:
        report_lines.append(f"- {col}: {input_df.loc[0, col]}")
    report_text = "\n".join(report_lines)
    pdf_bytes = to_pdf_bytes(report_text, filename=f"turnover_report_{timestamp}.pdf")

    download_label = "Download PDF Report" if HAS_REPORTLAB else "Download TXT Report (reportlab not installed)"
    st.download_button(label=download_label, data=pdf_bytes,
                       file_name=f"turnover_report_{timestamp}.{'pdf' if HAS_REPORTLAB else 'txt'}",
                       mime="application/pdf" if HAS_REPORTLAB else "text/plain")

    st.success("Finished. You can download the report and inspect the explainability charts above.")

# End of app
