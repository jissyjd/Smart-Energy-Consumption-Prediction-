from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load


@st.cache_resource
def load_energy_model():
    model_path = Path(__file__).resolve().parent / "energy_model.joblib"
    if not model_path.exists():
        st.error("The trained model file was not found. Please make sure energy_model.joblib is in the project folder.")
        st.stop()
    return load(model_path)


st.set_page_config(page_title="Smart Energy Predictor", page_icon="⚡", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f8fbff 0%, #eef6ff 100%);
    }
    .main .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        color: #475569;
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    .info-card {
        background: white;
        border: 1px solid #dbeafe;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }
    .section-label {
        color: #2563eb;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    [data-testid="stSidebar"] {
        background: #0f172a;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }
    .stButton > button {
        border-radius: 999px;
        background: linear-gradient(90deg, #2563eb, #4f46e5);
        color: white;
        border: none;
        padding: 0.65rem 1rem;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #1d4ed8, #4338ca);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

model = load_energy_model()

st.markdown('<div class="hero-title">⚡ Smart Energy Consumption Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Estimate household electricity use with a clean, guided interface.</div>',
    unsafe_allow_html=True,
)

info_col1, info_col2 = st.columns(2)
with info_col1:
    st.markdown(
        '<div class="info-card"><div class="section-label">Quick insight</div>Fill in a few household details to estimate daily energy consumption in kWh.</div>',
        unsafe_allow_html=True,
    )
with info_col2:
    st.markdown(
        '<div class="info-card"><div class="section-label">How it works</div>This tool uses a trained machine learning model to provide a practical estimate based on your inputs.</div>',
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.header("About")
    st.write("Enter a few household and usage details to estimate expected energy consumption in kWh.")
    st.info("The estimate is designed for quick planning and awareness.")

with st.form("energy_prediction_form"):
    st.markdown('<div class="section-label">Enter your inputs</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        household_size = st.number_input("Household Size", min_value=1, value=3, step=1)
        average_temperature = st.number_input("Average Temperature (°C)", value=28.0, step=0.5)
        peak_hours_usage = st.number_input("Peak Hours Usage (kWh)", min_value=0.0, value=6.0, step=0.5)
        has_ac = st.selectbox("Air Conditioner", ["No", "Yes"])

    with col2:
        year = st.number_input("Year", min_value=2000, max_value=2100, value=2026, step=1)
        month = st.slider("Month", 1, 12, 6)
        day = st.slider("Day", 1, 31, 15)
        day_of_week = st.slider("Day of Week", 0, 6, 2)

    submitted = st.form_submit_button("Predict Energy Consumption", use_container_width=True)

if submitted:
    has_ac_value = 1 if has_ac == "Yes" else 0

    sample = pd.DataFrame(
        {
            "Household_Size": [household_size],
            "Avg_Temperature_C": [average_temperature],
            "Peak_Hours_Usage_kWh": [peak_hours_usage],
            "Has_AC": [has_ac_value],
            "Year": [year],
            "Month": [month],
            "Day": [day],
            "DayOfWeek": [day_of_week],
        }
    )

    feature_order = list(getattr(model, "feature_names_in_", [
        "Household_Size",
        "Avg_Temperature_C",
        "Has_AC",
        "Peak_Hours_Usage_kWh",
        "Year",
        "Month",
        "Day",
        "DayOfWeek",
    ]))
    sample = sample.reindex(columns=feature_order)

    prediction = model.predict(sample)[0]

    st.markdown(
        '<div class="info-card"><div class="section-label">Prediction result</div></div>',
        unsafe_allow_html=True,
    )
    st.success("Prediction complete")
    st.metric("Estimated Energy Consumption", f"{prediction:.2f} kWh")

    if prediction > 20:
        st.caption("This appears to be a relatively high estimate. Double-check your inputs for peak usage and temperature.")
    elif prediction < 8:
        st.caption("This is a lower-than-average estimate. You may want to review the peak-hour usage value.")
    else:
        st.caption("This estimate falls within a typical range for many households.")
