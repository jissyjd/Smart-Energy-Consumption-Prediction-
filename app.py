import os

print("Current folder:", os.getcwd())
print("Files:", os.listdir())

import streamlit as st
import pandas as pd
from joblib import load

model = load("energy_model.joblib")

st.title("Smart Energy Consumption Prediction")

household_size = st.number_input("Household Size", min_value=1)
temperature = st.number_input("Average Temperature")
peak_usage = st.number_input("Peak Hours Usage")
has_ac = st.selectbox("Has AC", ["No", "Yes"])

year = st.number_input("Year", value=2024)
month = st.slider("Month", 1, 12)
day = st.slider("Day", 1, 31)
dayofweek = st.slider("Day Of Week", 0, 6)

has_ac = 1 if has_ac == "Yes" else 0

if st.button("Predict"):

    sample = pd.DataFrame({
        'Household_Size':[household_size],
        'Avg_Temperature_C':[temperature],
        'Peak_Hours_Usage_kWh':[peak_usage],
        'Has_AC':[has_ac],
        'Year':[year],
        'Month':[month],
        'Day':[day],
        'DayOfWeek':[dayofweek]
    })

    prediction = model.predict(sample)

    st.success(
        f"Predicted Energy Consumption: {prediction[0]:.2f} kWh"
    )
