import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------
# Load model and scaler ONLY
# -------------------------------
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -------------------------------
# FIXED CATEGORY MAPPING
# -------------------------------
type_mapping = {
    "charge": 0,
    "discharge": 1,
    "impedance": 2
}

# -------------------------------
# Prediction function
# -------------------------------
def predict_temperature(type_discharge, capacity, re, rct):
    type_encoded = type_mapping[type_discharge]

    X_input = np.array([[type_encoded, capacity, re, rct]])
    X_input_scaled = scaler.transform(X_input)

    prediction = model.predict(X_input_scaled)
    return prediction[0]

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Battery Temperature Prediction", layout="centered")

st.title("🔋 Battery Ambient Temperature Prediction")
st.markdown("**Random Forest Regression Model**")

type_discharge = st.selectbox(
    "Battery Operation Type",
    ["charge", "discharge", "impedance"]
)

capacity = st.number_input("Capacity", min_value=0.0, value=1.5)
re = st.number_input("Re (Internal Resistance)")
rct = st.number_input("Rct (Charge Transfer Resistance)")

if st.button("Predict Temperature"):
    result = predict_temperature(type_discharge, capacity, re, rct)
    st.success(f"🌡 Predicted Ambient Temperature: **{result:.2f} °C**")

# -------------------------------
# Feature Importance Plot
# -------------------------------
st.markdown("---")
st.subheader("Feature Importance")

features = ["type", "Capacity", "Re", "Rct"]
importances = model.feature_importances_

fig, ax = plt.subplots()
ax.barh(features, importances)
ax.set_xlabel("Importance")
ax.set_title("Random Forest Feature Importance")
ax.invert_yaxis()
st.pyplot(fig)
