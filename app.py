import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# =========================
# LOAD FILES
# =========================
model = load_model('sepsis_final_model.keras')
scaler = joblib.load('scaler.pkl')
feature_cols = joblib.load('features.pkl')
mean_values = joblib.load('mean_values.pkl')

# FIX dtype issue
mean_values = mean_values.astype(float)

threshold = float(open('threshold.txt').read())
#threshold = 0.5

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Sepsis Prediction System", layout="wide")

st.title("🩺 Sepsis Early Detection System")
st.markdown("AI-powered ICU monitoring using LSTM")

# =========================
# MODE SELECTION
# =========================
mode = st.sidebar.selectbox("Select Mode", ["Single Prediction", "Time Series Prediction"])

# =========================
# FEATURE GROUPS
# =========================
vital_features = ['HR', 'O2Sat', 'Temp', 'SBP', 'DBP', 'Resp']
lab_features = ['WBC', 'Platelets', 'Lactate', 'Creatinine']

# =========================
# SINGLE PREDICTION UI
# =========================
if mode == "Single Prediction":

    st.header("🔍 Patient Data Input")

    col1, col2 = st.columns(2)

    input_values = {}

    # -------- VITALS --------
    with col1:
        st.subheader("🫀 Vital Signs")

        input_values['HR'] = st.slider("Heart Rate (HR)", 30, 180, 80)
        input_values['O2Sat'] = st.slider("Oxygen Saturation (O2Sat)", 50, 100, 98)
        input_values['Temp'] = st.slider("Temperature (°C)", 30.0, 42.0, 37.0)
        input_values['SBP'] = st.slider("Systolic BP", 50, 200, 120)
        input_values['DBP'] = st.slider("Diastolic BP", 30, 120, 80)
        input_values['Resp'] = st.slider("Respiration Rate", 5, 40, 18)

    # -------- LABS --------
    with col2:
        st.subheader("🧪 Lab Parameters")

        input_values['WBC'] = st.number_input("WBC", value=float(mean_values.get('WBC', 7.0)))
        input_values['Platelets'] = st.number_input("Platelets", value=float(mean_values.get('Platelets', 250.0)))
        input_values['Lactate'] = st.number_input("Lactate", value=float(mean_values.get('Lactate', 1.5)))
        input_values['Creatinine'] = st.number_input("Creatinine", value=float(mean_values.get('Creatinine', 1.0)))

    # -------- PREDICTION --------
    if st.button("🔍 Predict Sepsis Risk"):

        # Start with mean values
        input_full = mean_values.values.reshape(1, -1)

        # Replace selected features
        for feature, value in input_values.items():
            if feature in feature_cols:
                idx = feature_cols.index(feature)
                input_full[0][idx] = value

        # Scale
        input_scaled = scaler.transform(input_full)

        # Create sequence
        sequence = np.repeat(input_scaled, 6, axis=0).reshape(1, 6, -1)

        # Predict
        prob = model.predict(sequence)[0][0]

        # -------- COLOR OUTPUT --------
        if prob > 0.7:
            st.error(f"🔴 HIGH RISK: Sepsis Likely ({prob:.2f})")
        elif prob > threshold:
            st.warning(f"🟠 MODERATE RISK: Monitor Patient ({prob:.2f})")
        else:
            st.success(f"🟢 LOW RISK: No Sepsis ({prob:.2f})")

        # Progress bar
        st.progress(float(prob))

# =========================
# TIME SERIES UI
# =========================
else:
    st.header("📈 Time-Series Prediction")

    st.markdown("Upload CSV with patient time-series data (same columns as training)")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.write("Preview:", df.head())

        try:
            df = df[feature_cols]

            data_scaled = scaler.transform(df)

            if data_scaled.shape[0] < 6:
                st.error("Need at least 6 rows for prediction")
            else:
                sequence = data_scaled[-6:].reshape(1, 6, -1)

                prob = model.predict(sequence)[0][0]

                if prob > 0.7:
                    st.error(f"🔴 HIGH RISK: Sepsis Likely ({prob:.2f})")
                elif prob > threshold:
                    st.warning(f"🟠 MODERATE RISK: Monitor Patient ({prob:.2f})")
                else:
                    st.success(f"🟢 LOW RISK: No Sepsis ({prob:.2f})")

                st.line_chart(df)

        except Exception as e:
            st.error(f"Error: {e}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("Developed using LSTM for Early Sepsis Detection")

