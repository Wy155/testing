import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Load your saved model
@st.cache_resource
def load_model():
    model = joblib.load("AI.joblib")
    return model

# Load the trained model
model = load_model()

# (Optional) Load scaler if you have one — for now assume model expects already scaled input
# scaler = joblib.load("/mnt/data/scaler.joblib")  # Uncomment if you have scaler

# Title
st.title("🏦 Credit Risk Prediction Dashboard (Pretrained Model)")

st.sidebar.header("📝 Input Features")

# Feature inputs
person_age = st.sidebar.number_input("Person Age", min_value=0, max_value=100, value=30)
person_income = st.sidebar.number_input("Person Income", min_value=0, value=50000)
person_emp_length = st.sidebar.number_input("Person Employment Length (years)", min_value=0, max_value=50, value=5)
loan_amnt = st.sidebar.number_input("Loan Amount", min_value=0, value=10000)
loan_int_rate = st.sidebar.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=100.0, value=5.0)
loan_percent_income = st.sidebar.number_input("Loan Percent of Income (%)", min_value=0.0, max_value=100.0, value=10.0)
cb_person_cred_hist_length = st.sidebar.number_input("Credit History Length (years)", min_value=0, max_value=50, value=10)

# Combine input features
input_data = pd.DataFrame({
    'person_age': [person_age],
    'person_income': [person_income],
    'person_emp_length': [person_emp_length],
    'loan_amnt': [loan_amnt],
    'loan_int_rate': [loan_int_rate],
    'loan_percent_income': [loan_percent_income],
    'cb_person_cred_hist_length': [cb_person_cred_hist_length],
})

# (Optional) If your model expects scaled input, then apply scaler:
# input_data_scaled = scaler.transform(input_data)
# prediction = model.predict(input_data_scaled)
# If not scaled:
prediction = model.predict(input_data)

# Show prediction
st.subheader("🔮 Prediction Result")
if prediction[0] == 0:
    st.success("✅ **Prediction: Low Risk**")
else:
    st.error("⚠️ **Prediction: High Risk**")

# Since you only uploaded the model (not test data),
# we cannot plot confusion matrix here unless you upload test set also
st.info("Note: Confusion matrix and model evaluation require test data which is not uploaded yet.")
