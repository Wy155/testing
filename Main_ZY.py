# --- Import Libraries ---
import streamlit as st
import pandas as pd
from joblib import load

# --- Load Dataset ---
data = pd.read_csv("credit_risk_dataset.csv")

# Drop label column
X = data.drop(columns=["loan_status"])

# --- Load Model ---
st.sidebar.title("Model Selection")
model_option = st.sidebar.selectbox("Choose Model", ["Gradient Boosting", "XGBoost", "Naive Bayes"])

if model_option == "Gradient Boosting":
    model = load("gradient_boosting_model.joblib")
elif model_option == "XGBoost":
    model = load("xgb_classifier_model.joblib")
elif model_option == "Naive Bayes":
    model = load("gaussian_nb_model.joblib")

# --- Select a sample from dataset ---
sample_index = st.sidebar.slider("Select Sample Index", min_value=0, max_value=len(X)-1, value=0)
input_data = X.iloc[[sample_index]]  # Select one row

# --- Predict
if model_option == "Naive Bayes":
    probability = model.predict_proba(input_data.values)  # Naive Bayes needs array
else:
    probability = model.predict_proba(input_data)          # GBC, XGBoost accept DataFrame

prediction = (probability[:, 1] >= 0.5).astype(int)

# --- Output
st.write("🔮 Prediction Result:")
if prediction[0] == 0:
    st.success("✅ Low Risk Applicant")
else:
    st.error("⚠️ High Risk Applicant")

st.write(f"Low Risk Probability: {probability[0][0]*100:.2f}%")
st.write(f"High Risk Probability: {probability[0][1]*100:.2f}%")
