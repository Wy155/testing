# --- Import Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from joblib import load

# --- Load Models ---
st.sidebar.title("Model Selection")
model_option = st.sidebar.selectbox("Choose Model", ["Gradient Boosting", "XGBoost", "Naive Bayes"])

if model_option == "Gradient Boosting":
    model = load("gradient_boosting_model.joblib")
elif model_option == "XGBoost":
    model = load("xgb_classifier_model.joblib")
elif model_option == "Naive Bayes":
    model = load("gaussian_nb_model.joblib")

# --- Streamlit App Title ---
st.title("🏦 Credit Risk Prediction Dashboard (Fully Fixed Version)")

# --- Sidebar - User Input Form ---
st.sidebar.header("📝 Applicant Information")
with st.sidebar.form(key="input_form"):
    person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
    person_income = st.number_input("Annual Income ($)", min_value=0.0, value=50000.0)
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
    person_emp_length = st.number_input("Employment Length (Years)", min_value=0, value=5)
    loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    loan_amnt = st.number_input("Loan Amount ($)", min_value=0.0, value=10000.0)
    loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=100.0, value=10.0)
    if person_income > 0:
        loan_percent_income = (loan_amnt / person_income) * 100
    else:
        loan_percent_income = 0.0
    st.number_input("Loan Percent Income (%)", value=loan_percent_income, format="%.2f", disabled=True)
    cb_person_default_on_file = st.selectbox("Default on File", ["N", "Y"])
    cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", min_value=0, value=8)

    submit_button = st.form_submit_button(label="Predict")

# --- Prepare Input and Predict ---
if submit_button:
    # Encode categorical variables
    home_ownership_mapping = {"RENT": 0, "OWN": 1, "MORTGAGE": 2}
    loan_intent_mapping = {"PERSONAL": 0, "EDUCATION": 1, "MEDICAL": 2, "VENTURE": 3, "HOMEIMPROVEMENT": 4, "DEBTCONSOLIDATION": 5}
    default_mapping = {"N": 0, "Y": 1}

    input_data = pd.DataFrame({
        'person_age': [person_age],
        'person_income': [person_income],
        'person_home_ownership': [home_ownership_mapping.get(person_home_ownership, 0)],
        'person_emp_length': [person_emp_length],
        'loan_intent': [loan_intent_mapping.get(loan_intent, 0)],
        'loan_amnt': [loan_amnt],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_default_on_file': [default_mapping.get(cb_person_default_on_file, 0)],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length],
    })

    # --- Predict
    if model_option == "Naive Bayes":
        probability = model.predict_proba(input_data.values)  # Naive Bayes needs array
    else:
        probability = model.predict_proba(input_data)         # XGBoost/GBC accept DataFrame

    prediction = (probability[:, 1] >= 0.5).astype(int)  # Default threshold 0.5

    # --- Display Prediction Result
    st.subheader("🔮 Prediction Result")
    if prediction[0] == 0:
        st.success("✅ **Prediction: Low Risk Applicant**")
    else:
        st.error("⚠️ **Prediction: High Risk Applicant**")

    st.write(f"Low Risk Probability: **{probability[0][0]*100:.2f}%**")
    st.write(f"High Risk Probability: **{probability[0][1]*100:.2f}%**")

    # --- Simulate Dummy Test for Metrics ---
    y_test_simulated = np.array([0])  # Assume "True" label is Low Risk for simulation
    y_pred_simulated = prediction

    # Metrics (based on 1 sample, for demonstration)
    accuracy = accuracy_score(y_test_simulated, y_pred_simulated)
    precision = precision_score(y_test_simulated, y_pred_simulated, zero_division=0)
    recall = recall_score(y_test_simulated, y_pred_simulated, zero_division=0)
    f1 = f1_score(y_test_simulated, y_pred_simulated, zero_division=0)
    roc_auc = roc_auc_score(y_test_simulated, probability[:, 1])

    # --- Show Model Metrics ---
    st.subheader(f"📊 {model_option} Model Performance (Demo Based on 1 Applicant)")
    st.table(pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
        'Score': [f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", f"{roc_auc:.4f}"]
    }))

    # --- Show Confusion Matrix ---
    st.subheader("🧩 Confusion Matrix")
    cm = confusion_matrix(y_test_simulated, y_pred_simulated)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Low Risk", "High Risk"], yticklabels=["Low Risk", "High Risk"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig_cm)

    # --- Show ROC Curve ---
    st.subheader("📈 ROC Curve")
    fpr, tpr, _ = roc_curve(y_test_simulated, probability[:, 1])
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, color='blue', label=f"AUC = {roc_auc:.2f}")
    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='grey')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("Receiver Operating Characteristic (ROC)")
    ax_roc.legend()
    st.pyplot(fig_roc)
