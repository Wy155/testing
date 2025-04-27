# --- Import Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# --- Load Dataset ---
data = pd.read_csv("credit_risk_dataset.csv")
X_full = data.drop(columns=["loan_status"])
y_full = data["loan_status"]

# --- Load Models ---
st.sidebar.title("Model Selection")
model_option = st.sidebar.selectbox("Choose Model", ["Random Forest", "Gradient Boosting", "XGBoost", "Naive Bayes"])

if model_option == "Random Forest":
    model = load("random_forest_model.joblib")
elif model_option == "Gradient Boosting":
    model = load("gradient_boosting_model.joblib")
elif model_option == "XGBoost":
    model = load("xgb_classifier_model.joblib")
elif model_option == "Naive Bayes":
    model = load("gaussian_nb_model.joblib")

# --- Streamlit App Title ---
st.title("🏦 Credit Risk Prediction Dashboard")

# --- Sidebar - Select a sample from dataset ---
st.sidebar.header("📝 Select an Applicant")
sample_index = st.sidebar.slider("Select Sample Index", min_value=0, max_value=len(X_full)-1, value=0)

# --- Prepare input ---
input_data = X_full.iloc[[sample_index]]  # keep as DataFrame

# --- Handle Naive Bayes differently ---
if model_option == "Naive Bayes":
    # Naive Bayes trained only on these features
    naive_bayes_features = [
        'person_age',
        'person_income',
        'person_emp_length',
        'loan_amnt',
        'loan_int_rate',
        'loan_percent_income',
        'cb_person_cred_hist_length'
    ]
    input_data = input_data[naive_bayes_features]
    probability = model.predict_proba(input_data.values)  # Naive Bayes needs numpy array
else:
    probability = model.predict_proba(input_data)  # GBC, XGB, RF accept DataFrame

prediction = (probability[:, 1] >= 0.5).astype(int)  # Default threshold 0.5

# --- Display Prediction Result ---
st.subheader("🔮 Prediction Result")
if prediction[0] == 0:
    st.success("✅ Prediction: Low Risk Applicant")
else:
    st.error("⚠️ Prediction: High Risk Applicant")

st.write(f"Low Risk Probability: **{probability[0][0]*100:.2f}%**")
st.write(f"High Risk Probability: **{probability[0][1]*100:.2f}%**")

# --- Real label from dataset (only for performance measurement demo) ---
true_label = y_full.iloc[sample_index]

# --- Metrics calculation ---
y_test_simulated = np.array([true_label])
y_pred_simulated = prediction

accuracy = accuracy_score(y_test_simulated, y_pred_simulated)
precision = precision_score(y_test_simulated, y_pred_simulated, zero_division=0)
recall = recall_score(y_test_simulated, y_pred_simulated, zero_division=0)
f1 = f1_score(y_test_simulated, y_pred_simulated, zero_division=0)
roc_auc = roc_auc_score(y_test_simulated, probability[:, 1])

# --- Show Model Metrics ---
st.subheader(f"📊 {model_option} Model Performance (1 Applicant Evaluation)")
st.table(pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
    'Score': [f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", f"{roc_auc:.4f}"]
}))

# --- Confusion Matrix ---
st.subheader("🧩 Confusion Matrix")
cm = confusion_matrix(y_test_simulated, y_pred_simulated)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Low Risk", "High Risk"], yticklabels=["Low Risk", "High Risk"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig_cm)

# --- ROC Curve ---
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
