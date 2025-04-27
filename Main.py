import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE

# --- Functions ---

@st.cache_data
def load_data():
    df = pd.read_csv("credit_risk_dataset.csv")
    return df

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return accuracy, precision, recall, f1, roc_auc, y_pred

def get_model(model_option):
    if model_option == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif model_option == "SVM":
        model = SVC(probability=True, random_state=42)
    elif model_option == "Naive Bayes":
        param_grid = {'var_smoothing': np.logspace(-9, -6, 10)}
        grid_GaussianNB = GridSearchCV(GaussianNB(priors=[0.5, 0.5]), param_grid, cv=5)
        model = grid_GaussianNB
    return model

# --- Main App ---

# Title
st.title("🏦 Credit Risk Prediction Dashboard")

# Sidebar
st.sidebar.header("🔍 Model and Input Settings")
model_option = st.sidebar.selectbox("Select Model", ["Random Forest", "SVM", "Naive Bayes"])
use_smote = st.sidebar.checkbox("Apply SMOTE to Balance Classes (Recommended for Naive Bayes)", value=True)

# Data Loading
df = load_data()
df = df.assign(
    person_emp_length=df['person_emp_length'].fillna(df['person_emp_length'].median()),
    loan_int_rate=df['loan_int_rate'].fillna(df['loan_int_rate'].median())
)

# Data Split
X = df.drop(columns=['loan_status'], axis=1)
X = X.select_dtypes(include=[np.number])
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply SMOTE if selected
if use_smote:
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

# Get Model
model = get_model(model_option)
model.fit(X_train, y_train)

# Evaluate Model
accuracy, precision, recall, f1, roc_auc, y_test_pred = evaluate_model(model, X_test, y_test)

# Input Form
st.sidebar.header("📝 Input Features")
with st.sidebar.form(key="input_form"):
    person_age = st.number_input("Person Age", min_value=0, max_value=100, value=25)
    person_income = st.number_input("Person Income", min_value=0, value=50000)
    person_emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)
    loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000)
    loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=100.0, value=5.0)
    loan_percent_income = st.number_input("Loan Percent Income (%)", min_value=0.0, max_value=100.0, value=10.0)
    cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=10)
    submit_button = st.form_submit_button(label="Predict")

# Prepare Input
input_data = pd.DataFrame({
    'person_age': [person_age],
    'person_income': [person_income],
    'person_emp_length': [person_emp_length],
    'loan_amnt': [loan_amnt],
    'loan_int_rate': [loan_int_rate],
    'loan_percent_income': [loan_percent_income],
    'cb_person_cred_hist_length': [cb_person_cred_hist_length],
})

input_data_scaled = scaler.transform(input_data)

# Prediction
if submit_button:
    prediction = model.predict(input_data_scaled)
    probability = model.predict_proba(input_data_scaled)

    st.subheader("🔮 Prediction Result")
    if prediction[0] == 0:
        st.success("✅ **Low Risk**")
    else:
        st.error("⚠️ **High Risk**")

    st.write(f"Low Risk Probability: **{probability[0][0]*100:.2f}%**")
    st.write(f"High Risk Probability: **{probability[0][1]*100:.2f}%**")

# Show Metrics
st.subheader(f"📊 {model_option} Model Performance")
st.table(pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
    'Score': [f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", f"{roc_auc:.4f}"]
}))

# Confusion Matrix
st.subheader("🧩 Confusion Matrix on Test Set")
cm = confusion_matrix(y_test, y_test_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Low Risk", "High Risk"], yticklabels=["Low Risk", "High Risk"])
plt.xlabel("Predicted")
plt.ylabel("True")
st.pyplot(fig)
