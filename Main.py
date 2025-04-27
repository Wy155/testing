import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from imblearn.over_sampling import SMOTE

# --- Functions ---

@st.cache_data
def load_data():
    df = pd.read_csv("credit_risk_dataset.csv")
    return df

def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)  # Use prob for ROC AUC
    return accuracy, precision, recall, f1, roc_auc, y_pred, y_prob

def get_model(model_option):
    if model_option == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif model_option == "SVM":
        model = SVC(probability=True, random_state=42)
    elif model_option == "Naive Bayes":
        param_grid = {'var_smoothing': np.logspace(-9, -6, 10)}
        grid_GaussianNB = GridSearchCV(MultinomialNB(priors=[0.5, 0.5]), param_grid, cv=5)
        model = grid_GaussianNB
    return model

def find_optimal_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youdens_j = tpr - fpr
    best_threshold = thresholds[np.argmax(youdens_j)]
    return best_threshold

# --- Main App ---

# Title
st.title("🏦 Credit Risk Prediction Dashboard")

# Sidebar
st.sidebar.header("🔍 Model and Input Settings")
model_option = st.sidebar.selectbox("Select Model", ["Random Forest", "SVM", "Naive Bayes"])
use_smote = st.sidebar.checkbox("Apply SMOTE to Balance Classes (Recommended for Naive Bayes)", value=True)
apply_pca = st.sidebar.checkbox("Apply PCA (Recommended for Naive Bayes)", value=True)
n_components = st.sidebar.slider("Number of PCA Components", min_value=2, max_value=10, value=5)

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

# Apply PCA if selected
if apply_pca:
    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

# Get Model
model = get_model(model_option)
model.fit(X_train, y_train)

# Predict Probabilities
accuracy_default, precision_default, recall_default, f1_default, roc_auc_default, y_test_pred_default, y_prob = evaluate_model(model, X_test, y_test, 0.5)

# Find Optimal Threshold (Youden's J)
optimal_threshold = find_optimal_threshold(y_test, y_prob)

# Sidebar - Threshold Tuning
threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, optimal_threshold, 0.01)
st.sidebar.markdown(f"🧠 **Recommended Optimal Threshold (Youden's J): {optimal_threshold:.2f}**")

# Evaluate with selected threshold
accuracy, precision, recall, f1, roc_auc, y_test_pred, _ = evaluate_model(model, X_test, y_test, threshold)

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
if apply_pca:
    input_data_scaled = pca.transform(input_data_scaled)

# Prediction
if submit_button:
    probability = model.predict_proba(input_data_scaled)
    prediction = (probability[:,1] >= threshold).astype(int)

    st.subheader("🔮 Prediction Result")
    if prediction[0] == 0:
        st.success("✅ **Low Risk**")
    else:
        st.error("⚠️ **High Risk**")

    st.write(f"Low Risk Probability: **{probability[0][0]*100:.2f}%**")
    st.write(f"High Risk Probability: **{probability[0][1]*100:.2f}%**")
    st.write(f"Applied Threshold: **{threshold:.2f}**")

# Show Metrics
st.subheader(f"📊 {model_option} Model Performance (Threshold = {threshold:.2f})")
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

# ROC Curve
st.subheader("📈 ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, color='blue', label=f"ROC curve (AUC = {roc_auc_default:.2f})")
ax2.plot([0, 1], [0, 1], color='grey', linestyle='--')
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("Receiver Operating Characteristic (ROC)")
ax2.legend()
st.pyplot(fig2)
