import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Use cache_data (new Streamlit caching)
@st.cache_data
def load_data():
    df = pd.read_csv("credit_risk_dataset.csv")
    return df

# Load data
df = load_data()
df = df.assign(
    person_emp_length=df['person_emp_length'].fillna(df['person_emp_length'].median()),
    loan_int_rate=df['loan_int_rate'].fillna(df['loan_int_rate'].median())
)

# Split data
X = df.drop(columns=['loan_status'], axis=1)
X = X.select_dtypes(include=[np.number])
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model evaluation function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return accuracy, precision, recall, f1, roc_auc, y_pred

# Streamlit layout
st.title("🏦 Credit Risk Prediction Dashboard")

st.sidebar.header("🔍 Choose a Model")
model_option = st.sidebar.selectbox("Select a model:", ["Random Forest", "SVM", "Naive Bayes"])

st.sidebar.header("📝 Input Features")

person_age = st.sidebar.number_input("Person Age", min_value=0, max_value=100, value=18)
person_income = st.sidebar.number_input("Person Income", min_value=0, value=0)
person_emp_length = st.sidebar.number_input("Person Employment Length (years)", min_value=0, max_value=50, value=0)
loan_amnt = st.sidebar.number_input("Loan Amount", min_value=0, value=0)
loan_int_rate = st.sidebar.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=100.0, value=0.0)
loan_percent_income = st.sidebar.number_input("Loan Percent of Income (%)", min_value=0.0, max_value=100.0, value=0.0)
cb_person_cred_hist_length = st.sidebar.number_input("Credit History Length (years)", min_value=0, max_value=50, value=0)

input_data = pd.DataFrame({
    'person_age': [person_age],
    'person_income': [person_income],
    'person_emp_length': [person_emp_length],
    'loan_amnt': [loan_amnt],
    'loan_int_rate': [loan_int_rate],
    'loan_percent_income': [loan_percent_income],
    'cb_person_cred_hist_length': [cb_person_cred_hist_length],
})

# Standardize input
input_data_scaled = scaler.transform(input_data)

# Model training and prediction
if model_option == "Random Forest":
    model = RandomForestClassifier(random_state=42)
elif model_option == "SVM":
    model = SVC(probability=True, random_state=42)
elif model_option == "Naive Bayes":
    param_grid = {'var_smoothing': np.logspace(-9, -6, 10)}
    grid_GaussianNB = GridSearchCV(GaussianNB((priors=[0.5,0.5])), param_grid, cv=5)
    model = grid_GaussianNB.fit(X_train, y_train)

model.fit(X_train, y_train)
accuracy, precision, recall, f1, roc_auc, y_test_pred = evaluate_model(model, X_test, y_test)
prediction = model.predict(input_data_scaled)

# Show metrics
st.subheader(f"📊 {model_option} Model Performance")
st.write(f"**Accuracy:** {accuracy:.4f}")
st.write(f"**Precision:** {precision:.4f}")
st.write(f"**Recall:** {recall:.4f}")
st.write(f"**F1 Score:** {f1:.4f}")
st.write(f"**ROC AUC Score:** {roc_auc:.4f}")

# Prediction result
st.subheader("🔮 Prediction for Your Input")
if prediction[0] == 0:
    st.success("✅ **Prediction: Low Risk**")
else:
    st.error("⚠️ **Prediction: High Risk**")

# Confusion matrix
st.subheader("🧩 Confusion Matrix on Test Set")
cm = confusion_matrix(y_test, y_test_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["Low Risk", "High Risk"],
            yticklabels=["Low Risk", "High Risk"])
plt.xlabel("Predicted")
plt.ylabel("True")
st.pyplot(fig)
