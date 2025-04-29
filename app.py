import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessor
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.set_page_config(page_title="Telco Churn Predictor", layout="centered")
st.title("📉 Telco Customer Churn Prediction")

st.markdown("Fill out the customer details below:")

with st.form("churn_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    
    monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

    submit = st.form_submit_button("Predict")

if submit:
    input_df = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }])

    try:
        X_transformed = preprocessor.transform(input_df)
        prediction = model.predict(X_transformed)[0]
        probability = model.predict_proba(X_transformed)[0][1]

        st.success(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
        st.info(f"Churn Probability: {probability:.2%}")
    except ValueError as e:
        st.error(f"Prediction failed: {e}")
