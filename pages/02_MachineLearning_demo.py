import streamlit as st
import joblib
import pandas as pd

# Load pre-trained models
random_forest_model = joblib.load('churn_RandomForest.pkl')  # Random Forest Model
xgboost_pipeline = joblib.load('churn_XGBoost.pkl')  # XGBoost Model ที่รวม Preprocessing

# Create the UI
st.title("Customer Churn Prediction")

# Get input from the user
age = st.number_input('Age', min_value=18, max_value=100, value=30)
gender = st.selectbox('Gender', ['Male', 'Female'])
tenure = st.number_input('Tenure (years)', min_value=0, max_value=10, value=2)
usage_frequency = st.number_input('Usage Frequency', min_value=0, max_value=100, value=50)
support_calls = st.number_input('Support Calls', min_value=0, max_value=10, value=2)
payment_delay = st.number_input('Payment Delay', min_value=0, max_value=100, value=5)
subscription_type = st.selectbox('Subscription Type', ['Basic', 'Standard', 'Premium'])
contract_length = st.selectbox('Contract Length', ['Monthly', 'Quarterly', 'Annual'])
total_spend = st.number_input('Total Spend ($)', min_value=0, max_value=1000, value=200)
last_interaction = st.number_input('Last Interaction (days)', min_value=0, max_value=365, value=30)

# Prepare input data for prediction (as a DataFrame)
input_data = pd.DataFrame([[age, gender, tenure, usage_frequency, support_calls, payment_delay, 
                            subscription_type, contract_length, total_spend, last_interaction]],
                          columns=['Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 
                                   'Subscription Type', 'Contract Length', 'Total Spend', 'Last Interaction'])

# Model selection dropdown
model_choice = st.selectbox('Select Model', ['Random Forest', 'XGBoost'])

# Prediction button
if st.button('Predict'):
    if model_choice == 'Random Forest':
        prediction = random_forest_model.predict(input_data)
    elif model_choice == 'XGBoost':
        prediction = xgboost_pipeline.predict(input_data)  # ใช้ Pipeline ที่มี Preprocessing อยู่แล้ว

    prediction_text = 'Churn' if prediction[0] == 1 else 'Not Churn'
    st.write(f'Prediction: {prediction_text}')
