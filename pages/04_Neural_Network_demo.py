import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import shuffle

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('data/car.csv')

data = load_data()

# Data preprocessing
X = data[['year', 'km_driven', 'seats', 'max_power (in bph)', 'Mileage', 'Engine (CC)']]
y = data['selling_price']  # Use actual selling price without log transformation

# Convert year to car age
X['car_age'] = 2024 - X['year']
X['km_driven'] = np.log1p(X['km_driven'])  # Reduce scale of km_driven
X = X.drop(columns=['year'])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate MAE, MSE, R2
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Prediction function for the UI
def predict_price(features):
    # Scale the input features
    features_scaled = scaler.transform([features])

    # Make prediction
    prediction = model.predict(features_scaled)  # Predict actual price without using expm1

    return prediction[0]

# Confidence calculation function
def calculate_confidence(price_prediction, true_value):
    # Calculate confidence from Error Percentage
    absolute_error = abs(price_prediction - true_value)
    error_percentage = (absolute_error / true_value) * 100
    confidence = max(0, 100 - error_percentage)
    return confidence

# UI
st.title("Car Price Prediction System")

with st.form(key='prediction_form'):
    year = st.number_input('Manufacturing Year', min_value=1990, max_value=2025, value=2020)
    km_driven = st.number_input('Distance Driven (in km)', min_value=0, value=10000)
    seats = st.number_input('Number of Seats', min_value=2, max_value=10, value=5)
    max_power = st.number_input('Maximum Power (in BHP)', min_value=0, value=100)
    mileage = st.number_input('Mileage (in KM/L)', min_value=0.0, value=15.0)
    engine_cc = st.number_input('Engine Size (in CC)', min_value=800, max_value=5000, value=1500)

    submit_button = st.form_submit_button(label='Predict Price')

if submit_button:
    try:
        input_features = [year, km_driven, seats, max_power, mileage, engine_cc]
        price_prediction = predict_price(input_features)

        # Calculate confidence
        true_value = y_train.iloc[0]  # Use actual value from y_train
        confidence = calculate_confidence(price_prediction, true_value)

        st.subheader('Prediction Result')
        st.write(f'Predicted Price: ${price_prediction:,.2f}')
        st.write(f'Confidence: {confidence:.2f}%')

    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
