import pandas as pd
import streamlit as st

# Page title
st.title("Model Description: Car Price Prediction Model")

# Header
st.header("Model Theory")
st.write("""
This model predicts the selling price of used cars based on various features like year, mileage, engine power, etc. 
The model uses a deep learning approach with a neural network built using TensorFlow and Keras.
The goal is to predict the selling price of a car, given its attributes.

The key aspects of the model are:
- **Feature Selection**: We used a subset of features, including the year, mileage, engine power, and more.
- **Data Transformation**: The target variable (`selling_price`) was transformed using a logarithmic scale to reduce skewness.
- **Neural Network**: A simple feed-forward neural network with two hidden layers and dropout for regularization.

For more information, you can refer to the original model development and theory: [TensorFlow Documentation](https://www.tensorflow.org/).
""")

# Header: MLP Explanation
st.header("Model Theory: MLP ")
st.write("""
A **Multi-Layer Perceptron (MLP)** is a type of artificial neural network composed of multiple layers of neurons, typically including an input layer, one or more hidden layers, and an output layer. Each neuron in one layer is connected to every neuron in the next layer, making it a fully connected network. MLPs are capable of learning non-linear relationships between input features and output targets, making them highly suitable for regression and classification tasks.

We use MLPs because they can approximate complex functions, learn from large amounts of data, and generalize well to new, unseen data. In this model, the MLP learns the relationship between various car features (like mileage, engine power, and year) and the target variable (selling price). By using multiple layers, MLPs can capture intricate patterns in the data that simpler models might miss.

For more detailed information, you can refer to the following resource: [Understanding MLPs](https://www.geeksforgeeks.org/multi-layer-perceptron-learning-in-tensorflow/)
""")

# Header: Feature Descriptions
st.header("Feature Descriptions and Example Values")
data = {
    'Feature': ['name', 'year', 'selling_price', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'seats', 'max_power (in bph)', 'Mileage Unit', 'Mileage', 'Engine (CC)'],
    'Description': [
        'The name of the car (e.g., make and model).',
        'The year of manufacture of the car.',
        'The selling price of the car (log-transformed for the model).',
        'The number of kilometers the car has been driven.',
        'The fuel type used by the car (e.g., petrol, diesel).',
        'The type of seller (e.g., individual, dealer).',
        'The type of transmission (e.g., manual, automatic).',
        'The number of previous owners of the car.',
        'The number of seats in the car.',
        'The maximum power of the car (measured in brake horsepower).',
        'The unit of measurement for mileage (e.g., kmpl).',
        'The mileage of the car.',
        'The engine capacity of the car in cubic centimeters (CC).'
    ],
    'Example 1': ['Honda City', 2015, 10.2, 45000, 'Petrol', 'Individual', 'Automatic', 1, 5, 100, 'kmpl', 15.6, 1500],
    'Example 2': ['Toyota Corolla', 2017, 11.3, 30000, 'Diesel', 'Dealer', 'Manual', 2, 5, 120, 'kmpl', 18.2, 1800],
    'Example 3': ['Ford Figo', 2016, 8.5, 70000, 'Petrol', 'Individual', 'Manual', 1, 5, 90, 'kmpl', 12.5, 1200]
}

# Create a DataFrame for displaying the table
df = pd.DataFrame(data)

# Display the table in Streamlit
st.write("### Feature Descriptions and Example Values")
st.write(df)

# Dataset source
st.write("**Dataset Source**: The dataset is sourced from [Kaggle - Sample34](https://www.kaggle.com/datasets/jacksondivakarr/sample34).")


# Code explanation
st.header("Code Explanation")
st.subheader("1. Data Loading and Preprocessing")
st.write("""
The dataset is loaded using `pandas` from a CSV file. The 'selling_price' is then log-transformed to reduce skewness and make the distribution more normal. We use `train_test_split` to divide the dataset into training and testing sets.
""")
st.code("""
data = pd.read_csv('data/car.csv')
data['selling_price'] = np.log1p(data['selling_price'])
X = data[['year', 'km_driven', 'seats', 'max_power (in bph)', 'Mileage', 'Engine (CC)']]
y = data['selling_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
""")

st.subheader("2. Data Normalization")
st.write("""
We scale the feature data (`X_train` and `X_test`) using `StandardScaler` to standardize the input features so that the neural network can train efficiently.
""")
st.code("""
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
""")

st.subheader("3. Model Architecture")
st.write("""
We build a Sequential model with:
- An input layer of 512 neurons with ReLU activation
- A dropout layer to prevent overfitting
- A hidden layer with 256 neurons and ReLU activation
- An output layer with 1 neuron (for price prediction)

The model is compiled with the Adam optimizer and Mean Squared Error loss function, optimized to minimize the error between predicted and actual prices.
""")
st.code("""
model = Sequential([
    Dense(512, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])
""")

st.subheader("4. Early Stopping and Training")
st.write("""
We use EarlyStopping to prevent overfitting and ensure the model stops training once the validation loss stops improving for 20 epochs.
The model is trained for a maximum of 1000 epochs with a batch size of 32.
""")
st.code("""
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
""")

st.subheader("5. Model Evaluation")
st.write("""
After training, the model is evaluated on the test data. We print the **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)** to assess the model's performance. The lower the MSE and MAE, the better the model.
""")
st.code("""
loss, mae = model.evaluate(X_test, y_test)
print(f'Mean Squared Error on Test Data: {loss}')
print(f'Mean Absolute Error on Test Data: {mae}')
""")

# Display model performance
st.header("Model Performance")
st.write("""
- **Mean Squared Error (MSE) on Test Data**: 0.056
- **Mean Absolute Error (MAE) on Test Data**: 0.170
""")

# Model Save and Scaler Save
st.header("Model and Scaler Saving")
st.write("""
The trained model is saved to a `.h5` file, and the scaler used to normalize the data is saved using `pickle` for future use. This allows us to make predictions on new, unseen data without retraining the model.
""")
st.code("""
model.save('car_model.h5')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
""")

st.write("**See full code on GitHub**: [Intelligent_Project GitHub Repository](https://github.com/TenNatthaphong/Intelligent_Project)")