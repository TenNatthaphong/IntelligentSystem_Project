import streamlit as st
import pandas as pd

# Set page title
st.set_page_config(page_title="Model Description", page_icon="📊")

# Header for the page
st.title("Customer Churn Model - Description")
st.write("This page explains two different machine learning models used to predict customer churn based on a dataset from Kaggle.")

# Example data (you can replace this with actual dataset rows if you'd like)
data = {
    'Feature': ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Gender', 'Subscription Type', 'Contract Length'],
    'Description': [
        'The age of the customer.',
        'The duration (in months) the customer has been with the service.',
        'The frequency of customer usage of the service (e.g., daily, weekly).',
        'The number of calls made by the customer to support.',
        'The number of days the customer delayed payment.',
        'The total money spent by the customer.',
        'The gender of the customer.',
        'Type of subscription the customer has.',
        'Duration of the contract in months.'
    ],
    'Example 1': [25, 12, 'Daily', 1, 5, 200.50, 'Male', 'Basic', 12],
    'Example 2': [40, 24, 'Weekly', 3, 10, 150.00, 'Female', 'Premium', 24],
    'Example 3': [35, 36, 'Monthly', 0, 0, 300.75, 'Male', 'Gold', 36]
}

# Create a DataFrame for displaying the table
df = pd.DataFrame(data)

# Display the table in Streamlit
st.write("### Feature Descriptions and Example Values")
st.write(df)

# Dataset source
st.write("**Dataset Source**: The dataset is sourced from [Kaggle - Customer Churn Dataset](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset?resource=download).")

# Model selection menu
model_option = st.selectbox(
    "Select the model you want to learn about:",
    ("Random Forest", "XGBoost")
)

# Dataset feature summary
dataset_info = """
**Dataset Features**:
- **Age**: The age of the customer.
- **Tenure**: The number of months the customer has been with the service.
- **Usage Frequency**: How often the customer uses the service.
- **Support Calls**: Number of calls the customer has made to support.
- **Payment Delay**: The delay in the customer's payments.
- **Total Spend**: The total amount spent by the customer.
- **Gender**: The gender of the customer.
- **Subscription Type**: The type of subscription (e.g., Basic, Premium).
- **Contract Length**: Length of the contract (e.g., 1 year, 2 years).
"""

# Block-by-block explanation of models based on selection
if model_option == "Random Forest":
    st.header("Random Forest Classifier")
    st.subheader("Random Forest Theory")
    st.write("""
    Random Forest is an ensemble learning algorithm that works by combining multiple decision trees to make a prediction. It’s like having a team of experts (the trees) that vote on a decision, and the most popular vote wins.

    Here’s a simple way to understand how it works:

    1. **Building Multiple Decision Trees**: A decision tree is a model that splits data based on certain conditions, like "Is the customer older than 30?" or "Does the customer have a premium subscription?". The tree branches out to make predictions based on different features of the data. In a random forest, many decision trees are created, each one looking at a different random subset of the data.
      
    2. **Making Predictions**: Once the trees are built, the random forest uses them to make predictions. For classification (like predicting whether a customer will churn or not), each tree votes for a class (Churn or No Churn), and the class with the most votes becomes the final prediction.

    3. **Randomness**: The "random" part comes from two things:
      - Each tree is trained on a random subset of the data.
      - When making splits in the tree, the algorithm only considers a random subset of features, which makes the model more diverse and robust.

    ### Why Random Forest is Powerful:
    - **Reduces Overfitting**: While a single decision tree can easily overfit to noisy data, Random Forest averages many trees, which helps in reducing overfitting.
    - **Handles Missing Data**: Random Forest can handle missing values, so you don’t have to worry too much about cleaning your data.
    - **Handles Complex Data**: It can easily handle both categorical and numerical data, making it versatile.

    ### Example:
    Imagine you’re at a carnival with several fortune tellers (the trees). You ask each one about your future (whether the customer will churn or not). Some fortune tellers may have different predictions, but after getting opinions from many of them, you trust the one that is most common among all of them (the majority vote). This is how Random Forest works!

    ### Reference:
    For more detailed information, you can refer to this [Random Forest Wikipedia page](https://en.wikipedia.org/wiki/Random_forest).
    """)

    st.subheader("1. Data Preprocessing")
    st.write(f"""
        The dataset is loaded from CSV files, and missing values in the target variable **Churn** are dropped to ensure we only train on complete data.

        ### Feature Grouping:
        - **Numeric Features**: These features represent continuous or numerical data and are scaled using **StandardScaler** to bring them to the same scale for better model performance.
            - **Numeric Features Scaling**: The numeric features like Age and Total Spend are scaled so that they all have the same scale, preventing larger values from dominating the model.
          - Example: If Age is in the range [20, 50] and Total Spend is in the range [100, 1000], after scaling, both features will have similar ranges, typically centered around 0.


        - **Categorical Features**: These features are non-numeric and are one-hot encoded using **OneHotEncoder** to convert them into binary variables (0 or 1).
            - **One-Hot Encoding**: For categorical features like Gender and Subscription Type, one-hot encoding creates a binary variable for each category. For instance:
            - **Gender**: 
              - `Gender_Male` = 1 if the customer is male, and `Gender_Female` = 0.
              - `Gender_Female` = 1 if the customer is female, and `Gender_Male` = 0.
            - **Subscription Type**: 
              - `Subscription_Type_Basic` = 1 if the customer has a Basic subscription, and 0 otherwise.
              - `Subscription_Type_Premium` = 1 if the customer has a Premium subscription, and 0 otherwise.

        These preprocessing steps help standardize the data and ensure that the model can efficiently learn patterns from both numerical and categorical features.
        """)
    st.subheader("2. Model Building")
    st.write("""
        The Random Forest model is built using **50 estimators (trees)** and a fixed **random_state** for reproducibility.
        Random Forest works by creating multiple decision trees during training and outputs the class that is the mode of the classes output by individual trees.
    """)
    
    st.subheader("3. Evaluation")
    st.write("""
        The model is evaluated using:
        - **Accuracy**: Measures the percentage of correct predictions.
        - **Classification Report**: Includes precision, recall, and F1-score for each class.
        - **Confusion Matrix**: Shows the confusion between the actual and predicted values.
    """)
    
    st.subheader("4. Model Accuracy")
    st.write("Accuracy: `0.513`")

    st.subheader("5. Code Reference")
    st.write("""
        You can refer to the code block below for building and training the Random Forest model.
        The **dataset** is used consistently across both models.
    """)
    
    st.code("""
import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# Define file paths
train_path = 'data/customer_churn_dataset-training-master.csv'
test_path = 'data/customer_churn_dataset-testing-master.csv'

# Check if files exist
if not os.path.exists(train_path):
    raise FileNotFoundError(f"Training file not found: {train_path}")
if not os.path.exists(test_path):
    raise FileNotFoundError(f"Testing file not found: {test_path}")

# Load datasets
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Check if datasets are empty
if train_data.empty:
    raise ValueError("Training dataset is empty.")
if test_data.empty:
    raise ValueError("Testing dataset is empty.")

# Drop rows with missing target values in 'Churn'
train_data = train_data.dropna(subset=['Churn'])
test_data = test_data.dropna(subset=['Churn'])

# Define features and target
X_train = train_data.drop('Churn', axis=1)
y_train = train_data['Churn']
X_test = test_data.drop('Churn', axis=1)
y_test = test_data['Churn']

# Define feature groups
numeric_features = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend']
categorical_features = ['Gender', 'Subscription Type', 'Contract Length']

# Create transformers for numerical and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing numeric values
    ('scaler', StandardScaler())  # Scale numeric features
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing categorical values
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
])

# Combine transformers into a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Create the pipeline with preprocessing and classifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model
joblib.dump(model_pipeline, 'churn_RandomForest.pkl')
    """)
    
elif model_option == "XGBoost":
    st.header("XGBoost Classifier")

    st.subheader("XGBoost Theory")
    st.write("""
### What is XGBoost?

XGBoost, which stands for "Extreme Gradient Boosting", is a powerful and efficient machine learning algorithm that works by building multiple decision trees in a sequential manner. Each tree tries to correct the mistakes of the previous one, making the model better over time.

Here’s a simple way to understand how XGBoost works:

1. **Gradient Boosting**: Instead of training all the trees independently, XGBoost adds new trees to correct the errors made by the previous ones. The term "gradient" refers to the technique used to minimize errors by adjusting the model in small steps (like climbing down a hill). This process helps the model learn from its mistakes.

2. **Sequential Learning**: Each new tree focuses on the examples where the previous trees made errors. This is like having a team of teachers who learn from each other to improve. One teacher might not get everything right, but the next one can adjust based on the first teacher’s mistakes.

3. **Regularization**: XGBoost includes a regularization term (something to keep the model from becoming too complex), which helps prevent overfitting, a common problem when a model learns the noise in the data rather than the true patterns.

4. **Why It’s Fast**: XGBoost is known for being very fast. It has optimizations that allow it to scale well with larger datasets, and it uses parallel processing to speed up training.

### Why XGBoost is Powerful:
- **High Performance**: XGBoost has become the go-to algorithm for many machine learning competitions due to its high performance and accuracy.
- **Handles Missing Values**: Like Random Forest, XGBoost can handle missing data without needing to manually fill in those gaps.
- **Prevents Overfitting**: The regularization techniques in XGBoost help avoid overfitting, even with complex models.

### Example:
Imagine you're assembling a team of detectives (the trees). Each detective investigates a case (predicting churn), but they’re not perfect. The first detective misses a clue, and the second one improves by focusing on the clues the first one missed. This process continues, with each detective improving on the last one. This is how XGBoost works — a series of trees (detectives) that work together to get better results.

### Reference:
For more detailed information, you can refer to this [XGBoost Wikipedia page](https://en.wikipedia.org/wiki/XGBoost).
""")
    st.subheader("1. Data Preprocessing")
    st.write(f"""
        Similar to Random Forest, the dataset is loaded, and rows with missing target values are dropped.
        **Preprocessing** steps include:
        - **Standard scaling** for numeric features.
        - **One-hot encoding** for categorical features.
    """)
    
    st.subheader("2. Model Building")
    st.write("""
        XGBoost (Extreme Gradient Boosting) is a gradient boosting algorithm that combines multiple weak models to create a strong model.
        We use **100 estimators** and regularization parameters such as **max_depth** and **learning_rate** to control the complexity and overfitting.
    """)
    
    st.subheader("3. Evaluation")
    st.write("""
        Similar to Random Forest, the model is evaluated using:
        - **Accuracy**: Measures the percentage of correct predictions.
        - **Classification Report**: Includes precision, recall, and F1-score for each class.
        - **Confusion Matrix**: Shows the confusion between actual and predicted values.
    """)
    
    st.subheader("4. Model Accuracy")
    st.write("Accuracy: `0.513` ")

    st.subheader("5. Code Reference")
    st.write("""
        Below is the code block used to build and train the XGBoost model.
    """)
    
    st.code("""
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load datasets
train_data = pd.read_csv('data/customer_churn_dataset-training-master.csv')
test_data = pd.read_csv('data/customer_churn_dataset-testing-master.csv')

# Drop rows with missing target values in 'Churn'
train_data = train_data.dropna(subset=['Churn'])
test_data = test_data.dropna(subset=['Churn'])

# Define features and target
X_train = train_data.drop('Churn', axis=1)
y_train = train_data['Churn']
X_test = test_data.drop('Churn', axis=1)
y_test = test_data['Churn']

# Preprocessing for numerical and categorical data
numeric_features = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend']
categorical_features = ['Gender', 'Subscription Type', 'Contract Length']

# Create transformers for both numerical and categorical columns
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Drop CustomerID if exists
X_train = X_train.drop(columns=['CustomerID'], errors='ignore')
X_test = X_test.drop(columns=['CustomerID'], errors='ignore')

# Create a pipeline with preprocessing and XGBoost model
xgboost_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=100,          # Similar to number of trees in RandomForest
        max_depth=10,              # Control the depth of the trees (similar to RandomForest)
        learning_rate=0.1,         # Regularization rate (lower value = slower learning)
        subsample=0.8,             # Randomly sample training data for each tree (similar to RandomForest)
        colsample_bytree=0.8,      # Randomly sample features (similar to RandomForest)
        random_state=42            # Ensure reproducibility
    ))
])

# Train the model
xgboost_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = xgboost_pipeline.predict(X_test)

# Evaluate the model
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the entire pipeline (Preprocessing + Model)
joblib.dump(xgboost_pipeline, 'churn_XGBoost.pkl')
    """)
