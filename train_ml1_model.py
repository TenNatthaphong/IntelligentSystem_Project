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

# Check if required columns exist
required_columns = ['Churn', 'Age', 'Tenure', 'Usage Frequency', 'Support Calls', 
                    'Payment Delay', 'Total Spend', 'Gender', 'Subscription Type', 'Contract Length']
missing_train = [col for col in required_columns if col not in train_data.columns]
missing_test = [col for col in required_columns if col not in test_data.columns]
if missing_train:
    raise ValueError(f"Missing columns in training dataset: {missing_train}")
if missing_test:
    raise ValueError(f"Missing columns in testing dataset: {missing_test}")

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