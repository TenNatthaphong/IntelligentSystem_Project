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
