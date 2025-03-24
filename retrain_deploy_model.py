import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Print versions for debugging
print(f"NumPy version: {np.__version__}")
print(f"scikit-learn version: {sklearn.__version__ if 'sklearn' in globals() else 'Not imported directly'}")

# Load dataset
try:
    df = pd.read_csv("data/train.csv")
except FileNotFoundError:
    # Try relative path if absolute fails
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "../data/train.csv"))

print(f"Dataset shape: {df.shape}")

# Handle missing values
numeric_cols = ['Systolic BP', 'Diastolic', 'BS', 'BMI', 'Heart Rate']
categorical_cols = ['Previous Complications', 'Preexisting Diabetes', 'Gestational Diabetes', 'Mental Health', 'Risk Level']

# Fill missing values
df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.median()))
df[categorical_cols] = df[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))

# Drop rows where 'Risk Level' is still missing
df = df.dropna(subset=['Risk Level'])

# Encode categorical target variable
risk_mapping = {"Low": 0, "High": 1}
df = df[df["Risk Level"].isin(risk_mapping)]
df["Risk Level"] = df["Risk Level"].map(risk_mapping)

# Split features and target
X = df.drop(columns=['Risk Level'])
y = df['Risk Level']

# Use all features in the correct order
feature_names = [
    "Age", "Systolic BP", "Diastolic", "BS", "Body Temp", "BMI",
    "Previous Complications", "Preexisting Diabetes", "Gestational Diabetes",
    "Mental Health", "Heart Rate"
]

# Make sure dataset has all required features
for feature in feature_names:
    if feature not in X.columns:
        print(f"Warning: {feature} not in dataset. Available columns: {X.columns.tolist()}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a simpler model to avoid version issues
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model and scaler in the current directory for easy access
joblib.dump(model, "pregnancy_risk_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully!")
