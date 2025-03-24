import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# ✅ Load dataset
df = pd.read_csv("data/train.csv")

# ✅ Check missing values before handling
print("Missing values before handling:\n", df.isnull().sum())

# ✅ Separate numeric and categorical columns
numeric_cols = ['Systolic BP', 'Diastolic', 'BS', 'BMI', 'Heart Rate']
categorical_cols = ['Previous Complications', 'Preexisting Diabetes', 'Gestational Diabetes', 'Mental Health', 'Risk Level']

# ✅ Fill missing values in numeric columns with median
df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.median()))

# ✅ Fill missing values in categorical columns with mode (most frequent value)
df[categorical_cols] = df[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))

# ✅ Drop rows where 'Risk Level' is still missing
df = df.dropna(subset=['Risk Level'])

# ✅ Encode categorical target variable (ONLY "Low" and "High")
risk_mapping = {"Low": 0, "High": 1}
df = df[df["Risk Level"].isin(risk_mapping)]  # Remove any unexpected classes
df["Risk Level"] = df["Risk Level"].map(risk_mapping)

# ✅ Verify unique classes
print("\n🟢 Unique Risk Levels in Dataset:", df["Risk Level"].unique())

# ✅ Split features and target
X = df.drop(columns=['Risk Level'])
y = df['Risk Level']

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Train model with optimized parameters
model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
model.fit(X_train_scaled, y_train)

# ✅ Model Predictions
y_pred = model.predict(X_test_scaled)

# ✅ Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.2f}")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=["Low", "High"]))

# ✅ Ensure the directory exists before saving
model_dir = "ai_model"
os.makedirs(model_dir, exist_ok=True)  # Create directory if it doesn't exist

# ✅ Save the trained model & scaler
joblib.dump(model, os.path.join(model_dir, "pregnancy_risk_model.pkl"))
joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

print("\n✅ Model and Scaler saved successfully!")