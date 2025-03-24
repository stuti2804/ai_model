import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Load trained model and scaler
model = joblib.load("ai_model/pregnancy_risk_model.pkl")
scaler = joblib.load("ai_model/scaler.pkl")

# Feature names
feature_names = [
    "Age", "Systolic BP", "Diastolic", "BS", "Body Temp", "BMI",
    "Previous Complications", "Preexisting Diabetes", "Gestational Diabetes",
    "Mental Health", "Heart Rate"
]

# Load test dataset
test_df = pd.read_csv("data/test.csv")

# ✅ Ensure test data has required columns
if set(feature_names + ["Risk Level"]) - set(test_df.columns):
    print("🔴 ERROR: Test dataset is missing required columns.")
    exit()

# ✅ Encode Risk Level (Only "Low" and "High")
risk_mapping = {"Low": 0, "High": 1}
test_df["Risk Level"] = test_df["Risk Level"].map(risk_mapping)

# ✅ Check if mapping worked correctly
print("\n🟢 Risk Level Distribution in Test Data:")
print(test_df["Risk Level"].value_counts())

# ✅ Split features and labels
X_test = test_df[feature_names]
y_test = test_df["Risk Level"]

# ✅ Scale input features
# ✅ Ensure feature names are properly handled during inference
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# ✅ Predict
y_pred = model.predict(X_test_scaled)
# ✅ Check if model predicts both classes (0 = Low, 1 = High)
unique_preds = np.unique(y_pred)
expected_classes = [0, 1]  # Only "Low" and "High"
missing_classes = [c for c in expected_classes if c not in unique_preds]

if missing_classes:
    print(f"\n🔴 WARNING: Model did NOT predict these classes: {missing_classes}")

# ✅ Model Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy on Test Data: {accuracy:.2f}")

# ✅ Classification Report (Only for "Low" and "High")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=["Low", "High"]))

# ✅ Single Sample Prediction
sample_features = np.array([[22, 90, 60, 9, 100, 18, 1, 1, 0, 1, 80]])
sample_scaled = pd.DataFrame(scaler.transform(sample_features), columns=feature_names)
sample_prediction = model.predict(sample_scaled)

reverse_mapping = {0: "Low", 1: "High"}
print(f"\n🔍 Sample Prediction: {reverse_mapping[sample_prediction[0]]} (Risk Level)")