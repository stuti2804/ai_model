import joblib
import numpy as np
import pandas as pd

# Load trained model and scaler
model = joblib.load("ai_model/pregnancy_risk_model.pkl")
scaler = joblib.load("ai_model/scaler.pkl")

# Feature names
feature_names = [
    "Age", "Systolic BP", "Diastolic", "BS", "Body Temp", "BMI",
    "Previous Complications", "Preexisting Diabetes", "Gestational Diabetes",
    "Mental Health", "Heart Rate"
]

# Function to get user input
def get_user_input():
    print("\n🔹 Please enter the following details:")
    
    try:
        age = int(input("Age (years): "))
        systolic_bp = float(input("Systolic BP (mmHg): "))
        diastolic = float(input("Diastolic BP (mmHg): "))
        bs = float(input("Blood Sugar Level (mg/dL): "))
        body_temp = float(input("Body Temperature (°F): "))
        bmi = float(input("BMI: "))
        
        print("\n🔹 Enter '1' for Yes, '0' for No:")
        prev_complications = int(input("Previous Complications (Yes=1, No=0): "))
        pre_diabetes = int(input("Preexisting Diabetes (Yes=1, No=0): "))
        gestational_diabetes = int(input("Gestational Diabetes (Yes=1, No=0): "))
        mental_health = int(input("Mental Health Issues (Yes=1, No=0): "))
        
        heart_rate = float(input("Heart Rate (bpm): "))

        # Store input as a NumPy array
        user_data = np.array([[age, systolic_bp, diastolic, bs, body_temp, bmi,
                               prev_complications, pre_diabetes, gestational_diabetes, 
                               mental_health, heart_rate]])
        return user_data

    except ValueError:
        print("\n🔴 ERROR: Invalid input! Please enter numeric values only.")
        return None

# Get user input
user_input = get_user_input()

if user_input is not None:
    # Convert input to DataFrame for consistency
    user_df = pd.DataFrame(user_input, columns=feature_names)

    # Scale input data
    user_scaled = scaler.transform(user_df)

    # Predict risk level
    prediction = model.predict(user_scaled)

    # Mapping prediction back to risk level
    risk_mapping = {0: "Low", 1: "High"}
    print(f"\n🔍 Pregnancy Risk Prediction: **{risk_mapping[prediction[0]]} Risk**")