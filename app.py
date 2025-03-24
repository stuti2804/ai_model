from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Load the model and scaler
model = joblib.load("pregnancy_risk_model.pkl")
scaler = joblib.load("scaler.pkl")

feature_names = [
    "Age", "Systolic BP", "Diastolic", "BS", "Body Temp", "BMI",
    "Previous Complications", "Preexisting Diabetes", "Gestational Diabetes",
    "Mental Health", "Heart Rate"
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract features in the correct order
        user_input = np.array([[
            data['age'],
            data['systolicBP'],
            data['diastolic'],
            data['bs'],
            data['bodyTemp'],
            data['bmi'],
            data['prevComplications'],
            data['preexistingDiabetes'],
            data['gestationalDiabetes'],
            data['mentalHealth'],
            data['heartRate']
        ]])

        # Convert to DataFrame and scale
        user_df = pd.DataFrame(user_input, columns=feature_names)
        user_scaled = scaler.transform(user_df)
        
        # Make prediction
        prediction = model.predict(user_scaled)
        risk_level = "High" if prediction[0] == 1 else "Low"
        
        return jsonify({
            'status': 'success',
            'prediction': risk_level
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
