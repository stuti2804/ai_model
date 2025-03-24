import os
import sys
import joblib
import numpy as np

print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")

try:
    # Try to import scikit-learn and print its version
    import sklearn
    print(f"scikit-learn version: {sklearn.__version__}")
except ImportError:
    print("scikit-learn not installed")

try:
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Try to load the model and scaler with explicit path
    model_path = os.path.join(current_dir, "pregnancy_risk_model.pkl")
    
    print(f"Attempting to load model from: {model_path}")
    print(f"File exists: {os.path.exists(model_path)}")
    
    model = joblib.load(model_path)
    print("Model loaded successfully!")
    
    # Test with a dummy input
    dummy_input = np.zeros((1, 11))  # Matching the 11 features
    result = model.predict(dummy_input)
    print(f"Test prediction result: {result}")
    
except Exception as e:
    print(f"Error loading model: {str(e)}")
    import traceback
    traceback.print_exc()
