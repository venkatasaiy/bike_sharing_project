import torch
import pickle
import pandas as pd
from model import BikeSharingModel  # Adjusted to import from src.model

def load_model_and_preprocessor():
    """
    Load the saved model and preprocessor.
    
    Returns:
    - model (BikeSharingModel): Loaded model.
    - preprocessor (ColumnTransformer): Loaded preprocessor.
    """
    with open("models/preprocessor.pkl", "rb") as f:  # Relative path from src/ to models/
        preprocessor = pickle.load(f)
    
    checkpoint = torch.load("models/bike_sharing.pkl")  # Relative path from src/ to models/
    input_size = checkpoint["input_size"]
    model = BikeSharingModel(input_size)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    return model, preprocessor

def predict(model, preprocessor, input_data):
    """
    Make a prediction based on input data.
    
    Parameters:
    - model (BikeSharingModel): Loaded model.
    - preprocessor (ColumnTransformer): Loaded preprocessor.
    - input_data (dict): Input data for prediction.
    
    Returns:
    - prediction (float): Predicted bike-sharing demand.
    - classification (str): Demand classification ("high" or "low").
    """
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Preprocess input data
    input_preprocessed = preprocessor.transform(input_df).toarray()
    input_tensor = torch.tensor(input_preprocessed, dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(input_tensor).item()
    
    # Classify demand (note: correcting 'high'/'low' to match 'High'/'Low' in app.py for consistency)
    classification = "High" if prediction >= 100 else "Low"
    
    return prediction, classification

if __name__ == "__main__":
    # Test the module
    model, preprocessor = load_model_and_preprocessor()
    sample_input = {
        "season": 1, "holiday": 0, "workingday": 1, "weather": 1,
        "temp": 20.0, "humidity": 50, "windspeed": 10,
        "hour": 8, "dayofweek": 2, "month": 6
    }
    pred, clas = predict(model, preprocessor, sample_input)
    print(f"Prediction: {pred:.2f}, Classification: {clas}")