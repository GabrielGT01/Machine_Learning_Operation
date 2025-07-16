
import os
import joblib
import pandas as pd
from xgboost import XGBRegressor
import mlflow

# === Configuration ===
PREPROCESSOR_URI = "mlflow-artifacts:/1/ab7ac267f5e945cba9566b7213e58524/artifacts/preprocessor/preprocessing.pkl"
MODEL_URI = "mlflow-artifacts:/1/ab7ac267f5e945cba9566b7213e58524/artifacts/nyc-duration-model/model.xgb"

# === Load Preprocessor ===
def load_preprocessor():
    try:
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=PREPROCESSOR_URI)
        preprocessor = joblib.load(local_path)
        print("[INFO] Preprocessor loaded successfully.")
        return preprocessor
    except Exception as e:
        print(f"[ERROR] Failed to load preprocessor: {e}")
        raise

# === Load XGBoost Model ===
def load_model():
    try:
        model_path = mlflow.artifacts.download_artifacts(artifact_uri=MODEL_URI)
        model = XGBRegressor()
        model.load_model(model_path)
        print("[INFO] Model loaded successfully.")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise

# === Make Prediction ===
def predict_duration(preprocessor, model, ride_df):
    try:
        X_processed = preprocessor.transform(ride_df)
        prediction = model.predict(X_processed)
        return prediction[0]
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        raise

# === Predict from Dictionary ===
def predict_from_dict(ride: dict):
    try:
        preprocessor = load_preprocessor()
        model = load_model()
        df = pd.DataFrame([ride])
        return predict_duration(preprocessor, model, df)
    except Exception as e:
        print(f"[ERROR] Failed to predict from dict: {e}")
        return None


if __name__ == "__main__":
    
    try:
        preprocessor = load_preprocessor()
        model = load_model()
        predicted_duration = predict_duration(preprocessor, model, pd.DataFrame([sample_ride]))
        print(f"[RESULT] Predicted trip duration: {predicted_duration:.2f} minutes")
    except Exception:
        print("[FAILED] Prediction pipeline could not complete.")
