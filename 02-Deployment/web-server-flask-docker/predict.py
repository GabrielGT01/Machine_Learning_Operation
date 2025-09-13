
import joblib
import pandas as pd
from xgboost import XGBRegressor

def load_preprocessor(path: str):
    try:
        preprocessor = joblib.load(path)
        print("[INFO] Preprocessor loaded successfully.")
        return preprocessor
    except FileNotFoundError:
        print(f"[ERROR] Preprocessor file not found: {path}")
        raise
    except Exception as e:
        print(f"[ERROR] Failed to load preprocessor: {e}")
        raise

def load_model(path: str):
    try:
        model = XGBRegressor()
        model.load_model(path) 
        print("[INFO] Model loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"[ERROR] Model file not found: {path}")
        raise
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise

def predict_duration(preprocessor, model, ride_df):
    try:
        X_processed = preprocessor.transform(ride_df)
        prediction = model.predict(X_processed)
        return prediction[0]
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        raise

def predict_from_dict(ride: dict):
    """function to predict from a simple ride dictionary."""
    try:
        preprocessor = load_preprocessor("preprocessing.pkl")
        model = load_model("my_model.ubj")

        df = pd.DataFrame([ride])
        return predict_duration(preprocessor, model, df)
    except Exception as e:
        print(f"[ERROR] Failed to predict from dict: {e}")
        return None

if __name__ == "__main__":
    try:
        preprocessor = load_preprocessor("preprocessing.pkl")
        model = load_model("my_model.ubj")
        predicted_duration = predict_duration(preprocessor, model, ride)
        print(f"[RESULT] Predicted trip duration: {predicted_duration:.2f} minutes")
    except Exception:
        print("[FAILED] Prediction pipeline could not complete.")
