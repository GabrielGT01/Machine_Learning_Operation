import os
import joblib
import pandas as pd

# ======== LOCAL PATHS () ========
PREPROCESSOR_PATH = "/var/folders/pk/nk0t185511z8g6hmxr_bhkfw0000gn/T/tmp2ygt0l9s/preprocessing.pkl"
MODEL_DIR = "/var/folders/pk/nk0t185511z8g6hmxr_bhkfw0000gn/T/tmp02cvs65w/artifacts"  # contains MLmodel, model.pkl, etc.
# =============================================================================

def load_preprocessor_local(preprocessor_path: str):
    try:
        if not os.path.isfile(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found at: {preprocessor_path}")
        preprocessor = joblib.load(preprocessor_path)
        print("[INFO] Preprocessor loaded successfully.")
        return preprocessor
    except Exception as e:
        print(f"[ERROR] Failed to load preprocessor: {e}")
        raise

def load_model_local(model_dir: str):
    try:
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        model_pkl = os.path.join(model_dir, "model.pkl")
        if not os.path.isfile(model_pkl):
            raise FileNotFoundError(f"Model file not found: {model_pkl}")
        model = joblib.load(model_pkl)   
        print(f"[INFO] Model loaded successfully:")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise

def predict_duration(preprocessor, model, ride_df: pd.DataFrame) -> float:
    """
    Transform the raw input using the preprocessor and predict duration with the model.
    """
    X_processed = preprocessor.transform(ride_df)
    y_pred = model.predict(X_processed)
    return float(y_pred[0])

def predict_from_dict(ride: dict):
    """
    Convenience wrapper: build a DataFrame from dict, transform, and predict.
    """
    df = pd.DataFrame([ride])
    preprocessor = load_preprocessor_local(PREPROCESSOR_PATH)
    model = load_model_local(MODEL_DIR)
    return predict_duration(preprocessor, model, df)

if __name__ == "__main__":
    # Example input (edit as needed)
    sample_ride = {
        "passenger_count": 1.0,
        "trip_distance": 5.93,
        "fare_amount": 24.70,
        "total_amount": 34.00,
        "PULocationID": 75,
        "DOLocationID": 235,
    }

    try:
        pred = predict_from_dict(sample_ride)
        print(f"[RESULT] Predicted trip duration: {pred:.2f} minutes")
    except Exception as e:
        print(f"[FAILED] Prediction pipeline could not complete: {e}")
