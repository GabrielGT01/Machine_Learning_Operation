import os
import uuid
import argparse
import pandas as pd
import joblib
from xgboost import XGBRegressor

def generate_uuids(n):
    return [str(uuid.uuid4()) for _ in range(n)]


def main(data_path: str):
    # Step 1: Load data 
    print(f"[INFO] Reading data from: {data_path}")
    df = pd.read_parquet(data_path)

    # Select required columns
    df = df[['lpep_pickup_datetime', 'lpep_dropoff_datetime',
             'PULocationID', 'DOLocationID',
             'trip_distance', 'fare_amount',
             'total_amount', 'passenger_count']].copy()

    # Calculate duration in minutes
    df['duration'] = df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
    df['duration'] = df['duration'].dt.total_seconds() / 60

    # Filter outliers
    df = df[(df['duration'] >= 0) & (df['duration'] <= 60)]
    df = df[(df['trip_distance'] >= 0) & (df['trip_distance'] <= 20)]
    df = df[(df['passenger_count'] >= 0) & (df['passenger_count'] <= 5)]

    # Convert location IDs to string
    df[['PULocationID', 'DOLocationID']] = df[['PULocationID', 'DOLocationID']].astype(str)

    # Final columns
    df = df[['passenger_count', 'trip_distance', 'fare_amount',
             'total_amount', 'PULocationID', 'DOLocationID', 'duration']]

    # Step 2: Load preprocessor and model (from local files)
    preprocessor = joblib.load("preprocessing.pkl")
    model = XGBRegressor()
    model = model.load_model("my_model.ubj") 
    print("[INFO] Preprocessor and Model loaded.")

    preprocessor = joblib.load("preprocessing.pkl")
    model = XGBRegressor()
    model.load_model("my_model.ubj")           # ✅ loads into `model`
    print("[INFO] Preprocessor and Model loaded.")

    # Step 3: Transform and Predict
    features = df[['passenger_count', 'trip_distance', 'fare_amount',
                   'total_amount', 'PULocationID', 'DOLocationID']]
    transformed = preprocessor.transform(features)
    predictions = model.predict(transformed) 

    # Step 4: Create result
    df_result = df.copy()
    df_result['predicted_duration'] = predictions
    df_result['difference'] = df_result['duration'] - df_result['predicted_duration']
    df_result['ride_id'] = generate_uuids(len(df_result))

    # Step 5: Save to CSV
    df_result.to_csv("predictions_output.csv", index=False)
    print("Batch prediction complete. Output saved to: predictions_output.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch predictor for ride duration")
    parser.add_argument("--data_path", required=True, help="Path to the Parquet data file")
    args = parser.parse_args()

    main(data_path=args.data_path)
