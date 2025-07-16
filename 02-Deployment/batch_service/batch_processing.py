
import os
import uuid
import argparse
import shutil

import pandas as pd
import joblib
import mlflow


def generate_uuids(n):
    return [str(uuid.uuid4()) for _ in range(n)]


def main(data_path: str, run_id: str):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    #  Step 1: Load data 
    print(f"[INFO] Reading data from: {data_path}")
    df = pd.read_parquet(data_path)

    df = df[['lpep_pickup_datetime', 'lpep_dropoff_datetime', 'PULocationID', 'DOLocationID', 'trip_distance']].copy()

    # Calculate duration
    df['duration'] = df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
    df['duration'] = df['duration'].dt.total_seconds() / 60

    # Filter
    df = df[(df['duration'] >= 1) & (df['duration'] <= 62)]
    df = df[(df['trip_distance'] >= 1) & (df['trip_distance'] <= 20)]

    # Convert location IDs to string
    df[['PULocationID', 'DOLocationID']] = df[['PULocationID', 'DOLocationID']].astype(str)

    df = df[['PULocationID', 'DOLocationID', 'trip_distance', 'duration']]

    #Step 2: Load Preprocessor 
    preprocessor_uri = f"mlflow-artifacts:/1/{run_id}/artifacts/preprocessor/preprocessing.pkl"
    target_folder = "./artifacts"
    os.makedirs(target_folder, exist_ok=True)

    print("Downloading preprocessor...")
    temp_path = mlflow.artifacts.download_artifacts(artifact_uri=preprocessor_uri)
    destination_path = os.path.join(target_folder, "preprocessing.pkl")
    shutil.copy(temp_path, destination_path)
    preprocessor = joblib.load(destination_path)
    print("Preprocessor loaded.")

    #Step 3: Load Model
    logged_model = f"runs:/{run_id}/nyc-duration-model"
    print(f"Loading model from: {logged_model}")
    model = mlflow.pyfunc.load_model(logged_model)

    #Step 4: Transform and Predict
    features = df[['PULocationID', 'DOLocationID', 'trip_distance']]
    transformed = preprocessor.transform(features)
    predictions = model.predict(pd.DataFrame(transformed))

    #Step 5: create result
    df_result = df.copy()
    df_result['predicted_duration'] = predictions
    df_result['difference'] = df_result['duration'] - df_result['predicted_duration']
    df_result['ride_id'] = generate_uuids(len(df_result))
    df_result['model_version'] = run_id

    # Step 6: Save to CSV
    df_result.to_csv("predictions_output.csv", index=False)
    print(" Batch prediction complete. Output saved to: predictions_output.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch predictor for ride duration")
    parser.add_argument("--data_path", required=True, help="Path to the Parquet data file")
    parser.add_argument("--run_id", required=True, help="MLflow run ID for model and preprocessor")
    args = parser.parse_args()

    main(data_path=args.data_path, run_id=args.run_id)
