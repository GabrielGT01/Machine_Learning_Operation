
from datetime import datetime
from airflow.decorators import dag, task
from airflow.sensors.base import PokeReturnValue
import os
import requests
import pandas as pd
from typing import Dict, Any

# Parameters tht can be changed
YEAR = 2025
MONTH = 5  

@dag(
    dag_id="data_engineering",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["nyc-taxi"]
)
def data_engineering():
    
    @task.sensor(poke_interval=30, timeout=300)
    def is_api_available() -> PokeReturnValue:
        """
        checks if the link still works
        """
        url = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
        try:
            resp = requests.get(url, timeout=15)
            print(f"Status: {resp.status_code}")
            return PokeReturnValue(is_done=resp.ok)
        except Exception as e:
            print(f"Request failed: {e}")
            return PokeReturnValue(is_done=False)
    
    @task
    def download_green_taxi_data(year: int, month: int, output_dir: str = "/opt/airflow/data") -> str:
        """
        Download the Green Taxi trip record Parquet file for the given year and month.
        Returns the local destination path.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data"
        month_str = f"{month:02d}"
        filename = f"green_tripdata_{year}-{month_str}.parquet"
        url = f"{base_url}/{filename}"
        dest_path = os.path.join(output_dir, filename)
        
        print(f"Downloading {url} -> {dest_path}")
        
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
        print(f"Saved to {dest_path}")
        return dest_path
    
    @task
    def feature_selection(link: str) -> str:
        """
        Load and preprocess test data from parquet file
        
        Parameters:
        link (str): Path to parquet file
        
        Returns:
        str: Path to cleaned parquet file with the four columns required for training data
        """
        data = pd.read_parquet(link)
        data = data[['lpep_pickup_datetime', 'lpep_dropoff_datetime', 
                     'PULocationID', 'DOLocationID', 'trip_distance']]
        
        # Calculate trip duration in minutes
        data['duration'] = (
            data['lpep_dropoff_datetime'] - data['lpep_pickup_datetime']
        ).dt.total_seconds() / 60
        
        # Select final columns
        data = data[['PULocationID', 'DOLocationID', 'trip_distance', 'duration']]
        
        # Remove outliers - filter duration and distance
        data = data[(data['duration'] >= 1) & (data['duration'] <= 62)]
        data = data[(data['trip_distance'] >= 1) & (data['trip_distance'] <= 20)]
        
        # Convert location IDs to categorical data
        data[['PULocationID', 'DOLocationID']] = (
            data[['PULocationID', 'DOLocationID']].astype('str'))
        
        output_path = "/opt/airflow/data/green_taxi_cleaned.parquet"
        # Save as Parquet
        data.to_parquet(output_path, engine="pyarrow", index=False)
        
        print(f"Cleaned data saved to: {output_path}")
        return output_path
    
    # Define tasks
    api_ok = is_api_available()
    # returns dest_path
    downloaded = download_green_taxi_data.override(task_id="fetch_taxi_data")(YEAR, MONTH)
    
    cleaned_data = feature_selection.override(task_id="clean_taxi_data")(downloaded)
    
    # Set up dependencies
    api_ok >> downloaded >> cleaned_data

# Instantiate the DAG
data_engineering()
