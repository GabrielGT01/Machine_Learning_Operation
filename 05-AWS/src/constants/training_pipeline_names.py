
import os
import sys
import numpy as np
import pandas as pd

"""
Data ingestion related constants, such as the data link, and the downloaded folder path
"""

DATA_INGESTION_TRAIN_LINK: str = "./data/green_tripdata_2025-01.parquet"
DATA_INGESTION_TEST_LINK: str = "./data/green_tripdata_2025-02.parquet" 
DATA_INGESTION_INGESTED_DIR: str = "ingested_data"
SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

"""
Data transformation features, the columns required
"""
numeric_features = ['trip_distance', 'passenger_count','fare_amount','total_amount']
target_encoder_features = ['PULocationID', 'DOLocationID']
TARGET_COLUMN = ['duration']

"""
Model Trainer related constant start with MODEL TRAINER VAR NAME
"""
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "duration_time_model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.7
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD: float = 0.05
