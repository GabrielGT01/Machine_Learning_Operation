# main.py

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass 
from src.logger import logging
from typing import List
from src.constants.config_entity import (
    DataIngestionConfig, 
    TrainingPipelineConfig, 
    DataTransformationConfig,
    ModelTrainerConfig
)
from src.constants.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.ModelTrainer import ModelTrainer


if __name__ == "__main__":
    try:
        # Initialize pipeline configuration
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        
        # Create data ingestion instance
        data_ingestion = DataIngestion(data_ingestion_config)
        
        logging.info("Initiating data reading and processing")
        
        # Start data ingestion
        data_ingestion_artifact = data_ingestion.start_data_ingestion()
        
        print("Data ingestion completed successfully!")
        print(f"Train file: {data_ingestion_artifact.train_file_path}")
        print(f"Test file: {data_ingestion_artifact.test_file_path}")
        
        logging.info("Transforming data")
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        
        # Create data transformation instance
        data_transform = DataTransformation(data_ingestion_artifact, data_transformation_config)
        data_transformation_artifact = data_transform.start_data_transformation()
        
        print("Data transformation completed successfully!")
        print(f"Preprocessor file: {data_transformation_artifact.transformed_object_file_path}")
        print(f"Transformed train file: {data_transformation_artifact.transformed_train_file_path}")
        print(f"Transformed test file: {data_transformation_artifact.transformed_test_file_path}")
        logging.info("Transforming ended")
        
        logging.info("model training")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        # Create model training instance
        model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
        model_trainer_artifact = model_trainer.start_model_trainer()
        
        print("Model successfully trained!")
        logging.info("Model training ended")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise e
