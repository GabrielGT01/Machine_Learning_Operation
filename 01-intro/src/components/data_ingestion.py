
import os
import sys
import json
import pandas as pd
import numpy as np
import yaml  
from dataclasses import dataclass 
from src.logger import logging
from typing import List
from src.constants.config_entity import DataIngestionConfig, TrainingPipelineConfig
from src.constants.artifact_entity import DataIngestionArtifact
from src.constants import training_pipeline_names  

"""
This class handles:
- assume to read train/test from an external source 
- remove unwanted columns
- Validating column consistency
- saves the data to my machine
"""

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            self._schema_config = self.read_yaml_file(training_pipeline_names.SCHEMA_FILE_PATH)
        except Exception as e:
            raise Exception(f"Error initializing DataIngestion: {e}") 
            
    @staticmethod
    def read_yaml_file(file_path: str) -> dict:
        """
        Static method to read YAML configuration files
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"YAML file not found: {file_path}")
                
            with open(file_path, "r", encoding="utf-8") as yaml_file:  
                content = yaml.safe_load(yaml_file)
                if content is None:
                    raise ValueError(f"YAML file is empty or invalid: {file_path}")
                return content
        except yaml.YAMLError as e:
            raise Exception(f"Error parsing YAML file {file_path}: {e}")
        except Exception as e:
            raise Exception(f"Error reading YAML file {file_path}: {e}")
    
    def read_external_dataframe(self):
        """
        Read data from any data source and apply transformations
        """
        try:
            train_data_link = self.data_ingestion_config.train_data_source
            test_data_link = self.data_ingestion_config.test_data_source
            logging.info('Reading train and test data sources')
            
            # Check if files exist before reading
            if not os.path.exists(train_data_link):
                raise FileNotFoundError(f"Train data file not found: {train_data_link}")
            if not os.path.exists(test_data_link):
                raise FileNotFoundError(f"Test data file not found: {test_data_link}")
            
            # Read parquet files
            train_data = pd.read_parquet(train_data_link)
            test_data = pd.read_parquet(test_data_link)
            
            logging.info(f"Loaded {len(train_data)} training and {len(test_data)} test records")
            print(f"Loaded {len(train_data)} training and {len(test_data)} test records from database")
      
            # Define columns to drop (taxi-specific columns)
            columns_to_drop = [
                'VendorID', 'store_and_fwd_flag', 'RatecodeID', 
                'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 
                'ehail_fee', 'improvement_surcharge', 'payment_type', 
                'trip_type', 'congestion_surcharge', 'cbd_congestion_fee'
            ]
            
            # Check if columns exist before dropping
            existing_cols_train = [col for col in columns_to_drop if col in train_data.columns]
            if existing_cols_train:
                train_data = train_data.drop(columns=existing_cols_train, axis=1)
                logging.info(f"Dropped {len(existing_cols_train)} columns from training data")
            
            existing_cols_test = [col for col in columns_to_drop if col in test_data.columns]
            if existing_cols_test:
                test_data = test_data.drop(columns=existing_cols_test, axis=1)
                logging.info(f"Dropped {len(existing_cols_test)} columns from test data")
                
            # Convert all "na" strings to numpy NaN
            train_data.replace({"na": np.nan}, inplace=True)
            test_data.replace({"na": np.nan}, inplace=True)
            
            logging.info("Data transformation completed successfully")
            return train_data, test_data
            
        except Exception as e:
            logging.error(f"Error in reading and transforming dataframe: {e}")
            raise e

    def validate_column(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> bool:
        """
        Validate that the required columns are present in both datasets
        """
        try:
            required_columns = []  # Fixed variable name
            
            # Extract required columns from schema config
            if "columns" in self._schema_config:
                for column_dict in self._schema_config["columns"]:
                    if isinstance(column_dict, dict):
                        required_columns.extend(column_dict.keys())
                    elif isinstance(column_dict, str):
                        required_columns.append(column_dict)
            else:
                logging.warning("No 'columns' key found in schema config")
                return True  # Skip validation if no schema defined
            
            logging.info(f"Required columns: {required_columns}")
            logging.info(f"Train columns: {list(train_data.columns)}")
            logging.info(f"Test columns: {list(test_data.columns)}")
            
            # Validate train data columns
            train_validation = set(required_columns) == set(train_data.columns)
            if train_validation:
                logging.info("Train Column validation passed.")
            else:
                logging.error("Train Column validation failed.")
                missing_train = set(required_columns) - set(train_data.columns)
                extra_train = set(train_data.columns) - set(required_columns)
                if missing_train:
                    logging.error(f"Missing columns in train data: {missing_train}")
                if extra_train:
                    logging.error(f"Extra columns in train data: {extra_train}")
            
            # Validate test data columns
            test_validation = set(required_columns) == set(test_data.columns)
            if test_validation:
                logging.info("Test Column validation passed.")
            else:
                logging.error("Test Column validation failed.")
                missing_test = set(required_columns) - set(test_data.columns)
                extra_test = set(test_data.columns) - set(required_columns)
                if missing_test:
                    logging.error(f"Missing columns in test data: {missing_test}")
                if extra_test:
                    logging.error(f"Extra columns in test data: {extra_test}")
            
            # Return True only if both validations pass
            return train_validation and test_validation
            
        except Exception as e:
            logging.error(f"Error validating columns: {e}")
            raise e
            
    def save_data_to_machine(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """
        Save the processed data to local machine
        """
        try:
            # Create directory if it doesn't exist
            data_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(data_path, exist_ok=True)
            
            logging.info(f"Created directory: {data_path}")
            
            # Save dataframes as CSV files
            train_data.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_data.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)
            
            logging.info("Dataframes successfully saved to local machine")
            
            return self.data_ingestion_config.training_file_path, self.data_ingestion_config.test_file_path
            
        except Exception as e:
            logging.error(f"Error in saving data to machine: {e}")
            raise e
    
    def start_data_ingestion(self):
        """
        Main method to start the data ingestion process
        """
        try:
            logging.info("Starting data ingestion process")
            
            # Read and transform data
            train_data, test_data = self.read_external_dataframe()

            # Validate columns
            validation_result = self.validate_column(train_data, test_data)
            if not validation_result:
                raise Exception("Column validation failed. Check logs for details.")
            
            # Save data to machine
            train_file_path, test_file_path = self.save_data_to_machine(train_data, test_data)
            
            # Create artifact
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )
            
            logging.info("Data ingestion completed successfully")
            return data_ingestion_artifact
            
        except Exception as e:
            logging.error(f"Error in data ingestion process: {e}")
            raise e
