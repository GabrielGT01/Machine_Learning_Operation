
# src/components/data_transformation.py
import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from src.constants.training_pipeline_names import ( 
    numeric_features,
    target_encoder_features,
    TARGET_COLUMN
)
from src.constants.config_entity import (
    DataIngestionConfig,
    TrainingPipelineConfig,
    DataTransformationConfig
)
from src.constants.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from src.logger import logging

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            
        except Exception as e:
            raise e
    
    @staticmethod
    def read_data(filepath) -> pd.DataFrame:
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            raise e
    
    @staticmethod
    def save_numpy_data(file_path: str, array: np.array):
        """
        Save numpy array data to file
        file_path: str location of file to save
        array: np.array data to save
        """
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            with open(file_path, "wb") as file_obj:
                np.save(file_obj, array)
        except Exception as e:
            raise e
    
    @staticmethod
    def save_object(file_path: str, obj: object) -> None:
        """
        this static method saves the preprocessor pkl
        """
        try:
            logging.info("Entered the save_object method class")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as file_obj:
                joblib.dump(obj, file_obj)
            logging.info("Exited the save_object method class")
        except Exception as e:
            raise e
    
    def create_data_transformer(self):
        """
        This function creates and returns the preprocessing object with imputation
        """
        logging.info("creating encoders and imputers")
        try:
            # Create preprocessing pipelines for each feature type
            numeric_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('target_encoder', TargetEncoder())
            ])
            
            logging.info("Initiating encoders and scaling")
            # Create preprocessor using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical_transformer', numeric_pipeline, numeric_features),
                    ('categorical_transformer', categorical_pipeline, target_encoder_features)
                ],
                remainder='drop'
            )
            return preprocessor
        except Exception as e:
            logging.error(f'Could not transform the inputs: {e}')
            raise Exception(f"Error in creating preprocessor: {e}")
    
    def create_processed_data(self, untransformed_data: pd.DataFrame):
        """
        Load and preprocess data from DataFrame
        
        creates the target column
        
        Returns:
        processed dataframe with target column and selected features
        """
        # Calculate trip duration in minutes
        untransformed_data['lpep_pickup_datetime'] = pd.to_datetime(untransformed_data['lpep_pickup_datetime'])
        untransformed_data['lpep_dropoff_datetime'] = pd.to_datetime(untransformed_data['lpep_dropoff_datetime'])
        
        untransformed_data['duration'] = (
            untransformed_data['lpep_dropoff_datetime'] - untransformed_data['lpep_pickup_datetime']
        ).dt.total_seconds() / 60
        
        # Select final columns
        data = untransformed_data[['PULocationID', 'DOLocationID', 'passenger_count','trip_distance','fare_amount','total_amount','duration']]
        	
        
        # Remove outliers - filter duration and distance, duration more than one hour and 20 miles
        data = data[(data['duration'] >= 0) & (data['duration'] <= 60)]
        data = data[(data['trip_distance'] >= 0) & (data['trip_distance'] <= 15)]
        data = data[(data['passenger_count'] >= 0) & (data['passenger_count'] <= 5)]
        
        # Convert location IDs to categorical data
        data[['PULocationID', 'DOLocationID']] = (
            data[['PULocationID', 'DOLocationID']].astype('str')
        )
        return data
        
    def start_data_transformation(self):
        """
        This method initiates the data transformation with imputation
        """
        logging.info("Starting data transformation")
        try:
            # Read training and testing data via the static method function
            train_df = DataTransformation.read_data(self.data_ingestion_artifact.train_file_path)
            test_df = DataTransformation.read_data(self.data_ingestion_artifact.test_file_path)
            logging.info("Read both test and train data successfully")
            
            # Check for missing values before transformation
            logging.info(f"Train data missing values:\n{train_df.isnull().sum()}")
            logging.info(f"Test data missing values:\n{test_df.isnull().sum()}")
            
            train_df = self.create_processed_data(train_df)
            test_df = self.create_processed_data(test_df)
            logging.info("created the duration target column")
            
            # Separate features and target variable
            input_feature_train_df = train_df.drop(columns=TARGET_COLUMN, axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN[0]]  # TARGET_COLUMN is a list
            logging.info("Split the train data into features and target")
            
            input_feature_test_df = test_df.drop(columns=TARGET_COLUMN, axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN[0]]  # TARGET_COLUMN is a list
            logging.info("Split the test data into features and target")
            
            # Get the preprocessor object
            preprocessor = self.create_data_transformer()
            
            # Apply transformations
            logging.info("Applying transformations on training data")
            transformed_input_train_feature = preprocessor.fit_transform(input_feature_train_df, target_feature_train_df)
            logging.info("Applying transformations on test data")
            transformed_input_test_feature = preprocessor.transform(input_feature_test_df)
            
            # Combine transformed features with target variable
            logging.info("concatenate the transformed array")
            train_arr = np.c_[
                transformed_input_train_feature, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                transformed_input_test_feature, np.array(target_feature_test_df)
            ]
            
            logging.info("Transformation completed successfully")
            logging.info(f"Transformed train array shape: {train_arr.shape}")
            logging.info(f"Transformed test array shape: {test_arr.shape}")
            
            # Save the train, test and transformation file
            logging.info("saving the train, test and transformer")
            DataTransformation.save_numpy_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            DataTransformation.save_numpy_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            DataTransformation.save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            
            # Prepare data transformation artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
            
        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise Exception(f"Data transformation failed: {e}")
