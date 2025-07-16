

from datetime import datetime
import os
from src.constants import training_pipeline_names




class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.data_artifact_folder = training_pipeline_names.DATA_INGESTION_INGESTED_DIR
        self.data_folder = os.path.join(self.data_artifact_folder, f"untransformed_data-{timestamp}")  
        self.timestamp: str = timestamp

class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.train_data_source: str = training_pipeline_names.DATA_INGESTION_TRAIN_LINK
        self.test_data_source: str = training_pipeline_names.DATA_INGESTION_TEST_LINK
        self.training_file_path: str = os.path.join(training_pipeline_config.data_folder, "train_data.csv") 
        self.test_file_path: str = os.path.join(training_pipeline_config.data_folder, "test_data.csv")  

class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.timestamp: str = timestamp
        self.transformed_folder = os.path.join(training_pipeline_config.data_artifact_folder, f'transformed_data-{timestamp}')
        self.model_path = os.path.join(training_pipeline_config.data_artifact_folder, f'model-{timestamp}')
        self.transformed_train_file_path: str = os.path.join(self.transformed_folder, "train.npy")
        self.transformed_test_file_path: str = os.path.join(self.transformed_folder, "test.npy")
        self.transformed_object_file_path: str = os.path.join(self.model_path, "preprocessing.pkl")

class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.timestamp: str = timestamp
        self.model_trainer_dir: str = os.path.join(training_pipeline_config.data_artifact_folder, f'model-{timestamp}')
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir, training_pipeline_names.MODEL_TRAINER_TRAINED_MODEL_NAME
        )
