#MLflow setup:
#MLflow setup:
#- tracking server: yes, local server
#- backend store: sqlite database, this houses meta data, metrics, params
#- artifacts store: local filesystem
#To run this example you need to launch the mlflow server locally by running the following command in your terminal:
#lsof -ti :5000 | xargs kill -9 lsof -ti :8000 | xargs kill -9

##run this in terminal but in the folder 

#mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

#export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"



import os
import sys
import numpy as np
import joblib
from src.constants.training_pipeline_names import MODEL_TRAINER_EXPECTED_SCORE, MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD
from src.constants.config_entity import (
    DataTransformationConfig,
    ModelTrainerConfig
)
from src.constants.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    RegressionMetricArtifact
)
from src.logger import logging
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from mlflow.models import infer_signature
import mlflow

def load_object(file_path: str):
    """Load object from file"""
    try:
        with open(file_path, "rb") as file_obj:
            return joblib.load(file_obj)
    except Exception as e:
        raise e

def load_numpy_array_data(file_path: str) -> np.array:
    """Load numpy array data from file"""
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise e

def evaluate_model(y_true, y_pred):
    """Evaluate model performance"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, 
                 data_transformation_artifact: DataTransformationArtifact):
        
        try:
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            mlflow.set_tag('development','duration model')
            logging.info("Starting ModelTrainer initialization and setting up mlflow")
            print(f"tracking URI: '{mlflow.get_tracking_uri()}'")
            logging.info(f"tracking URI: '{mlflow.get_tracking_uri()}'")
            
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            logging.error(f"Error in ModelTrainer initialization: {e}")
            raise e

    def _ensure_mlflow_run_ended(self):
        """Safely end any active MLflow run"""
        try:
            if mlflow.active_run():
                logging.info("Active MLflow run detected, ending it safely")
                mlflow.end_run()
                logging.info("Previous MLflow run ended successfully")
        except Exception as e:
            logging.warning(f"Error ending MLflow run: {e}")
            # Continue execution as this is not critical

    def _save_model_safely(self, model, model_dir_path: str, model_filename: str = 'my_model.ubj'):
        """
        Safely save XGBoost model with proper error handling and path management
        
        Args:
            model: Trained XGBoost model
            model_dir_path: Base directory path from config
            model_filename: Name of the model file (default: 'my_model.ubj')
            
        Returns:
            str: Full path where model was saved
        """
        try:
            # Option 3: Most robust - ensure directory exists and create full path
            model_dir = os.path.dirname(model_dir_path)
            
            # Ensure the directory exists
            os.makedirs(model_dir, exist_ok=True)
            logging.info(f"Model directory created/verified: {model_dir}")
            
            # Create full model path
            model_path = os.path.join(model_dir, model_filename)
            
            # Validate path before saving
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
                
            # Save the model with error handling
            logging.info(f"Attempting to save model to: {model_path}")
            
            try:
                model.save_model(model_path)
                logging.info(f"Model saved successfully to: {model_path}")
                
                # Verify the file was actually created
                if os.path.exists(model_path):
                    file_size = os.path.getsize(model_path)
                    logging.info(f"Model file verified - Size: {file_size} bytes")
                else:
                    raise FileNotFoundError(f"Model file was not created: {model_path}")
                    
                return model_path
                
            except Exception as e:
                logging.error(f"Failed to save model: {e}")
                raise e
                
        except Exception as e:
            logging.error(f"Error in _save_model_safely: {e}")
            raise e

    def train_model(self, X_train, y_train, X_test, y_test):
        
        try:
            # Safely end any active MLflow run
            self._ensure_mlflow_run_ended()
            
            mlflow.set_experiment("new-york-taxi-drive-duration-train-January")
            with mlflow.start_run() as run:
                results = [] 
                logging.info("Starting model training process")
                logging.info(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
                logging.info(f"Test data shape: X_test={X_test.shape}, y_test={y_test.shape}")
    
                logging.info("Loading the Xgboost model")
                
                # Train model
                model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
                model.fit(X_train, y_train)
                logging.info("Making predictions with the model")
                
                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                logging.info("Calculating regression metrics")
                train_mae, train_rmse, train_r2 = evaluate_model(y_train, y_train_pred)
                test_mae, test_rmse, test_r2 = evaluate_model(y_test, y_test_pred)
                
                ##signature for mlflow
                signature = infer_signature(X_test, y_test_pred)
                
                # Create metric artifacts
                regression_train_metric = RegressionMetricArtifact(
                    mean_absolute_error=train_mae,
                    root_mean_squared_error=train_rmse,
                    r2_score=train_r2
                )
                
                regression_test_metric = RegressionMetricArtifact(
                    mean_absolute_error=test_mae,
                    root_mean_squared_error=test_rmse,
                    r2_score=test_r2
                )
                
                # Append results
                results.append({
                    'Model': "XGBOOST",
                    'Train_MAE': train_mae,
                    'Train_RMSE': train_rmse,
                    'Train_R2': train_r2,
                    'Test_MAE': test_mae,
                    'Test_RMSE': test_rmse,
                    'Test_R2': test_r2,
                    'Overfit_Check': train_r2 - test_r2  
                })
    
                logging.info(f"Training metrics: {train_mae, train_rmse, train_r2}")
                logging.info(f"Test metrics: {test_mae, test_rmse, test_r2}")
                logging.info("logging MLflow metrics and params")
                
                params = {
                    "model_name": "XGBOOST", 
                    "training_data_shape": str(X_train.shape),
                    "test_data_shape": str(X_test.shape), 
                    "n_estimators": 100, 
                    "max_depth": 3, 
                    "learning_rate": 0.1
                }
                mlflow.log_params(params)
            
                mlflow.log_metrics({
                    'Train_MAE': train_mae,
                    'Train_RMSE': train_rmse,
                    'Train_R2': train_r2,
                    'Test_MAE': test_mae,
                    'Test_RMSE': test_rmse,
                    'Test_R2': test_r2,
                    'Overfit_Check': train_r2 - test_r2
                })

                # log and save model
                model_name = "nyc-taxi-duration-model-January"
                registered_model_name="XGBoostdurationModel"
                
                
                if test_r2 > MODEL_TRAINER_EXPECTED_SCORE:
                    try:
                        mlflow.xgboost.log_model(
                            xgb_model=model,
                            name=model_name,
                            registered_model_name=registered_model_name,
                            input_example=X_test[:5],
                            signature=signature,
                            model_format="ubj"
                        )
                        logging.info(f"Model registered as: {registered_model_name}")
                        print(f"Model registered as: {registered_model_name}")
                        print(f"Run ID: {run.info.run_id}")
                    except Exception as e:
                        logging.error(f"Failed to register model in MLflow: {e}")
                        # Continue execution, just log the model without registration
                        mlflow.xgboost.log_model(model, artifact_path="nyc-duration-model")
                        logging.info("Model logged without registration due to error")
                else:
                    # Just log without registering if performance is poor
                    #mlflow.xgboost.log_model(model, artifact_path="nyc-duration-model")
                    mlflow.xgboost.log_model(model, name=model_name)                        
                    logging.info("Model not registered due to poor performance")
                    
                # Load preprocessor to save to mlflow as artifact
                try:
                    preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
                    logging.info("Preprocessor loaded successfully")
                    mlflow.log_artifact(self.data_transformation_artifact.transformed_object_file_path, artifact_path="preprocessor")
                    logging.info("Preprocessor logged to MLflow successfully")
                except Exception as e:
                    logging.error(f"Failed to load or log preprocessor: {e}")
                    # Continue execution as this is not critical for model saving
    
                # Save the trained model using the robust method
                logging.info("Starting model save process")
                try:
                    saved_model_path = self._save_model_safely(
                        model=model,
                        model_dir_path=self.model_trainer_config.trained_model_file_path,
                        model_filename='my_model.ubj'
                    )
                    logging.info(f"Model saved successfully at: {saved_model_path}")
                    
                    # Update the artifact path to the actual saved path
                    actual_model_path = saved_model_path
                    
                except Exception as e:
                    logging.error(f"Critical error: Failed to save model: {e}")
                    raise e
    
                # Model performance evaluation
                if results[-1]['Test_R2'] > MODEL_TRAINER_EXPECTED_SCORE:
                    remark = "good score"
                else:
                    remark = "bad"
    
                # Create Model Trainer Artifact with the actual saved path
                model_trainer_artifact = ModelTrainerArtifact(
                    trained_model_file_path=actual_model_path,
                    train_metric_artifact=regression_train_metric,
                    test_metric_artifact=regression_test_metric,
                    model_performance=remark,
                    over_fitting_under_fitting=results[-1]['Overfit_Check']
                )
                
                logging.info(f"Model trainer artifact created: {model_trainer_artifact}")
                logging.info("Model training process completed successfully")
    
                print("model file path:")
                print(model_trainer_artifact.trained_model_file_path)
                print('---' * 20)
                print("Training Metrics:")
                print(model_trainer_artifact.train_metric_artifact)
                print('---' * 20)
                print("Test Metrics:")
                print(model_trainer_artifact.test_metric_artifact)
                print('---' * 20)
                print("Over_fitting_under_fitting_Metrics:")
                print(model_trainer_artifact.over_fitting_under_fitting)
                print('---' * 20)
                print("Training performance:")
                print(model_trainer_artifact.model_performance)
                
                return model_trainer_artifact
            
        except Exception as e:
            logging.error(f"Error in train_model method: {e}")
            # Ensure MLflow run is ended even if there's an error
            self._ensure_mlflow_run_ended()
            raise e

    def start_model_trainer(self) -> ModelTrainerArtifact:
        
        try:
            logging.info("Starting model trainer pipeline")
            
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path 
            
            logging.info(f"Loading training data from: {train_file_path}")
            logging.info(f"Loading test data from: {test_file_path}")
            
            # Loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)
            
            logging.info(f"Training array shape: {train_arr.shape}")
            logging.info(f"Test array shape: {test_arr.shape}")
            
            # Split features and target
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )
            
            logging.info("Data split completed - features and target separated")
            logging.info(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
            logging.info(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
            
            # Train model with all required parameters
            logging.info("Calling train_model method")
            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            
            logging.info("Model trainer pipeline completed successfully")
            return model_trainer_artifact
            
        except Exception as e:
            logging.error(f"Error in start_model_trainer method: {e}")
            # Ensure any active MLflow runs are ended
            self._ensure_mlflow_run_ended()
            raise e
