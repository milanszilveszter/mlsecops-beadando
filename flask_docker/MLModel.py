import json
import pickle
import pandas as pd
from flask import jsonify
from constants import CONTINOUS_COLUMNS
import numpy as np
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import mlflow
from mlflow.artifacts import download_artifacts
import numpy as np
import os

class MLModel:
    def __init__(self, client):
        """
        Initialize the MLModel with an MLflow client and attempt to load the latest staging model.

        Parameters:
            client (MlflowClient): The MLflow client instance used to interact with the MLflow model registry.

        Attributes:
            model (object or None): The loaded ML model, if available.
            standard_scaler_dict (dict or None): Dictionary of StandardScaler objects for continuous columns.
        """
        self.client = client
        self.model = None
        self.standard_scaler_dict = None
        self.load_staging_model()

    def load_staging_model(self):
        """
        Load the latest model version tagged as 'Staging' from the MLflow registry.

        If a staging model exists, loads the model and its associated artifacts.
        Prints a warning if no staging model is found.

        Returns:
            None
        """
        try:
            latest_staging_model = None
            for model in self.client.search_registered_models():
                for latest_version in model.latest_versions:
                    if latest_version.current_stage == "Staging":
                        latest_staging_model = latest_version
                        break
                if latest_staging_model:
                    break
            
            if latest_staging_model:
                model_uri = latest_staging_model.source
                self.model = mlflow.sklearn.load_model(model_uri)
                print("Staging model loaded successfully.")
                
                # Load associated artifacts
                artifact_uri = latest_staging_model.source.rpartition('/')[0]
                self.load_artifacts(artifact_uri)
            else:
                print("No staging model found.")
                
        except Exception as e:
            print(f"Error loading model or artifacts: {e}")

    def load_artifacts(self, artifact_uri):
        """
        Load necessary artifacts (e.g., scalers, encoders) from the specified artifact URI.

        Parameters:
            artifact_uri (str): The URI of the artifact directory containing required files.

        Returns:
            None
        """
        try:
            # Load StandardScaler dictionary
            scaler_path = download_artifacts(artifact_uri=f"""
                    {artifact_uri}/standard_scaler_dict.pkl""")
            with open(scaler_path, 'rb') as f:
                self.standard_scaler_dict = pickle.load(f)

            print("Artifacts loaded successfully.")

        except Exception as e:
            print(f"Error loading artifacts: {e}")
    
    def predict(self, inference_row):
        """
        Generate a prediction for a single input row using the loaded model.

        This method preprocesses the input, applies the trained model, and returns the prediction.
        Handles errors gracefully and returns a JSON error response if prediction fails.

        Parameters:
            inference_row (list or pd.Series): The input data for prediction. Must match the expected feature order.

        Returns:
            str: The model's prediction as a string if successful.
            Response: A Flask JSON response with error details and status code 500 if prediction fails.
        """
        try:
            infer_array = pd.Series(inference_row, dtype=str)
            infer_df = infer_array.to_frame().T
            infer_df.columns = ["V" + str(i) for i in range(1, 8)]

            df = self.preprocessing_pipeline_inference(infer_df)

            y_pred = self.model.predict(df)

            return str(y_pred)

        except Exception as e:
            return jsonify({'message': 'Internal Server Error. ',
                        'error': str(e)}), 500
            
    def preprocessing_pipeline(self, df):
        """
        Preprocess the input DataFrame for training.

        Handles missing values, feature creation, categorical encoding, and normalization using StandardScaler.
        Saves the fitted scalers as artifacts for later use.

        Parameters:
            df (pd.DataFrame): The raw input data.

        Returns:
            pd.DataFrame: The preprocessed DataFrame ready for model training.
        """
        list_column_names = ["V" + str(i) for i in range(1, 9)]
        df.columns = list_column_names

        folder = 'artifacts/encoders'
        MLModel.create_new_folder(folder)

        folder = 'artifacts/preprocessed_data'
        MLModel.create_new_folder(folder)

        folder = 'artifacts/models'
        MLModel.create_new_folder(folder)
        
        self.standard_scaler_dict = {}
        for col in CONTINOUS_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].astype(float)
            
            ss = StandardScaler()
            df[col] = ss.fit_transform(df[[col]])
            self.standard_scaler_dict[col] = ss
        
        # Log artifacts to MLflow
        mlflow.log_dict(self.standard_scaler_dict, 
                        "standard_scaler_dict.json")

        # Serialize and log scalers and encoders
        with open("standard_scaler_dict.pkl", "wb") as f:
            pickle.dump(self.standard_scaler_dict, f)
        mlflow.log_artifact("standard_scaler_dict.pkl")

        return df
    
    def preprocessing_pipeline_inference(self, sample_data):
        """
        Preprocess a single inference row to match the training data's feature transformations.

        Converts data types and applies the stored StandardScaler transformations.
        Drops the 'V8' column if present.

        Parameters:
            sample_data (pd.DataFrame): The input data for inference.

        Returns:
            pd.DataFrame: The preprocessed inference data.
        """
        for col in CONTINOUS_COLUMNS:  
            sample_data[col] = pd.to_numeric(sample_data[col], errors='coerce')
            sample_data[col] = sample_data[col].astype(float) 
        
        for col, scaler in self.standard_scaler_dict.items():
            if col in sample_data.columns:
                sample_data[col] = scaler.transform(sample_data[[col]])
        
        if 'V8' in sample_data.columns:
            sample_data = sample_data.drop(columns=['V8'])

        return sample_data
    
    def get_accuracy(self, X_train, X_test, y_train, y_test):
        """
        Calculate and print the accuracy of the model on both training and test datasets.

        Parameters:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Test features.
            y_train (pd.Series): Training labels.
            y_test (pd.Series): Test labels.

        Returns:
            tuple: (train_accuracy, test_accuracy)
        """
        y_train_pred = self.model.predict(X_train)

        y_test_pred = self.model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        print("Train Accuracy: ", train_accuracy)
        print("Test Accuracy: ", test_accuracy)

        return train_accuracy, test_accuracy
    
    def get_accuracy_full(self, X, y):
        """
        Calculate and print the accuracy of the model on the provided dataset.

        Parameters:
            X (pd.DataFrame): Features.
            y (pd.Series): Labels.

        Returns:
            float: The accuracy score.
        """
        y_pred = self.model.predict(X)

        accuracy = accuracy_score(y, y_pred)

        print("Accuracy: ", accuracy)

        return accuracy

    def train_and_save_model(self, df):
        """
        Train a RandomForestClassifier on the provided DataFrame and return accuracy metrics.

        Splits the data into training and test sets, fits the model, and updates the internal model attribute.

        Parameters:
            df (pd.DataFrame): The preprocessed data including features and target.

        Returns:
            tuple: (train_accuracy, test_accuracy, trained_model)
        """
        y = df["V8"]
        X = df.drop(columns="V8")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X_train, y_train)

        self.model = rf

        train_accuracy, test_accuracy = self.get_accuracy(X_train, X_test, y_train, y_test)

        return train_accuracy, test_accuracy, rf
    
    @staticmethod
    def create_new_folder(folder):
        """Create a new folder if it doesn't exist.
        
        Keyword arguments:
        folder -- Path to the folder

        Returns:
        None
        """
        Path(folder).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def save_model(model, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)

    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model