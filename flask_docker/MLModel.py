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
        Initialize the MLModel with the given MLflow client and 
        load the staging model if available.

        Parameters:
            client (MlflowClient): The MLflow client used to 
            interact with the MLflow registry.

        Attributes:
            model (object): The loaded model, or None if no model 
                is loaded.
            fill_values_nominal (dict): Dictionary of fill values 
                for nominal columns.
            fill_values_discrete (dict): Dictionary of fill values 
                for discrete columns.
            fill_values_continuous (dict): Dictionary of fill values 
                for continuous columns.
            min_max_scaler_dict (dict): Dictionary of MinMaxScaler objects 
                for continuous columns.
            onehot_encoders (dict): Dictionary of OneHotEncoder objects 
                for nominal columns.
        """
        self.client = client
        self.model = None
        self.standard_scaler_dict = None
        self.load_staging_model()

    def load_staging_model(self):
        """
        Load the latest model tagged with 'Staging' stage from MLflow 
        if available.
        
        If a model with the 'Staging' tag exists, it loads the model 
        and associated artifacts. Otherwise, prints a warning.

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
        Load necessary artifacts (e.g., scalers, encoders) from the given 
        artifact URI.

        Parameters:
            artifact_uri (str): The URI of the artifact directory containing 
            necessary files.

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
        Predicts the outcome based on the input data row.

        This method applies the preprocessing pipeline to the input data, performs necessary
        transformations, and uses the preloaded model to make a prediction. The 'V24' column
        is removed from the data frame as part of the preprocessing steps. If an error occurs
        during the prediction process, it catches the exception and returns a JSON object with
        the error message and a 500 status code.

        Parameters:
        - inference_row: A single row of input data meant for prediction. Expected to be a list or
        a series that matches the format and order expected by the preprocessing pipeline and model.

        Returns:
        - On success: Returns the prediction as an integer.
        - On failure: Returns a JSON response object with an error message and a 500 status code.

        Notes:
        - Ensure that the input data row is in the correct format and contains the expected features
        excluding 'V24', which is not required and will be removed during preprocessing.
        - The method is wrapped in a try-except block to handle unexpected errors during prediction.
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
        """Preprocess the data to handle missing values,
        create new features, encode categorical features, 
        and normalize the data using min max scaling.
        Returns the preprocessed dataframe.
        
        Keyword arguments:
        df -- DataFrame with the data

        Returns:
        df -- DataFrame with the preprocessed data
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
        """Preprocess the inference row to match
        the features we created for training data.
        Returns the preprocessed dataframe for inference.
        
        Keyword arguments:
        sample_data -- Pandas series with the inference data

        Returns:
        input_df -- DataFrame with the preprocessed inference data
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
        Calculate and print the accuracy of the model on both the training and test data sets.
        
        Args:
            X_train: Features for the training set.
            X_test: Features for the test set.
            y_train: Actual labels for the training set.
            y_test: Actual labels for the test set.

        Returns:
            A tuple containing the training accuracy and the test accuracy.
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
        Calculate and print the overall accuracy of the model using a data set.

        Args:
            X: Features for the data set.
            y: Actual labels for the data set.

        Returns:
            The accuracy of the model on the provided data set.
        """
        y_pred = self.model.predict(X)

        accuracy = accuracy_score(y, y_pred)

        print("Accuracy: ", accuracy)

        return accuracy

    def train_and_save_model(self, df):
        """Train the model and save it to a file. 
        Returns the train and test accuracy.
        
        Keyword arguments:
        df -- DataFrame with the preprocessed data

        Returns:
        train_accuracy -- Accuracy of the model on the training set
        test_accuracy -- Accuracy of the model on the test set
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