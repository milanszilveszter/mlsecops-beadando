from flask import Flask, request
from flask_restx import Api, Resource, fields
from werkzeug.datastructures import FileStorage
import os
import pandas as pd
from MLModel import MLModel
import mlflow
from mlflow import MlflowClient
from datetime import datetime
from mlflow.exceptions import MlflowException

# Configure MLflow tracking
mlflow.set_tracking_uri("http://mlflow:5102")
experiment_name = "default_experiment"
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

app = Flask(__name__)
api = Api(app, version='1.0', title='API Documentation',
          description='Machine Learning Model Management API')

@app.route("/hello")
def hello():
    """Health check endpoint to verify service availability.
    
    Returns:
        str: Simple confirmation message
    """
    return "Flask is running!"

@app.before_request
def log_every_request():
    """Log details for every incoming request.
    
    Prints:
        Request method, path, and client IP address
    """
    print(f"📥 {request.method} {request.path} from {request.remote_addr}")

# Initialize MLflow client and model
client = MlflowClient()

try:
    obj_mlmodel = MLModel(client=client)
    if obj_mlmodel.model is None:
        print("⚠️  Warning: No 'Staging' model found. Training is still possible.")
except Exception as e:
    print(f"⚠️  Warning: Could not load 'Staging' model. Error: {e}")
    obj_mlmodel = MLModel(client=client)

# API Models
predict_model = api.model('PredictModel', {
    'inference_row': fields.List(fields.Raw, required=True,
                                 description='Input data row for prediction')
})

file_upload = api.parser()
file_upload.add_argument('file', location='files',
                         type=FileStorage, required=True,
                         help='CSV training data file')

ns = api.namespace('model', description='Model training and prediction operations')

@ns.route('/train')
class Train(Resource):
    @ns.expect(file_upload)
    def post(self):
        """Train and register new model version.
        
        Processes uploaded CSV data, trains model, logs metrics to MLflow,
        and transitions new model version to Staging.
        
        Returns:
            tuple: Response JSON with training results (200),
            error messages with appropriate status codes (400, 500)
        """
        args = file_upload.parse_args()
        uploaded_file = args['file']
        
        if os.path.splitext(uploaded_file.filename)[1] != '.csv':
            return {'error': 'Invalid file type'}, 400
        
        try:
            data_path = 'temp_crop_data.csv'
            uploaded_file.save(data_path)
            run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_name = "RandomForest"
            
            with mlflow.start_run(run_name=run_name) as run:
                df = pd.read_csv(data_path)
                input_example = df.drop(columns="label").iloc[:1]
                signature = mlflow.models.infer_signature(df.drop(columns="label"), df["label"])
                
                df = obj_mlmodel.preprocessing_pipeline(df)
                mlflow.log_artifact(data_path, "datasets")
                
                train_acc, test_acc, model = obj_mlmodel.train_and_save_model(df)
                
                mlflow.log_metrics({
                    "train_accuracy": train_acc,
                    "test_accuracy": test_acc
                })

                mlflow.sklearn.log_model(
                    sk_model=model, 
                    artifact_path="model", 
                    input_example=input_example, 
                    signature=signature
                )
                
                model_uri = f"runs:/{run.info.run_id}/model"
                model_version = mlflow.register_model(model_uri, model_name)
                
                client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage="Staging"
                )
                
                os.remove(data_path)
                return {
                    'message': 'Model trained and deployed to Staging',
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc
                }, 200
                
        except MlflowException as mfe:
            return {'message': 'MLflow operation failed', 'error': str(mfe)}, 500
        except Exception as e:
            return {'message': 'Training process failed', 'error': str(e)}, 500

@ns.route('/predict')
class Predict(Resource):
    @api.expect(predict_model)
    def post(self):
        """Make prediction using current staging model.
        
        Requires:
            JSON payload with 'inference_row' containing input features
            
        Returns:
            tuple: Prediction result (200),
            error messages with appropriate status codes (400, 500)
        """
        try:
            data = request.get_json()
            if 'inference_row' not in data:
                return {'error': 'Missing required inference_row'}, 400
                
            if not data['inference_row']:
                return {'error': 'Empty inference_row'}, 400
                
            if not obj_mlmodel.model:
                return {'error': 'No deployed model available'}, 404
                
            run_name = f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with mlflow.start_run(run_name=run_name):
                prediction = obj_mlmodel.predict(data['inference_row'])
                mlflow.log_params({
                    "input": data['inference_row'],
                    "output": prediction
                })
                
            return {'prediction': str(prediction)}, 200
            
        except Exception as e:
            return {'message': 'Prediction failed', 'error': str(e)}, 500

if __name__ == '__main__':
    """Start Flask development server.
    
    Configures:
        Host: 0.0.0.0 (public access)
        Port: 8080
        Debug mode: Off (production setting)
    """
    print("Starting model service...")
    app.run(host='0.0.0.0', port=8080, debug=False)
