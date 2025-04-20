from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import yaml
import numpy as np
from preprocessing import DataPreprocessor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load configuration and initialize preprocessor
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    preprocessor = DataPreprocessor()
    logger.info("Successfully loaded config and initialized preprocessor")
except Exception as e:
    logger.error(f"Error loading config: {e}")
    raise

# Load all models
models = {}
try:
    models['neural_network'] = tf.keras.models.load_model('models/neural_network_model')
    models['xgboost'] = joblib.load('models/xgboost_model.joblib')
    models['random_forest'] = joblib.load('models/random_forest_model.joblib')
    models['svm'] = joblib.load('models/svm_model.joblib')
    logger.info(f"Successfully loaded models: {list(models.keys())}")
except Exception as e:
    logger.error(f"Error loading models: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for making predictions."""
    try:
        data = request.get_json()
        logger.info(f"Received prediction request with data: {data}")
        
        # Validate input features
        required_features = config['data']['feature_columns']
        if not all(feature in data for feature in required_features):
            error_msg = f'Missing features. Required features: {required_features}'
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 400
            
        # Prepare input data
        input_data = np.array([[data[feature] for feature in required_features]])
        input_scaled = preprocessor.scaler.transform(input_data)
        
        # Make predictions with all models
        predictions = {}
        for model_name, model in models.items():
            if model_name == 'neural_network':
                pred = float((model.predict(input_scaled) > 0.5)[0][0])
            else:
                pred = float(model.predict(input_scaled)[0])
            predictions[model_name] = pred
            logger.info(f"{model_name} prediction: {pred}")
            
        # Calculate ensemble prediction (majority voting)
        ensemble_pred = int(sum(predictions.values()) > len(predictions)/2)
        predictions['ensemble'] = ensemble_pred
        
        response = {
            'predictions': predictions,
            'leak_probability': float(sum(predictions.values()) / len(predictions))
        }
        logger.info(f"Returning predictions: {response}")
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Error making prediction: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    health_status = {
        'status': 'healthy' if models else 'degraded',
        'models_loaded': list(models.keys()),
        'config_loaded': bool(config),
        'preprocessor_initialized': bool(preprocessor)
    }
    return jsonify(health_status)

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000, debug=True)