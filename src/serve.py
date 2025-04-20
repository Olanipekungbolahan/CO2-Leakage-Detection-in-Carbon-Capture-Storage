from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import yaml
import numpy as np
from preprocessing import DataPreprocessor
import logging
import gc
import os
import threading
import queue

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global state
models = {}
tflite_interpreter = None
preprocessor = None
config = None
model_queue = queue.Queue()
model_loading_thread = None

def background_model_loader():
    """Background thread to load models asynchronously"""
    while True:
        model_name = model_queue.get()
        if model_name is None:
            break
        try:
            if model_name == 'neural_network':
                load_tflite_model()
            else:
                model_path = f'models/{model_name}_model_optimized.joblib'
                if not os.path.exists(model_path):
                    model_path = f'models/{model_name}_model.joblib'
                models[model_name] = joblib.load(model_path)
            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
        finally:
            model_queue.task_done()

def start_model_loading():
    """Start background model loading thread"""
    global model_loading_thread
    if model_loading_thread is None:
        model_loading_thread = threading.Thread(target=background_model_loader, daemon=True)
        model_loading_thread.start()
        # Queue models for loading
        for model_name in ['neural_network', 'lightgbm', 'random_forest']:
            model_queue.put(model_name)

def load_config():
    """Lazy load configuration"""
    global config
    if config is None:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    return config

def load_preprocessor():
    """Lazy load preprocessor"""
    global preprocessor
    if preprocessor is None:
        config = load_config()
        preprocessor = DataPreprocessor()
    return preprocessor

def load_tflite_model():
    """Load TFLite model"""
    global tflite_interpreter
    if tflite_interpreter is None:
        model_path = 'models/model_quantized.tflite'
        if os.path.exists(model_path):
            tflite_interpreter = tf.lite.Interpreter(model_path=model_path)
            tflite_interpreter.allocate_tensors()
            logger.info("Successfully loaded TFLite model")
        else:
            logger.error("TFLite model not found")
    return tflite_interpreter

def get_tflite_prediction(interpreter, input_data):
    """Get prediction from TFLite model"""
    if interpreter is None:
        return None
        
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return float(output_data[0][0] > 0.5)

def get_prediction(model_name, input_data):
    """Get prediction from a specific model"""
    if model_name == 'neural_network':
        interpreter = load_tflite_model()
        return get_tflite_prediction(interpreter, input_data)
    
    if model_name not in models:
        return None
        
    model = models[model_name]
    pred = float(model.predict(input_data)[0])
    gc.collect()
    return pred

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for making predictions."""
    try:
        data = request.get_json()
        logger.info(f"Received prediction request with data: {data}")
        
        # Load config and validate input features
        config = load_config()
        required_features = config['data']['feature_columns']
        if not all(feature in data for feature in required_features):
            error_msg = f'Missing features. Required features: {required_features}'
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 400
            
        # Prepare input data
        preprocessor = load_preprocessor()
        input_data = np.array([[data[feature] for feature in required_features]])
        input_scaled = preprocessor.scaler.transform(input_data)
        
        # Make predictions with available models
        predictions = {}
        requested_models = request.args.get('models', 'ensemble').split(',')
        
        if 'ensemble' in requested_models:
            model_names = ['neural_network', 'lightgbm', 'random_forest']
        else:
            model_names = [m for m in requested_models if m != 'ensemble']
            
        for model_name in model_names:
            pred = get_prediction(model_name, input_scaled)
            if pred is not None:
                predictions[model_name] = pred
                
        if predictions:
            # Calculate ensemble prediction if needed
            if 'ensemble' in requested_models:
                ensemble_pred = int(sum(predictions.values()) > len(predictions)/2)
                predictions['ensemble'] = ensemble_pred
            
            response = {
                'predictions': predictions,
                'leak_probability': float(sum(predictions.values()) / len(predictions))
            }
            logger.info(f"Returning predictions: {response}")
            return jsonify(response)
        else:
            return jsonify({'error': 'No models available for prediction'}), 503
            
    except Exception as e:
        error_msg = f"Error making prediction: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        config = load_config()
        preprocessor = load_preprocessor()
        
        health_status = {
            'status': 'healthy',
            'config_loaded': bool(config),
            'preprocessor_initialized': bool(preprocessor),
            'models_loading': not model_queue.empty(),
            'models_loaded': list(models.keys())
        }
        return jsonify(health_status)
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503

if __name__ == '__main__':
    # Start background model loading
    start_model_loading()
    
    # Start Flask server
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)