from flask import Flask, request, jsonify
import tensorflow as tf
import yaml
import numpy as np
from preprocessing import DataPreprocessor
from monitoring import ModelMonitor
import logging
import gc
import os
import time
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global state
tflite_interpreter = None
preprocessor = None
config = None
monitor = ModelMonitor(port=8000)

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
        try:
            model_path = 'models/model_quantized.tflite'
            if os.path.exists(model_path):
                tflite_interpreter = tf.lite.Interpreter(model_path=model_path)
                tflite_interpreter.allocate_tensors()
                logger.info("Successfully loaded TFLite model")
                
                # Log initial memory usage
                process = psutil.Process(os.getpid())
                monitor.update_memory_usage(process.memory_info().rss)
            else:
                logger.error("TFLite model not found")
                monitor.log_error("Model not found")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            monitor.log_error(f"Model loading error: {str(e)}")
    return tflite_interpreter

def get_prediction(input_data):
    """Get prediction from the neural network model"""
    start_time = time.time()
    try:
        interpreter = load_tflite_model()
        if interpreter is None:
            monitor.log_error("Model not loaded")
            return None
            
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = float(output_data[0][0])
        
        # Log prediction latency
        latency = time.time() - start_time
        monitor.log_prediction(
            features={f"feature_{i}": val for i, val in enumerate(input_data[0])},
            prediction=prediction,
            latency=latency
        )
        
        # Update memory usage
        process = psutil.Process(os.getpid())
        monitor.update_memory_usage(process.memory_info().rss)
        
        return prediction > 0.5
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        monitor.log_error(f"Prediction error: {str(e)}")
        return None

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
            monitor.log_error("Missing features")
            return jsonify({'error': error_msg}), 400
            
        # Prepare input data
        preprocessor = load_preprocessor()
        input_data = np.array([[data[feature] for feature in required_features]])
        input_scaled = preprocessor.scaler.transform(input_data)
        
        # Get prediction
        prediction = get_prediction(input_scaled)
        if prediction is not None:
            response = {
                'prediction': int(prediction),
                'leak_probability': float(prediction)
            }
            logger.info(f"Returning prediction: {response}")
            return jsonify(response)
        else:
            return jsonify({'error': 'Model not available for prediction'}), 503
            
    except Exception as e:
        error_msg = f"Error making prediction: {str(e)}"
        logger.error(error_msg)
        monitor.log_error(f"API error: {str(e)}")
        return jsonify({'error': error_msg}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        config = load_config()
        preprocessor = load_preprocessor()
        model = load_tflite_model()
        
        # Get current metrics
        metrics = monitor.get_current_metrics()
        
        health_status = {
            'status': 'healthy',
            'config_loaded': bool(config),
            'preprocessor_initialized': bool(preprocessor),
            'model_loaded': model is not None,
            'metrics': metrics
        }
        return jsonify(health_status)
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)