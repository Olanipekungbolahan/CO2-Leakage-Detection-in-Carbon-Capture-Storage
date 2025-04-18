from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import yaml
import numpy as np
from preprocessing import DataPreprocessor

app = Flask(__name__)

# Load configuration and initialize preprocessor
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
preprocessor = DataPreprocessor()

# Load all models
models = {}
try:
    models['neural_network'] = tf.keras.models.load_model('models/neural_network_model')
    models['xgboost'] = joblib.load('models/xgboost_model.joblib')
    models['random_forest'] = joblib.load('models/random_forest_model.joblib')
    models['svm'] = joblib.load('models/svm_model.joblib')
except Exception as e:
    print(f"Error loading models: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for making predictions."""
    try:
        data = request.get_json()
        
        # Validate input features
        required_features = config['data']['feature_columns']
        if not all(feature in data for feature in required_features):
            return jsonify({
                'error': f'Missing features. Required features: {required_features}'
            }), 400
            
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
            
        # Calculate ensemble prediction (majority voting)
        ensemble_pred = int(sum(predictions.values()) > len(predictions)/2)
        predictions['ensemble'] = ensemble_pred
        
        return jsonify({
            'predictions': predictions,
            'leak_probability': float(sum(predictions.values()) / len(predictions))
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'models_loaded': list(models.keys())})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)