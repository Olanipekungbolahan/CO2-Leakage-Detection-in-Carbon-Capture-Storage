import onnx
import tf2onnx
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow_model_optimization.quantization.keras import quantize_model
import mlflow
import joblib
from sklearn.preprocessing import StandardScaler
import tensorflow_model_optimization as tfmot
import os

class ModelOptimizer:
    def __init__(self):
        self.quantization_aware_training = True
        
    def quantize_model(self, model: Model):
        """Quantize TensorFlow model using quantization-aware training."""
        quantized_model = quantize_model(model)
        return quantized_model
        
    def export_to_onnx(self, model, model_path: str, input_shape: tuple):
        """Export model to ONNX format."""
        if isinstance(model, tf.keras.Model):
            # Convert Keras model to ONNX
            spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
            output_path = f"{model_path}.onnx"
            model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
            return output_path
        else:
            # For sklearn models, use skl2onnx
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            initial_type = [('float_input', FloatTensorType([None, input_shape[1]]))]
            onx = convert_sklearn(model, initial_types=initial_type)
            output_path = f"{model_path}.onnx"
            with open(output_path, "wb") as f:
                f.write(onx.SerializeToString())
            return output_path
            
    def optimize_model_size(self, model, input_shape):
        """Optimize model size through pruning and quantization."""
        if isinstance(model, tf.keras.Model):
            # Apply pruning
            import tensorflow_model_optimization.sparsity.keras as sparsity
            pruning_schedule = sparsity.PolynomialDecay(
                initial_sparsity=0.0, 
                final_sparsity=0.5,
                begin_step=0,
                end_step=1000
            )
            pruned_model = sparsity.prune_low_magnitude(model, pruning_schedule=pruning_schedule)
            
            # Apply quantization
            if self.quantization_aware_training:
                pruned_model = self.quantize_model(pruned_model)
                
            return pruned_model
        return model  # Return original model for non-TF models
        
    def benchmark_model(self, model, test_data, test_labels):
        """Benchmark model performance and size."""
        import time
        
        # Measure inference time
        start_time = time.time()
        predictions = model.predict(test_data)
        inference_time = (time.time() - start_time) / len(test_data)
        
        # Calculate model size
        if isinstance(model, tf.keras.Model):
            model.save('temp_model')
            import os
            model_size = sum(os.path.getsize(f'temp_model/{f}') for f in os.listdir('temp_model'))
            import shutil
            shutil.rmtree('temp_model')
        else:
            import joblib
            joblib.dump(model, 'temp_model.joblib')
            model_size = os.path.getsize('temp_model.joblib')
            os.remove('temp_model.joblib')
            
        # Log metrics
        metrics = {
            'inference_time_ms': inference_time * 1000,
            'model_size_mb': model_size / (1024 * 1024),
            'accuracy': accuracy_score(test_labels, (predictions > 0.5).astype(int) if isinstance(model, tf.keras.Model) else predictions)
        }
        
        mlflow.log_metrics(metrics)
        return metrics

    def optimize_tensorflow_model(self, model_path):
        """Optimize TensorFlow model through quantization"""
        # Load the model
        model = tf.keras.models.load_model(model_path)
        
        # Apply quantization aware training
        quantize_model = tfmot.quantization.keras.quantize_model
        
        # Create quantized model
        q_aware_model = quantize_model(model)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        # Save the quantized model
        tflite_path = os.path.join(os.path.dirname(model_path), 'model_quantized.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
            
        return tflite_path

    def optimize_sklearn_model(self, model_path):
        """Optimize scikit-learn based models through pruning"""
        model = joblib.load(model_path)
        
        # For tree-based models, reduce number of estimators if possible
        if hasattr(model, 'estimators_'):
            n_estimators = len(model.estimators_)
            if n_estimators > 50:  # Keep minimum 50 estimators
                model.estimators_ = model.estimators_[:50]
        
        # Save optimized model with compression
        optimized_path = model_path.replace('.joblib', '_optimized.joblib')
        joblib.dump(model, optimized_path, compress=9)
        
        return optimized_path

def optimize_all_models():
    """Optimize all models in the models directory"""
    optimizer = ModelOptimizer()
    models_dir = 'models'
    
    # Optimize neural network
    nn_path = os.path.join(models_dir, 'neural_network_model')
    if os.path.exists(nn_path):
        optimizer.optimize_tensorflow_model(nn_path)
    
    # Optimize scikit-learn models
    for model_name in ['xgboost', 'random_forest', 'svm']:
        model_path = os.path.join(models_dir, f'{model_name}_model.joblib')
        if os.path.exists(model_path):
            optimizer.optimize_sklearn_model(model_path)

if __name__ == '__main__':
    optimize_all_models()