import onnx
import tf2onnx
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow_model_optimization.quantization.keras import quantize_model
import mlflow

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