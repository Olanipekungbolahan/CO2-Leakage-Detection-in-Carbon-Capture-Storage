import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Any
import mlflow
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import json
import redis
from scipy.stats import chi2_contingency
import logging

class ModelMonitor:
    def __init__(self, redis_host='localhost', redis_port=6379):
        # Initialize Prometheus metrics
        self.prediction_latency = Histogram('prediction_latency_seconds', 'Time for model prediction')
        self.prediction_counter = Counter('prediction_total', 'Total number of predictions', ['model', 'result'])
        self.feature_drift = Gauge('feature_drift', 'Feature drift score', ['feature'])
        
        # Initialize Redis for A/B testing
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def start_monitoring_server(self, port=8000):
        """Start Prometheus metrics server"""
        start_http_server(port)
        self.logger.info(f"Model monitoring server started on port {port}")
        
    def log_prediction(self, model_name: str, features: Dict[str, float], 
                      prediction: int, latency: float, ground_truth: int = None):
        """Log prediction details for monitoring"""
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_metrics({
                'prediction_latency': latency,
                'prediction': prediction
            })
            if ground_truth is not None:
                mlflow.log_metric('accuracy', int(prediction == ground_truth))
        
        # Update Prometheus metrics
        self.prediction_latency.observe(latency)
        self.prediction_counter.labels(model=model_name, 
                                    result='correct' if ground_truth == prediction else 'incorrect'
                                    ).inc()
        
        # Store in Redis for A/B testing
        prediction_data = {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'prediction': prediction,
            'latency': latency,
            'ground_truth': ground_truth
        }
        self.redis_client.lpush(f'predictions:{model_name}', 
                              json.dumps(prediction_data))
                              
    def check_feature_drift(self, current_data: pd.DataFrame, 
                          reference_data: pd.DataFrame, threshold: float = 0.05):
        """Monitor feature drift using statistical tests"""
        drift_detected = False
        for column in current_data.columns:
            # Perform Chi-squared test for drift detection
            hist_current, _ = np.histogram(current_data[column], bins=10)
            hist_ref, _ = np.histogram(reference_data[column], bins=10)
            
            chi2_stat, p_value = chi2_contingency(np.vstack([hist_current, hist_ref]))[:2]
            
            # Update Prometheus metric
            self.feature_drift.labels(feature=column).set(p_value)
            
            if p_value < threshold:
                drift_detected = True
                self.logger.warning(f"Drift detected in feature {column}: p-value = {p_value}")
                
        return drift_detected
        
    def ab_test_analysis(self, model_a: str, model_b: str, 
                        time_window: timedelta = timedelta(days=7)):
        """Perform A/B testing analysis between two models"""
        cutoff_time = datetime.now() - time_window
        
        # Retrieve data from Redis
        results_a = self._get_model_results(model_a, cutoff_time)
        results_b = self._get_model_results(model_b, cutoff_time)
        
        if not results_a or not results_b:
            return None
            
        # Calculate metrics for both models
        metrics_a = self._calculate_metrics(results_a)
        metrics_b = self._calculate_metrics(results_b)
        
        # Perform statistical significance test
        chi2_stat, p_value = self._calculate_significance(
            metrics_a['correct'], metrics_a['total'],
            metrics_b['correct'], metrics_b['total']
        )
        
        return {
            'model_a': metrics_a,
            'model_b': metrics_b,
            'p_value': p_value,
            'significant_difference': p_value < 0.05
        }
        
    def _get_model_results(self, model_name: str, cutoff_time: datetime):
        """Retrieve model results from Redis"""
        results = []
        for item in self.redis_client.lrange(f'predictions:{model_name}', 0, -1):
            data = json.loads(item)
            if datetime.fromisoformat(data['timestamp']) >= cutoff_time:
                results.append(data)
        return results
        
    def _calculate_metrics(self, results: List[Dict[str, Any]]):
        """Calculate metrics for A/B testing"""
        total = len(results)
        correct = sum(1 for r in results if r['prediction'] == r['ground_truth'])
        avg_latency = np.mean([r['latency'] for r in results])
        
        return {
            'total': total,
            'correct': correct,
            'accuracy': correct/total if total > 0 else 0,
            'avg_latency': avg_latency
        }
        
    def _calculate_significance(self, correct_a: int, total_a: int, 
                              correct_b: int, total_b: int):
        """Calculate statistical significance using chi-square test"""
        contingency_table = np.array([
            [correct_a, total_a - correct_a],
            [correct_b, total_b - correct_b]
        ])
        return chi2_contingency(contingency_table)[:2]