import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
from prometheus_client import start_http_server, Counter, Histogram, Gauge, Summary
import logging

logger = logging.getLogger(__name__)

class ModelMonitor:
    def __init__(self, port: int = 8000):
        # Prometheus metrics
        self.prediction_counter = Counter(
            'model_predictions_total',
            'Total number of predictions made',
            ['result']  # 'correct' or 'incorrect'
        )
        
        self.prediction_latency = Histogram(
            'model_prediction_latency_seconds',
            'Time taken for model prediction',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
        )
        
        self.prediction_confidence = Histogram(
            'model_prediction_confidence',
            'Confidence scores of predictions',
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )
        
        self.model_errors = Counter(
            'model_errors_total',
            'Total number of model errors'
        )
        
        self.memory_gauge = Gauge(
            'model_memory_usage_bytes',
            'Memory usage of the model'
        )
        
        # Performance tracking
        self.accuracy_gauge = Gauge(
            'model_accuracy',
            'Current model accuracy'
        )
        
        self.drift_score = Gauge(
            'model_drift_score',
            'Current model drift score'
        )
        
        # Start Prometheus HTTP server
        start_http_server(port)
        logger.info(f"Started Prometheus metrics server on port {port}")
    
    def log_prediction(self, features: Dict[str, float], prediction: float, 
                      latency: float, ground_truth: int = None):
        """Log prediction details for monitoring"""
        # Log prediction latency
        self.prediction_latency.observe(latency)
        
        # Log prediction confidence
        self.prediction_confidence.observe(abs(prediction - 0.5) * 2)
        
        # Log accuracy if ground truth is available
        if ground_truth is not None:
            is_correct = int(prediction > 0.5) == ground_truth
            self.prediction_counter.labels(
                result='correct' if is_correct else 'incorrect'
            ).inc()
            
            # Update running accuracy
            current_acc = float(self.accuracy_gauge._value.get())
            total_preds = sum(self.prediction_counter.collect()[0].samples[0].value)
            new_acc = (current_acc * (total_preds - 1) + float(is_correct)) / total_preds
            self.accuracy_gauge.set(new_acc)
    
    def log_error(self, error_type: str):
        """Log model errors"""
        self.model_errors.inc()
        logger.error(f"Model error occurred: {error_type}")
    
    def update_memory_usage(self, bytes_used: int):
        """Update model memory usage metric"""
        self.memory_gauge.set(bytes_used)
    
    def update_drift_score(self, score: float):
        """Update feature drift score"""
        self.drift_score.set(score)
        if score > 0.1:  # Alert threshold
            logger.warning(f"High drift score detected: {score}")
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current monitoring metrics"""
        return {
            'accuracy': float(self.accuracy_gauge._value.get()),
            'drift_score': float(self.drift_score._value.get()),
            'error_rate': float(self.model_errors._value.get()) / max(1, sum(self.prediction_counter.collect()[0].samples[0].value)),
            'avg_latency': float(sum(self.prediction_latency._sum.get()) / max(1, self.prediction_latency._count.get())),
            'memory_usage_mb': float(self.memory_gauge._value.get()) / (1024 * 1024)
        }