import logging
import mlflow
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, Any, Optional, List
from monitoring import ModelMonitor
from train import main as train_models
from model_optimization import ModelOptimizer
import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoRetrainer:
    def __init__(
        self,
        accuracy_threshold: float = 0.90,
        drift_threshold: float = 0.05,
        evaluation_window: int = 3600,  # 1 hour
        min_samples_for_retraining: int = 1000,
        max_retraining_frequency: timedelta = timedelta(days=1)
    ):
        self.accuracy_threshold = accuracy_threshold
        self.drift_threshold = drift_threshold
        self.evaluation_window = evaluation_window
        self.min_samples_for_retraining = min_samples_for_retraining
        self.max_retraining_frequency = max_retraining_frequency
        self.monitor = ModelMonitor()
        self.model_optimizer = ModelOptimizer()
        self.last_retrain_time = None
        
    def check_retraining_needed(self, model_name: str) -> tuple[bool, str]:
        """Check if model needs retraining based on performance metrics"""
        # Get recent performance metrics
        end_time = datetime.now()
        start_time = end_time - timedelta(seconds=self.evaluation_window)
        results = self.monitor._get_model_results(model_name, start_time)
        
        if not results or len(results) < self.min_samples_for_retraining:
            return False, "Insufficient data for evaluation"
            
        # Check if we're within the minimum retraining frequency
        if (self.last_retrain_time and 
            datetime.now() - self.last_retrain_time < self.max_retraining_frequency):
            return False, "Too soon since last retraining"
            
        # Calculate current performance metrics
        metrics = self.monitor._calculate_metrics(results)
        
        # Get feature data for drift detection
        current_features = pd.DataFrame([r['features'] for r in results])
        reference_features = self._get_reference_features()
        
        # Check accuracy degradation
        if metrics['accuracy'] < self.accuracy_threshold:
            return True, "Accuracy below threshold"
            
        # Check for feature drift
        if self.monitor.check_feature_drift(current_features, reference_features, 
                                          self.drift_threshold):
            return True, "Significant feature drift detected"
            
        return False, "Model performing adequately"
        
    def _get_reference_features(self) -> pd.DataFrame:
        """Get reference feature distribution from training data"""
        # Load original training data
        data = pd.read_csv("Dataset_Brine_Injection_Ketzin_final--dynamic phase.csv")
        return data[['flow', 'mass', 'density', 'temp', 'conductivity', 'BHP_550m']]
        
    def retrain_model(self, model_name: str) -> bool:
        """Execute model retraining process"""
        try:
            logger.info(f"Starting retraining for model {model_name}")
            
            # Collect recent data for retraining
            recent_data = self._collect_recent_data()
            
            # Update training dataset with recent data
            self._update_training_data(recent_data)
            
            # Execute training pipeline
            train_models()
            
            # Optimize new model
            self._optimize_new_model(model_name)
            
            self.last_retrain_time = datetime.now()
            
            logger.info(f"Successfully retrained model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Retraining failed: {str(e)}")
            return False
            
    def _collect_recent_data(self) -> pd.DataFrame:
        """Collect recent prediction data with ground truth for retraining"""
        recent_results = self.monitor._get_model_results(
            "ensemble",  # Get results from ensemble predictions
            datetime.now() - timedelta(days=7)  # Last week's data
        )
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            **r['features'],
            'label': r['ground_truth']
        } for r in recent_results if r.get('ground_truth') is not None])
        
        return df
        
    def _update_training_data(self, new_data: pd.DataFrame):
        """Update training dataset with new data"""
        original_data = pd.read_csv("Dataset_Brine_Injection_Ketzin_final--dynamic phase.csv")
        
        # Combine original and new data
        updated_data = pd.concat([original_data, new_data], ignore_index=True)
        
        # Save updated dataset
        updated_data.to_csv("Dataset_Brine_Injection_Ketzin_final--dynamic phase.csv", index=False)
        
    def _optimize_new_model(self, model_name: str):
        """Optimize newly trained model"""
        model = mlflow.sklearn.load_model(f"models/{model_name}")
        optimized_model = self.model_optimizer.optimize_model_size(model, (None, 6))
        mlflow.sklearn.save_model(optimized_model, f"models/{model_name}_optimized")
        
    def start_monitoring(self):
        """Start continuous monitoring for model degradation"""
        while True:
            for model_name in ['neural_network', 'xgboost', 'random_forest', 'svm']:
                needs_retrain, reason = self.check_retraining_needed(model_name)
                
                if needs_retrain:
                    logger.info(f"Retraining triggered for {model_name}: {reason}")
                    success = self.retrain_model(model_name)
                    
                    if success:
                        logger.info(f"Successfully retrained {model_name}")
                    else:
                        logger.error(f"Failed to retrain {model_name}")
                        
            # Wait before next check
            time.sleep(self.evaluation_window)

if __name__ == "__main__":
    retrainer = AutoRetrainer()
    retrainer.start_monitoring()