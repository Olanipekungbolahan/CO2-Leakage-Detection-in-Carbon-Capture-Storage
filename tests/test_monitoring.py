import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.monitoring import ModelMonitor
import json

@pytest.fixture
def model_monitor():
    return ModelMonitor(redis_host='localhost', redis_port=6379)

@pytest.fixture
def sample_prediction_data():
    return {
        'features': {
            'flow': 619.41,
            'mass': 547.30,
            'density': 1.2,
            'temp': 25.0,
            'conductivity': 250.16,
            'BHP_550m': 64.37
        },
        'prediction': 1,
        'latency': 0.05,
        'ground_truth': 1
    }

def test_feature_drift_detection(model_monitor):
    # Create reference and current datasets
    reference_data = pd.DataFrame(np.random.normal(0, 1, (1000, 3)), 
                                columns=['feature1', 'feature2', 'feature3'])
    current_data = pd.DataFrame(np.random.normal(0.5, 1, (1000, 3)), 
                              columns=['feature1', 'feature2', 'feature3'])
    
    drift_detected = model_monitor.check_feature_drift(current_data, reference_data)
    assert isinstance(drift_detected, bool)

def test_ab_testing_analysis(model_monitor):
    # Generate synthetic A/B test data
    for i in range(100):
        model_monitor.log_prediction(
            'model_a',
            {'feature1': np.random.rand()},
            prediction=1,
            latency=0.1,
            ground_truth=1 if np.random.rand() > 0.2 else 0
        )
        model_monitor.log_prediction(
            'model_b',
            {'feature1': np.random.rand()},
            prediction=1,
            latency=0.15,
            ground_truth=1 if np.random.rand() > 0.3 else 0
        )
    
    results = model_monitor.ab_test_analysis('model_a', 'model_b', 
                                           time_window=timedelta(days=1))
    assert isinstance(results, dict)
    assert 'p_value' in results
    assert 'significant_difference' in results

def test_prediction_logging(model_monitor, sample_prediction_data):
    model_monitor.log_prediction(
        'test_model',
        sample_prediction_data['features'],
        sample_prediction_data['prediction'],
        sample_prediction_data['latency'],
        sample_prediction_data['ground_truth']
    )
    
    # Verify Redis storage
    results = model_monitor._get_model_results('test_model', 
                                             datetime.now() - timedelta(minutes=1))
    assert len(results) > 0
    assert isinstance(results[0], dict)