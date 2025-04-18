import pytest
import numpy as np
import pandas as pd
from src.preprocessing import DataPreprocessor
from src.models import ModelTrainer
import yaml
import os

@pytest.fixture
def config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'flow': np.random.rand(100),
        'mass': np.random.rand(100),
        'density': np.random.rand(100),
        'temp': np.random.rand(100),
        'conductivity': np.random.rand(100),
        'BHP_550m': np.random.rand(100),
        'label': np.random.randint(0, 2, 100)
    })

def test_data_preprocessor(sample_data, config):
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_features(sample_data)
    
    # Test shapes
    assert len(X_train) + len(X_test) >= len(sample_data)  # Account for SMOTE
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    
    # Test scaling
    assert np.all(X_train >= 0) and np.all(X_train <= 1)
    assert np.all(X_test >= 0) and np.all(X_test <= 1)

def test_model_trainer(sample_data, config):
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer()
    
    X_train, X_test, y_train, y_test = preprocessor.prepare_features(sample_data)
    
    # Test each model type
    for model_type in ['neural_network', 'xgboost', 'random_forest', 'svm']:
        model, metrics = trainer.train_and_evaluate(
            X_train, X_test, y_train, y_test, model_type
        )
        
        # Test metrics structure
        assert all(metric in metrics for metric in ['accuracy', 'precision', 'recall', 'f1'])
        assert all(0 <= metrics[metric] <= 1 for metric in metrics)

def test_hyperparameter_optimization(sample_data):
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer()
    
    X_train, X_test, y_train, y_test = preprocessor.prepare_features(sample_data)
    
    # Test optimization for tree-based models
    for model_type in ['xgboost', 'random_forest']:
        best_params = trainer.optimize_hyperparameters(model_type, X_train, y_train)
        assert isinstance(best_params, dict)
        assert len(best_params) > 0

def test_model_saving(sample_data, tmp_path):
    import joblib
    import tensorflow as tf
    
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer()
    
    X_train, X_test, y_train, y_test = preprocessor.prepare_features(sample_data)
    
    # Test saving for each model type
    for model_type in ['xgboost', 'random_forest', 'svm']:
        model, _ = trainer.train_and_evaluate(
            X_train, X_test, y_train, y_test, model_type
        )
        save_path = tmp_path / f"{model_type}_model.joblib"
        joblib.dump(model, save_path)
        assert save_path.exists()
        
    # Test saving neural network
    model, _ = trainer.train_and_evaluate(
        X_train, X_test, y_train, y_test, 'neural_network'
    )
    save_path = tmp_path / "neural_network_model"
    model.save(save_path)
    assert save_path.exists()