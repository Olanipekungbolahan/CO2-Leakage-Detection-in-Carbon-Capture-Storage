import mlflow
import optuna
import numpy as np
import tensorflow as tf
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import yaml
import wandb

class ModelTrainer:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
    def create_neural_network(self):
        """Create neural network model based on config."""
        model = tf.keras.Sequential()
        for layer in self.config['models']['neural_network']['architecture']:
            model.add(tf.keras.layers.Dense(
                units=layer['units'],
                activation=layer['activation']
            ))
        model.compile(
            optimizer=self.config['models']['neural_network']['optimizer'],
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def optimize_hyperparameters(self, model_type, X_train, y_train):
        """Optimize hyperparameters using Optuna."""
        def objective(trial):
            if model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0)
                }
                model = XGBClassifier(**params)
            elif model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 15)
                }
                model = RandomForestClassifier(**params)
                
            model.fit(X_train, y_train)
            return model.score(X_train, y_train)
            
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config['hyperparameter_optimization']['n_trials'])
        return study.best_params
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, model_type='neural_network'):
        """Train model and log metrics with MLflow."""
        with mlflow.start_run():
            if model_type == 'neural_network':
                model = self.create_neural_network()
                history = model.fit(
                    X_train, y_train,
                    epochs=self.config['models']['neural_network']['epochs'],
                    batch_size=self.config['models']['neural_network']['batch_size'],
                    validation_data=(X_test, y_test)
                )
                y_pred = (model.predict(X_test) > 0.5).astype(int)
                
            elif model_type in ['xgboost', 'random_forest']:
                best_params = self.optimize_hyperparameters(model_type, X_train, y_train)
                if model_type == 'xgboost':
                    model = XGBClassifier(**best_params)
                else:
                    model = RandomForestClassifier(**best_params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
            elif model_type == 'svm':
                model = SVC(**self.config['models']['svm'])
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
            
            # Log metrics
            mlflow.log_metrics(metrics)
            if model_type != 'neural_network':
                mlflow.sklearn.log_model(model, model_type)
            else:
                mlflow.tensorflow.log_model(model, 'neural_network')
                
            # Log confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            mlflow.log_metric("confusion_matrix", cm.tolist())
            
            return model, metrics

    def log_to_wandb(self, model, metrics, model_type):
        """Log results to Weights & Biases."""
        wandb.init(
            project=self.config['wandb']['project'],
            entity=self.config['wandb']['entity'],
            config=self.config['models'][model_type]
        )
        wandb.log(metrics)
        if model_type != 'neural_network':
            wandb.sklearn.plot_feature_importances(model)
        wandb.finish()