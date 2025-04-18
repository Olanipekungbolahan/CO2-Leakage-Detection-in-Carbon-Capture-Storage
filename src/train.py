import os
import yaml
import joblib
from preprocessing import DataPreprocessor
from models import ModelTrainer

def main():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize preprocessing and model training
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer()
    
    # Load and preprocess data
    data_path = os.path.join('data', 'Dataset_Brine_Injection_Ketzin_final--dynamic phase.csv')
    df = preprocessor.load_data(data_path)
    X_train, X_test, y_train, y_test = preprocessor.prepare_features(df)
    
    # Train all models
    results = {}
    for model_type in ['neural_network', 'xgboost', 'random_forest', 'svm']:
        print(f"\nTraining {model_type}...")
        model, metrics = trainer.train_and_evaluate(
            X_train, X_test, y_train, y_test, model_type
        )
        results[model_type] = metrics
        
        # Save model
        if model_type != 'neural_network':
            joblib.dump(model, f'models/{model_type}_model.joblib')
        else:
            model.save(f'models/{model_type}_model')
            
        # Log to W&B
        trainer.log_to_wandb(model, metrics, model_type)
        
        print(f"{model_type} Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()