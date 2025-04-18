import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import yaml

class DataPreprocessor:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.scaler = MinMaxScaler()
        
    def load_data(self, data_path):
        """Load and preprocess the dataset."""
        df = pd.read_csv(data_path)
        return self._clean_data(df)
    
    def _clean_data(self, df):
        """Clean the data by handling missing values and converting datatypes."""
        # Convert numeric columns
        numeric_columns = self.config['data']['feature_columns']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Handle missing values
        if self.config['preprocessing']['handle_missing'] == 'mean':
            df = df.fillna(df.mean())
            
        return df
    
    def prepare_features(self, df):
        """Prepare features for model training."""
        X = df[self.config['data']['feature_columns']]
        y = df[self.config['data']['target_column']]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Handle class imbalance if specified
        if self.config['preprocessing']['handle_imbalance'] == 'SMOTE':
            smote = SMOTE(random_state=self.config['data']['random_state'])
            X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
        else:
            X_resampled, y_resampled = X_scaled, y
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, 
            y_resampled,
            test_size=self.config['data']['train_test_split'],
            random_state=self.config['data']['random_state']
        )
        
        return X_train, X_test, y_train, y_test
    
    def inverse_transform_features(self, X):
        """Transform scaled features back to original scale."""
        return self.scaler.inverse_transform(X)