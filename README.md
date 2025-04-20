# CO2 Leakage Detection in CCS using Machine Learning

This project implements machine learning models for detecting CO2 leakage in Carbon Capture Storage systems using sensor data.

## Features

- Multiple ML models including Neural Networks, XGBoost, Random Forest, and SVM
- Automated hyperparameter optimization using Optuna
- Model performance tracking with MLflow
- Experiment monitoring with Weights & Biases
- REST API for model inference
- Comprehensive test suite
- Ensemble predictions

## Project Structure

```
├── config.yaml           # Configuration file
├── requirements.txt      # Project dependencies
├── setup.py             # Package setup file
├── data/                # Data directory
├── models/              # Saved model files
├── src/
│   ├── preprocessing.py # Data preprocessing
│   ├── models.py        # Model implementations
│   ├── train.py        # Training pipeline
│   └── serve.py        # REST API service
├── tests/              # Test suite
└── notebooks/          # Jupyter notebooks
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure settings in `config.yaml`

## Training Models

```bash
python src/train.py
```

This will:
- Load and preprocess the data
- Train all models with hyperparameter optimization
- Log metrics to MLflow and W&B
- Save models to the models/ directory

## Running the API

```bash
python src/serve.py
```

The API provides:
- POST /predict - Make predictions using all models
- GET /health - Health check endpoint

## Example API Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "flow": 619.41,
    "mass": 547.30,
    "density": 1.2,
    "temp": 25.0,
    "conductivity": 250.16,
    "BHP_550m": 64.37
  }'
```

## Running Tests

```bash
pytest tests/
```

## Model Performance

The system uses an ensemble of models for robust predictions:
- Neural Network with RMSprop optimizer (~93.82% accuracy)
- XGBoost with hyperparameter optimization
- Random Forest Classifier
- Support Vector Machine

Key metrics tracked:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Monitoring

- MLflow: Track experiments and model versions
- Weights & Biases: Monitor training metrics and model performance
- API health monitoring endpoint

## Deployment Challenges and Solutions

### Challenge 1: Heroku Slug Size Limit
Initially, the application exceeded Heroku's slug size limit (500MB) with a size of over 1GB. This was primarily due to:
- Multiple ML models being deployed
- Large model weights and dependencies
- Unused libraries in requirements.txt

Solutions implemented:
1. Model Optimization
   - Converted to TFLite format with quantization
   - Selected single best-performing model (Neural Network, 93.82% accuracy)
   - Removed ensemble prediction logic

2. Dependency Optimization
   - Switched to tensorflow-cpu from full tensorflow
   - Removed unused ML libraries
   - Minimized requirements to essential packages only

3. Code Optimization
   - Implemented lazy loading for model and configurations
   - Added memory usage monitoring
   - Streamlined API endpoints

Result: Successfully reduced slug size to under 500MB while maintaining model performance.

### Challenge 2: Performance Monitoring
To ensure reliable production deployment, we implemented comprehensive monitoring:

1. Real-time Metrics
   - Model prediction latency
   - Memory usage tracking
   - Prediction confidence scores
   - Error rate monitoring

2. Health Checks
   - Model availability
   - Prediction accuracy
   - Resource utilization
   - API endpoint status

3. Alerting System
   - Drift detection
   - Performance degradation alerts
   - Error rate thresholds
   - Memory usage warnings

### Best Practices Implemented
1. Use of TFLite for model optimization
2. Implementation of lazy loading
3. Comprehensive monitoring with Prometheus
4. Regular memory cleanup
5. Efficient error handling and logging

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License

