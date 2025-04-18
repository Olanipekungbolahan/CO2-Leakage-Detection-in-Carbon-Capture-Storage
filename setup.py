from setuptools import setup, find_packages

setup(
    name="co2_leakage_detection",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'Flask>=3.0.2',
        'numpy>=1.24.3',
        'pandas>=2.1.4',
        'scikit-learn>=1.4.0',
        'xgboost>=2.0.3',
        'tensorflow>=2.15.0',
        'optuna>=3.5.0',
        'mlflow>=2.10.0'
    ],
    author="CO2 Leakage Detection Team",
    description="Machine Learning models for CO2 leakage detection in CCS systems",
    python_requires=">=3.8",
)