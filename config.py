"""
Central configuration file for the MLOps project.
All paths, hyperparameters, and settings are defined here.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# Directory paths
PATHS = {
    "project_root": BASE_DIR,
    "data_dir": BASE_DIR / "2_data_management" / "data",
    "artifacts_dir": BASE_DIR / "4_model" / "artifacts",
    "monitoring_dir": BASE_DIR / "9_monitoring" / "monitoring_logs",
    "experiments_dir": BASE_DIR / "7_experiments" / "runs",
    "validated_data_dir": BASE_DIR / "5_validation" / "validated_data",
    "metrics_dir": BASE_DIR / "6_evaluation" / "metrics",
    "transformers_dir": BASE_DIR / "3_features" / "transformers",
}

# Create directories if they don't exist
for path in PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

# Data configuration
DATA_CONFIG = {
    "train_test_split": 0.2,
    "random_seed": 42,
    "validation_split": 0.2,
    "n_samples": 5000,  # Number of synthetic houses to generate
}

# Model configuration
MODEL_CONFIG = {
    "algorithm": "linear_regression",
    "hyperparams": {
        "fit_intercept": True,
    },
}

# Feature engineering configuration
FEATURE_CONFIG = {
    "numeric_features": ["square_feet", "bedrooms", "bathrooms", "age_years", "neighborhood_quality"],
    "categorical_features": ["has_garage"],
    "target": "price",
    "scaling_method": "standard",  # standard or minmax
}

# Deployment configuration
DEPLOYMENT_CONFIG = {
    "canary_start_pct": 0.01,
    "canary_step_pct": 0.10,  # Increase canary by 10% each step
    "canary_duration_hours": 2,
    "canary_max_pct": 1.0,
    "ab_test_duration_days": 7,
    "ab_test_traffic_split": 0.5,
    "latency_threshold_ms": 500,  # Maximum acceptable latency
}

# Monitoring configuration
MONITORING_CONFIG = {
    "prediction_log_frequency": 100,  # Log every N predictions
    "drift_check_frequency": "daily",
    "drift_threshold": 0.20,  # 20% change triggers alert
    "anomaly_threshold": 0.20,  # 20% deviation from baseline
    "metrics_retention_days": 90,
}

# Guardrails (hard constraints that must never be violated)
GUARDRAILS = {
    "mae_threshold": 50000,  # MAE must be <= $50,000
    "rmse_threshold": 75000,  # RMSE must be <= $75,000
    "r2_threshold": 0.85,  # R² must be >= 0.85
    "latency_ms": 500,  # Prediction latency must be < 500ms
    "max_error_rate": 0.01,  # Error rate must be < 1%
    "min_prediction": 0,  # Predictions must be >= 0 (no negative prices)
    "max_prediction": 10000000,  # Predictions must be <= $10M
    "max_missing_rate": 0.05,  # Missing values must be < 5%
}

# Business metrics
BUSINESS_METRICS = {
    "target_listing_time_reduction": 0.15,  # 15% reduction
    "target_user_engagement": 0.20,  # 20% increase in clicks
    "target_conversion_rate": 0.10,  # 10% increase in listings
}

# Retraining triggers
RETRAINING_CONFIG = {
    "schedule_days": 7,  # Retrain every 7 days
    "time_based_days": 30,  # Retrain every 30 days
    "data_based_samples": 1000,  # Retrain when 1000 new samples collected
    "drift_based_threshold": 0.20,  # Retrain when drift > 20%
    "performance_based_threshold": 0.15,  # Retrain when MAE increases by 15%
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": BASE_DIR / "mlops.log",
}
