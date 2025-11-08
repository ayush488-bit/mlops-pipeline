"""
Phase 4: First Model
Trains a simple Linear Regression model.
Keeps it simple - no hyperparameter tuning yet, just get it working end-to-end.
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any, Tuple, List
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import MODEL_CONFIG, PATHS, GUARDRAILS


class FirstModelTrainer:
    """
    Trains a simple Linear Regression model.
    
    Philosophy: Start simple, get end-to-end pipeline working first.
    Don't optimize prematurely - that comes later.
    """
    
    def __init__(self):
        """Initialize model trainer."""
        self.model = None
        self.training_metadata = {}
        self.is_trained = False
    
    def train(self, 
             X_train: pd.DataFrame, 
             y_train: np.ndarray,
             version: str = "v1") -> LinearRegression:
        """
        Train a Linear Regression model.
        
        Args:
            X_train: Training features (already scaled)
            y_train: Training target
            version: Model version identifier
            
        Returns:
            Trained model
        """
        print("\n" + "=" * 80)
        print(f"TRAINING MODEL {version}")
        print("=" * 80)
        
        print(f"\n🔧 Initializing Linear Regression...")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Features: {X_train.shape[1]}")
        
        # Initialize model with config
        self.model = LinearRegression(**MODEL_CONFIG["hyperparams"])
        
        # Train
        print(f"\n⚙️  Training...")
        self.model.fit(X_train, y_train)
        
        # Store metadata
        self.training_metadata = {
            "version": version,
            "algorithm": MODEL_CONFIG["algorithm"],
            "hyperparams": MODEL_CONFIG["hyperparams"],
            "n_samples": len(X_train),
            "n_features": X_train.shape[1],
            "feature_names": list(X_train.columns),
            "trained_at": datetime.now().isoformat(),
            "model_coefficients": {
                feature: float(coef) 
                for feature, coef in zip(X_train.columns, self.model.coef_)
            },
            "intercept": float(self.model.intercept_)
        }
        
        self.is_trained = True
        
        print(f"✅ Model trained successfully!")
        print(f"\n📊 Model Coefficients (feature importance):")
        
        # Sort coefficients by absolute value
        coef_dict = self.training_metadata["model_coefficients"]
        sorted_coefs = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for feature, coef in sorted_coefs[:5]:  # Top 5
            print(f"   {feature:30s}: {coef:>12.2f}")
        
        print(f"\n   Intercept: {self.model.intercept_:,.2f}")
        
        return self.model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features (already scaled)
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before predict. Call train() first.")
        
        return self.model.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Features
            y_true: True target values
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'n_samples': len(y_true)
        }
        
        return metrics
    
    def check_guardrails(self, metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Check if model meets guardrail requirements.
        
        Args:
            metrics: Evaluation metrics
            
        Returns:
            Tuple of (passed, violations)
        """
        violations = []
        
        if metrics['mae'] > GUARDRAILS['mae_threshold']:
            violations.append(
                f"MAE ${metrics['mae']:,.0f} exceeds threshold ${GUARDRAILS['mae_threshold']:,.0f}"
            )
        
        if metrics['rmse'] > GUARDRAILS['rmse_threshold']:
            violations.append(
                f"RMSE ${metrics['rmse']:,.0f} exceeds threshold ${GUARDRAILS['rmse_threshold']:,.0f}"
            )
        
        if metrics['r2'] < GUARDRAILS['r2_threshold']:
            violations.append(
                f"R² {metrics['r2']:.4f} below threshold {GUARDRAILS['r2_threshold']:.4f}"
            )
        
        passed = len(violations) == 0
        return passed, violations
    
    def display_metrics(self, metrics: Dict[str, float], dataset_name: str = "Test"):
        """
        Display evaluation metrics in readable format.
        
        Args:
            metrics: Dictionary with evaluation metrics
            dataset_name: Name of dataset (e.g., "Train", "Test")
        """
        print(f"\n📊 {dataset_name} Set Performance:")
        print(f"   MAE (Mean Absolute Error):  ${metrics['mae']:,.0f}")
        print(f"   RMSE (Root Mean Squared):   ${metrics['rmse']:,.0f}")
        print(f"   R² Score:                   {metrics['r2']:.4f}")
        print(f"   MAPE (Mean Abs % Error):    {metrics['mape']:.2f}%")
        print(f"   Samples:                    {metrics['n_samples']}")
        
        # Check guardrails
        passed, violations = self.check_guardrails(metrics)
        
        if passed:
            print(f"\n✅ All guardrails passed!")
        else:
            print(f"\n⚠️  Guardrail violations:")
            for violation in violations:
                print(f"   ❌ {violation}")
    
    def compare_to_baseline(self, model_metrics: Dict[str, float], 
                           baseline_metrics: Dict[str, float]):
        """
        Compare model performance to baseline.
        
        Args:
            model_metrics: Model evaluation metrics
            baseline_metrics: Baseline evaluation metrics
        """
        print("\n" + "=" * 80)
        print("MODEL vs BASELINE COMPARISON")
        print("=" * 80)
        
        mae_improvement = (baseline_metrics['mae'] - model_metrics['mae']) / baseline_metrics['mae'] * 100
        rmse_improvement = (baseline_metrics['rmse'] - model_metrics['rmse']) / baseline_metrics['rmse'] * 100
        
        print(f"\n📊 MAE:")
        print(f"   Baseline: ${baseline_metrics['mae']:,.0f}")
        print(f"   Model:    ${model_metrics['mae']:,.0f}")
        print(f"   Improvement: {mae_improvement:+.1f}%")
        
        print(f"\n📊 RMSE:")
        print(f"   Baseline: ${baseline_metrics['rmse']:,.0f}")
        print(f"   Model:    ${model_metrics['rmse']:,.0f}")
        print(f"   Improvement: {rmse_improvement:+.1f}%")
        
        print(f"\n📊 R² Score:")
        print(f"   Baseline: {baseline_metrics['r2']:.4f}")
        print(f"   Model:    {model_metrics['r2']:.4f}")
        
        # Decision
        if mae_improvement > 20:
            print(f"\n✅ Model beats baseline by {mae_improvement:.1f}% - WORTH DEPLOYING!")
        elif mae_improvement > 0:
            print(f"\n⚠️  Model beats baseline by only {mae_improvement:.1f}% - marginal improvement")
        else:
            print(f"\n❌ Model WORSE than baseline - DO NOT DEPLOY!")
        
        print("=" * 80)
    
    def save(self, version: str = "v1"):
        """
        Save model and metadata to disk.
        
        Args:
            version: Model version identifier
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving.")
        
        artifacts_dir = PATHS["artifacts_dir"]
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = artifacts_dir / f"model_{version}.pkl"
        joblib.dump(self.model, model_path)
        print(f"\n💾 Saved model to: {model_path}")
        
        # Save metadata
        metadata_path = artifacts_dir / f"model_{version}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.training_metadata, f, indent=2)
        print(f"📝 Saved metadata to: {metadata_path}")
        
        return model_path, metadata_path
    
    @staticmethod
    def load(version: str = "v1") -> Tuple[LinearRegression, Dict[str, Any]]:
        """
        Load model and metadata from disk.
        
        Args:
            version: Model version identifier
            
        Returns:
            Tuple of (model, metadata)
        """
        artifacts_dir = PATHS["artifacts_dir"]
        
        model_path = artifacts_dir / f"model_{version}.pkl"
        metadata_path = artifacts_dir / f"model_{version}_metadata.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        model = joblib.load(model_path)
        print(f"📂 Loaded model from: {model_path}")
        
        # Load metadata
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"📂 Loaded metadata from: {metadata_path}")
        else:
            metadata = {}
        
        return model, metadata


if __name__ == "__main__":
    print("First Model Trainer - Ready to train!")
    print("Use this module in main.py to train the model.")
