"""
Phase 6: Offline Evaluation
Evaluates model on held-out test data before production deployment.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any, List, Tuple
import json
from pathlib import Path
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PATHS, GUARDRAILS


class OfflineEvaluator:
    """
    Evaluates model performance on held-out test data.
    
    Purpose: Fast, cheap, safe way to check model quality before showing to users.
    """
    
    def __init__(self):
        """Initialize offline evaluator."""
        pass
    
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute regression metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Dictionary with metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Median Absolute Error
        median_ae = np.median(np.abs(y_true - y_pred))
        
        # Max error
        max_error = np.max(np.abs(y_true - y_pred))
        
        # Percentage within threshold
        within_50k = np.mean(np.abs(y_true - y_pred) <= 50000) * 100
        within_100k = np.mean(np.abs(y_true - y_pred) <= 100000) * 100
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'median_ae': median_ae,
            'max_error': max_error,
            'within_50k_pct': within_50k,
            'within_100k_pct': within_100k,
            'n_samples': len(y_true)
        }
        
        return metrics
    
    def evaluate_by_slice(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         X: pd.DataFrame, slice_column: str) -> Dict[str, Dict[str, float]]:
        """
        Evaluate performance by data slices.
        
        This catches hidden failures (e.g., works for expensive houses, fails for cheap ones).
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            X: Features (for slicing)
            slice_column: Column to slice by
            
        Returns:
            Dictionary with metrics per slice
        """
        print(f"\n📊 Evaluating by slice: {slice_column}")
        
        slice_metrics = {}
        
        for slice_value in X[slice_column].unique():
            mask = X[slice_column] == slice_value
            
            if mask.sum() > 0:  # Only if slice has samples
                slice_y_true = y_true[mask]
                slice_y_pred = y_pred[mask]
                
                metrics = self.evaluate_regression(slice_y_true, slice_y_pred)
                slice_metrics[str(slice_value)] = metrics
                
                print(f"   {slice_column}={slice_value}: MAE=${metrics['mae']:,.0f}, "
                      f"R²={metrics['r2']:.3f}, n={metrics['n_samples']}")
        
        return slice_metrics
    
    def evaluate_by_price_range(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate performance by price ranges.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Dictionary with metrics per price range
        """
        print(f"\n📊 Evaluating by price range")
        
        # Define price ranges
        ranges = [
            (0, 200000, "Under $200k"),
            (200000, 400000, "$200k-$400k"),
            (400000, 600000, "$400k-$600k"),
            (600000, 1000000, "$600k-$1M"),
            (1000000, float('inf'), "Over $1M")
        ]
        
        range_metrics = {}
        
        for min_price, max_price, label in ranges:
            mask = (y_true >= min_price) & (y_true < max_price)
            
            if mask.sum() > 0:
                range_y_true = y_true[mask]
                range_y_pred = y_pred[mask]
                
                metrics = self.evaluate_regression(range_y_true, range_y_pred)
                range_metrics[label] = metrics
                
                print(f"   {label}: MAE=${metrics['mae']:,.0f}, "
                      f"R²={metrics['r2']:.3f}, n={metrics['n_samples']}")
        
        return range_metrics
    
    def check_guardrails(self, metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Check if metrics meet guardrail requirements.
        
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
        Display metrics in readable format.
        
        Args:
            metrics: Evaluation metrics
            dataset_name: Name of dataset
        """
        print("\n" + "=" * 80)
        print(f"{dataset_name.upper()} SET EVALUATION")
        print("=" * 80)
        
        print(f"\n📊 Core Metrics:")
        print(f"   MAE (Mean Absolute Error):     ${metrics['mae']:,.0f}")
        print(f"   RMSE (Root Mean Squared):      ${metrics['rmse']:,.0f}")
        print(f"   R² Score:                      {metrics['r2']:.4f}")
        print(f"   MAPE (Mean Abs % Error):       {metrics['mape']:.2f}%")
        
        print(f"\n📊 Additional Metrics:")
        print(f"   Median Absolute Error:         ${metrics['median_ae']:,.0f}")
        print(f"   Max Error:                     ${metrics['max_error']:,.0f}")
        print(f"   Within $50k:                   {metrics['within_50k_pct']:.1f}%")
        print(f"   Within $100k:                  {metrics['within_100k_pct']:.1f}%")
        print(f"   Samples:                       {metrics['n_samples']}")
        
        # Check guardrails
        passed, violations = self.check_guardrails(metrics)
        
        print(f"\n🛡️  Guardrail Check:")
        if passed:
            print(f"   ✅ All guardrails passed!")
        else:
            print(f"   ❌ Guardrail violations:")
            for violation in violations:
                print(f"      • {violation}")
        
        print("=" * 80)
    
    def save_metrics(self, metrics: Dict[str, Any], model_version: str, 
                    dataset_name: str = "test"):
        """
        Save evaluation metrics to file.
        
        Args:
            metrics: Evaluation metrics
            model_version: Model version identifier
            dataset_name: Name of dataset
        """
        metrics_dir = PATHS["metrics_dir"]
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        metrics_with_metadata = {
            "model_version": model_version,
            "dataset": dataset_name,
            "evaluated_at": datetime.now().isoformat(),
            "metrics": metrics
        }
        
        # Save to file
        filename = f"metrics_{model_version}_{dataset_name}.json"
        filepath = metrics_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(metrics_with_metadata, f, indent=2)
        
        print(f"\n💾 Saved metrics to: {filepath}")


if __name__ == "__main__":
    print("Offline Evaluator - Ready to evaluate models!")
