"""
Phase 4: Baseline Model
Establishes the simplest possible benchmark that any ML model must beat.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))


class BaselineBuilder:
    """
    Builds baseline predictions using simple heuristics.
    
    For regression: predict the mean of training targets
    For classification: predict the majority class
    
    Purpose: Any ML model must beat this to be worth deploying.
    """
    
    def __init__(self):
        """Initialize baseline builder."""
        self.baseline_value = None
        self.is_fitted = False
    
    def fit(self, y_train: np.ndarray) -> 'BaselineBuilder':
        """
        Fit baseline by computing mean of training targets.
        
        Args:
            y_train: Training target values
            
        Returns:
            self
        """
        self.baseline_value = np.mean(y_train)
        self.is_fitted = True
        
        print(f"📊 Baseline fitted: predict ${self.baseline_value:,.0f} for all houses")
        return self
    
    def predict(self, n_samples: int) -> np.ndarray:
        """
        Generate baseline predictions.
        
        Args:
            n_samples: Number of predictions to generate
            
        Returns:
            Array of predictions (all equal to baseline_value)
        """
        if not self.is_fitted:
            raise ValueError("Baseline must be fitted before predict. Call fit() first.")
        
        return np.full(n_samples, self.baseline_value)
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluate baseline performance.
        
        Args:
            y_true: True target values
            y_pred: Predicted values (if None, generates baseline predictions)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if y_pred is None:
            y_pred = self.predict(len(y_true))
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
        
        return metrics
    
    def display_metrics(self, metrics: Dict[str, float]):
        """
        Display baseline metrics in readable format.
        
        Args:
            metrics: Dictionary with evaluation metrics
        """
        print("\n" + "=" * 80)
        print("BASELINE PERFORMANCE")
        print("=" * 80)
        print(f"Strategy: Predict mean price (${self.baseline_value:,.0f}) for all houses")
        print(f"\n📊 Metrics:")
        print(f"   MAE (Mean Absolute Error):  ${metrics['mae']:,.0f}")
        print(f"   RMSE (Root Mean Squared):   ${metrics['rmse']:,.0f}")
        print(f"   R² Score:                   {metrics['r2']:.4f}")
        print(f"   MAPE (Mean Abs % Error):    {metrics['mape']:.2f}%")
        print("\n💡 Any ML model must beat these numbers to be worth deploying!")
        print("=" * 80)


class AdvancedBaseline:
    """
    More sophisticated baseline using simple rules or heuristics.
    
    For house prices: predict based on neighborhood median or price per sqft.
    """
    
    def __init__(self, strategy: str = 'median_by_neighborhood'):
        """
        Initialize advanced baseline.
        
        Args:
            strategy: Baseline strategy ('median_by_neighborhood', 'price_per_sqft')
        """
        self.strategy = strategy
        self.baseline_map = {}
        self.is_fitted = False
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'AdvancedBaseline':
        """
        Fit advanced baseline.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            self
        """
        if self.strategy == 'median_by_neighborhood':
            # Compute median price per neighborhood
            if 'neighborhood_quality' in X_train.columns:
                for neighborhood in X_train['neighborhood_quality'].unique():
                    mask = X_train['neighborhood_quality'] == neighborhood
                    self.baseline_map[neighborhood] = y_train[mask].median()
                
                # Fallback for unseen neighborhoods
                self.baseline_map['default'] = y_train.median()
                
                print(f"📊 Advanced baseline fitted: median by neighborhood")
                print(f"   Neighborhoods: {len(self.baseline_map) - 1}")
        
        elif self.strategy == 'price_per_sqft':
            # Compute average price per square foot
            if 'square_feet' in X_train.columns:
                price_per_sqft = (y_train / X_train['square_feet']).median()
                self.baseline_map['price_per_sqft'] = price_per_sqft
                
                print(f"📊 Advanced baseline fitted: ${price_per_sqft:.2f} per square foot")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate advanced baseline predictions.
        
        Args:
            X: Features
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Baseline must be fitted before predict. Call fit() first.")
        
        predictions = np.zeros(len(X))
        
        if self.strategy == 'median_by_neighborhood':
            for i, neighborhood in enumerate(X['neighborhood_quality']):
                predictions[i] = self.baseline_map.get(neighborhood, 
                                                      self.baseline_map['default'])
        
        elif self.strategy == 'price_per_sqft':
            price_per_sqft = self.baseline_map['price_per_sqft']
            predictions = X['square_feet'] * price_per_sqft
        
        return predictions
    
    def evaluate(self, X: pd.DataFrame, y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate advanced baseline performance.
        
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
            'mape': mape
        }
        
        return metrics


def demo_baseline():
    """Demonstrate baseline building."""
    print("\n" + "=" * 80)
    print("BASELINE DEMO")
    print("=" * 80)
    
    # Create sample data
    y_train = np.array([300000, 400000, 500000, 600000, 350000])
    y_test = np.array([320000, 450000, 480000])
    
    # Build simple baseline
    baseline = BaselineBuilder()
    baseline.fit(y_train)
    
    # Evaluate
    metrics = baseline.evaluate(y_test)
    baseline.display_metrics(metrics)


if __name__ == "__main__":
    demo_baseline()
