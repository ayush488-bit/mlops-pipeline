"""
Phase 3: Feature Engineering
Implements fit/transform pattern for training-serving parity.
Critical: fit() only on training data, transform() on both train and test.
"""

import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
from typing import Dict, Any, List
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import FEATURE_CONFIG, PATHS


class FeatureEngineer:
    """
    Feature engineering with strict training-serving parity.
    
    Key principle: Statistics computed on training data (fit) are frozen
    and used for all future transformations (transform).
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.is_fitted = False
        self.feature_stats = {}
        self.numeric_features = FEATURE_CONFIG["numeric_features"]
        self.categorical_features = FEATURE_CONFIG["categorical_features"]
        self.target = FEATURE_CONFIG["target"]
        self.scaling_method = FEATURE_CONFIG["scaling_method"]
        
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'FeatureEngineer':
        """
        Fit the feature engineer on training data.
        Computes and stores statistics that will be used in transform.
        
        CRITICAL: Only call this on training data, never on test data!
        
        Args:
            X: Training features
            y: Training target (optional, not used but kept for sklearn compatibility)
            
        Returns:
            self
        """
        print("🔧 Fitting feature engineer on training data...")
        
        # Compute statistics for numeric features
        for feature in self.numeric_features:
            if feature in X.columns:
                self.feature_stats[feature] = {
                    'mean': float(X[feature].mean()),
                    'std': float(X[feature].std()),
                    'min': float(X[feature].min()),
                    'max': float(X[feature].max()),
                    'median': float(X[feature].median())
                }
        
        # For categorical features, store unique values
        for feature in self.categorical_features:
            if feature in X.columns:
                self.feature_stats[feature] = {
                    'unique_values': list(X[feature].unique()),
                    'mode': int(X[feature].mode()[0])
                }
        
        self.is_fitted = True
        print(f"✅ Feature engineer fitted on {len(X)} samples")
        print(f"   Numeric features: {len(self.numeric_features)}")
        print(f"   Categorical features: {len(self.categorical_features)}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using statistics computed during fit.
        
        This method can be called on training, validation, test, or production data.
        It uses the SAME statistics computed during fit() - this ensures
        training-serving parity.
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform. Call fit() first.")
        
        X_transformed = X.copy()
        
        # Handle missing values using training statistics
        for feature in self.numeric_features:
            if feature in X_transformed.columns:
                # Fill missing with median from training data
                fill_value = self.feature_stats[feature]['median']
                X_transformed[feature].fillna(fill_value, inplace=True)
        
        for feature in self.categorical_features:
            if feature in X_transformed.columns:
                # Fill missing with mode from training data
                fill_value = self.feature_stats[feature]['mode']
                X_transformed[feature].fillna(fill_value, inplace=True)
        
        # Scale numeric features using training statistics
        if self.scaling_method == "standard":
            for feature in self.numeric_features:
                if feature in X_transformed.columns:
                    mean = self.feature_stats[feature]['mean']
                    std = self.feature_stats[feature]['std']
                    if std > 0:  # Avoid division by zero
                        X_transformed[feature] = (X_transformed[feature] - mean) / std
                    else:
                        X_transformed[feature] = X_transformed[feature] - mean
        
        elif self.scaling_method == "minmax":
            for feature in self.numeric_features:
                if feature in X_transformed.columns:
                    min_val = self.feature_stats[feature]['min']
                    max_val = self.feature_stats[feature]['max']
                    if max_val > min_val:  # Avoid division by zero
                        X_transformed[feature] = (X_transformed[feature] - min_val) / (max_val - min_val)
        
        # Create feature crosses (interactions)
        if 'square_feet' in X_transformed.columns and 'neighborhood_quality' in X_transformed.columns:
            X_transformed['sqft_x_quality'] = (
                X_transformed['square_feet'] * X_transformed['neighborhood_quality']
            )
        
        if 'bedrooms' in X_transformed.columns and 'bathrooms' in X_transformed.columns:
            X_transformed['bed_bath_ratio'] = (
                X_transformed['bedrooms'] / (X_transformed['bathrooms'] + 1)  # +1 to avoid division by zero
            )
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fit and transform in one step (convenience method for training data).
        
        Args:
            X: Training features
            y: Training target (optional)
            
        Returns:
            Transformed features
        """
        self.fit(X, y)
        return self.transform(X)
    
    def save(self, filepath: Path = None):
        """
        Save feature engineer to disk.
        
        Args:
            filepath: Path to save feature engineer
        """
        if filepath is None:
            filepath = PATHS["transformers_dir"] / "feature_engineer.pkl"
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save state dict instead of the object to avoid pickling issues
        state = {
            'feature_stats': self.feature_stats,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'scaling_method': self.scaling_method,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(state, filepath)
        print(f"💾 Saved feature engineer to: {filepath}")
        
        # Also save feature stats as JSON for inspection
        # Convert numpy types to Python types for JSON serialization
        def convert_to_python_types(obj):
            if isinstance(obj, dict):
                return {k: convert_to_python_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        stats_path = filepath.parent / "feature_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(convert_to_python_types(self.feature_stats), f, indent=2)
        print(f"📝 Saved feature stats to: {stats_path}")
    
    @staticmethod
    def load(filepath: Path = None) -> 'FeatureEngineer':
        """
        Load feature engineer from disk.
        
        Args:
            filepath: Path to load feature engineer from
            
        Returns:
            Loaded feature engineer
        """
        if filepath is None:
            filepath = PATHS["transformers_dir"] / "feature_engineer.pkl"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Feature engineer not found: {filepath}")
        
        # Load state dict and reconstruct object
        state = joblib.load(filepath)
        
        # Create new instance and restore state
        fe = FeatureEngineer()
        fe.numeric_features = state['numeric_features']
        fe.categorical_features = state['categorical_features']
        fe.scaling_method = state['scaling_method']
        fe.feature_stats = state['feature_stats']
        fe.is_fitted = state['is_fitted']
        
        print(f"📂 Loaded feature engineer from: {filepath}")
        return fe
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names after transformation.
        
        Returns:
            List of feature names
        """
        features = self.numeric_features + self.categorical_features
        
        # Add engineered features
        if 'square_feet' in features and 'neighborhood_quality' in features:
            features.append('sqft_x_quality')
        if 'bedrooms' in features and 'bathrooms' in features:
            features.append('bed_bath_ratio')
        
        return features
    
    def display_stats(self):
        """Display feature statistics in readable format."""
        if not self.is_fitted:
            print("⚠️  Feature engineer not fitted yet")
            return
        
        print("\n" + "=" * 80)
        print("FEATURE STATISTICS (from training data)")
        print("=" * 80)
        
        for feature, stats in self.feature_stats.items():
            print(f"\n📊 {feature}:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.2f}")
                else:
                    print(f"   {key}: {value}")
        
        print("\n" + "=" * 80)


def test_training_serving_parity():
    """
    Test that ensures training-serving parity.
    Transform should give identical results when called multiple times.
    """
    print("\n" + "=" * 80)
    print("TESTING TRAINING-SERVING PARITY")
    print("=" * 80)
    
    # Create sample data
    X_train = pd.DataFrame({
        'square_feet': [1500, 2000, 2500],
        'bedrooms': [2, 3, 4],
        'bathrooms': [1, 2, 2],
        'age_years': [10, 20, 5],
        'neighborhood_quality': [7, 8, 9],
        'has_garage': [1, 1, 0]
    })
    
    # Fit on training data
    fe = FeatureEngineer()
    X_train_scaled_1 = fe.fit_transform(X_train)
    
    # Transform again (should be identical)
    X_train_scaled_2 = fe.transform(X_train)
    
    # Check if identical
    if np.allclose(X_train_scaled_1.values, X_train_scaled_2.values):
        print("✅ PASS: Training-serving parity verified!")
        print("   fit_transform() and transform() produce identical results")
    else:
        print("❌ FAIL: Training-serving parity violated!")
        print("   Results differ between fit_transform() and transform()")
    
    print("=" * 80)


if __name__ == "__main__":
    # Run parity test
    test_training_serving_parity()
