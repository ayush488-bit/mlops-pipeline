"""
Phase 5: Validation Gates
Validates data before training and serving to prevent bad data from entering the system.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import GUARDRAILS
sys.path.append(str(Path(__file__).parent))
from schema import SCHEMA


class DataValidator:
    """
    Validates data against schema before training or serving.
    
    Two validation gates:
    1. validate_before_training: Strict checks, fail if violated
    2. validate_before_serving: Graceful degradation, use fallback if violated
    """
    
    def __init__(self, schema: Dict[str, Any] = None):
        """
        Initialize validator.
        
        Args:
            schema: Data schema (uses default if None)
        """
        self.schema = schema or SCHEMA
    
    def validate_before_training(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate data before training.
        
        Checks:
        - All required columns present
        - No column has > 5% missing values
        - Numeric columns in valid range
        - Correct data types
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        print("\n🔍 Validating data before training...")
        errors = []
        
        # Check 1: All required columns present
        for field, rules in self.schema.items():
            if rules.get('required', False) and field not in df.columns:
                errors.append(f"Missing required column: {field}")
        
        # Check 2: Missing value rate
        max_missing_rate = GUARDRAILS.get('max_missing_rate', 0.05)
        for col in df.columns:
            missing_rate = df[col].isnull().sum() / len(df)
            if missing_rate > max_missing_rate:
                errors.append(
                    f"Column '{col}' has {missing_rate*100:.1f}% missing values "
                    f"(threshold: {max_missing_rate*100:.1f}%)"
                )
        
        # Check 3: Value ranges
        for field, rules in self.schema.items():
            if field in df.columns:
                # Check min
                if 'min' in rules:
                    below_min = (df[field] < rules['min']).sum()
                    if below_min > 0:
                        errors.append(
                            f"Column '{field}' has {below_min} values below minimum {rules['min']}"
                        )
                
                # Check max
                if 'max' in rules:
                    above_max = (df[field] > rules['max']).sum()
                    if above_max > 0:
                        errors.append(
                            f"Column '{field}' has {above_max} values above maximum {rules['max']}"
                        )
        
        # Check 4: Data types
        for field, rules in self.schema.items():
            if field in df.columns:
                expected_type = rules.get('type')
                actual_dtype = str(df[field].dtype)
                
                if expected_type == 'int' and not ('int' in actual_dtype):
                    errors.append(f"Column '{field}' should be int, got {actual_dtype}")
                elif expected_type == 'float' and not ('float' in actual_dtype or 'int' in actual_dtype):
                    errors.append(f"Column '{field}' should be float, got {actual_dtype}")
        
        # Display results
        is_valid = len(errors) == 0
        
        if is_valid:
            print("   ✅ All validation checks passed!")
        else:
            print(f"   ❌ Found {len(errors)} validation errors:")
            for i, error in enumerate(errors, 1):
                print(f"      {i}. {error}")
        
        return is_valid, errors
    
    def validate_before_serving(self, X: pd.DataFrame) -> Tuple[bool, List[str], pd.DataFrame]:
        """
        Validate data before serving predictions.
        
        More lenient than training validation - tries to fix issues.
        
        Args:
            X: Features to validate
            
        Returns:
            Tuple of (is_valid, list_of_warnings, cleaned_X)
        """
        warnings = []
        X_cleaned = X.copy()
        
        # Check 1: All required features present
        for field, rules in self.schema.items():
            if field == 'price':  # Skip target
                continue
            
            if rules.get('required', False) and field not in X_cleaned.columns:
                warnings.append(f"Missing required feature: {field}")
                # Add column with median value (would need to load from training stats)
                X_cleaned[field] = 0  # Placeholder
        
        # Check 2: Handle missing values
        for col in X_cleaned.columns:
            missing_count = X_cleaned[col].isnull().sum()
            if missing_count > 0:
                warnings.append(f"Feature '{col}' has {missing_count} missing values (will be imputed)")
                # Fill with median (in production, use training statistics)
                X_cleaned[col].fillna(X_cleaned[col].median(), inplace=True)
        
        # Check 3: Clip values to valid range
        for field, rules in self.schema.items():
            if field in X_cleaned.columns:
                if 'min' in rules:
                    below_min = (X_cleaned[field] < rules['min']).sum()
                    if below_min > 0:
                        warnings.append(f"Clipping {below_min} values in '{field}' to minimum {rules['min']}")
                        X_cleaned[field] = X_cleaned[field].clip(lower=rules['min'])
                
                if 'max' in rules:
                    above_max = (X_cleaned[field] > rules['max']).sum()
                    if above_max > 0:
                        warnings.append(f"Clipping {above_max} values in '{field}' to maximum {rules['max']}")
                        X_cleaned[field] = X_cleaned[field].clip(upper=rules['max'])
        
        # Check 4: Handle infinite values
        for col in X_cleaned.select_dtypes(include=[np.number]).columns:
            inf_count = np.isinf(X_cleaned[col]).sum()
            if inf_count > 0:
                warnings.append(f"Feature '{col}' has {inf_count} infinite values (will be replaced)")
                X_cleaned[col].replace([np.inf, -np.inf], X_cleaned[col].median(), inplace=True)
        
        is_valid = len(warnings) == 0
        
        return is_valid, warnings, X_cleaned
    
    def validate_predictions(self, predictions: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Validate predictions before returning to user.
        
        Args:
            predictions: Model predictions
            
        Returns:
            Tuple of (cleaned_predictions, list_of_warnings)
        """
        warnings = []
        cleaned_predictions = predictions.copy()
        
        # Check for NaN
        nan_count = np.isnan(cleaned_predictions).sum()
        if nan_count > 0:
            warnings.append(f"Found {nan_count} NaN predictions (will use fallback)")
            # Replace NaN with median of valid predictions
            valid_median = np.nanmedian(cleaned_predictions)
            cleaned_predictions = np.nan_to_num(cleaned_predictions, nan=valid_median)
        
        # Check for infinite
        inf_count = np.isinf(cleaned_predictions).sum()
        if inf_count > 0:
            warnings.append(f"Found {inf_count} infinite predictions (will clip)")
            cleaned_predictions = np.clip(cleaned_predictions, 
                                        GUARDRAILS['min_prediction'],
                                        GUARDRAILS['max_prediction'])
        
        # Check for negative prices
        negative_count = (cleaned_predictions < 0).sum()
        if negative_count > 0:
            warnings.append(f"Found {negative_count} negative predictions (will clip to 0)")
            cleaned_predictions = np.maximum(cleaned_predictions, 0)
        
        # Check for unrealistic prices
        too_high = (cleaned_predictions > GUARDRAILS['max_prediction']).sum()
        if too_high > 0:
            warnings.append(f"Found {too_high} predictions above ${GUARDRAILS['max_prediction']:,} (will clip)")
            cleaned_predictions = np.minimum(cleaned_predictions, GUARDRAILS['max_prediction'])
        
        return cleaned_predictions, warnings
    
    def display_validation_summary(self, is_valid: bool, issues: List[str], 
                                   validation_type: str = "training"):
        """
        Display validation summary.
        
        Args:
            is_valid: Whether validation passed
            issues: List of errors or warnings
            validation_type: Type of validation ("training" or "serving")
        """
        print("\n" + "=" * 80)
        print(f"VALIDATION SUMMARY ({validation_type.upper()})")
        print("=" * 80)
        
        if is_valid:
            print("✅ All validation checks passed!")
        else:
            print(f"⚠️  Found {len(issues)} issues:")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")
        
        print("=" * 80)


def demo_validation():
    """Demonstrate validation."""
    print("\n" + "=" * 80)
    print("VALIDATION DEMO")
    print("=" * 80)
    
    # Create sample data with issues
    df_bad = pd.DataFrame({
        'square_feet': [1500, -500, 2500, np.nan],  # Negative and missing
        'bedrooms': [2, 3, 4, 2],
        'bathrooms': [1, 2, 2, 1],
        'age_years': [10, 20, 5, 15],
        'neighborhood_quality': [7, 8, 9, 8],
        'has_garage': [1, 1, 0, 1],
        'price': [300000, 400000, 500000, 350000]
    })
    
    validator = DataValidator()
    
    # Validate before training
    is_valid, errors = validator.validate_before_training(df_bad)
    validator.display_validation_summary(is_valid, errors, "training")
    
    # Create good data
    df_good = pd.DataFrame({
        'square_feet': [1500, 2000, 2500, 3000],
        'bedrooms': [2, 3, 4, 5],
        'bathrooms': [1, 2, 2, 3],
        'age_years': [10, 20, 5, 15],
        'neighborhood_quality': [7, 8, 9, 8],
        'has_garage': [1, 1, 0, 1],
        'price': [300000, 400000, 500000, 600000]
    })
    
    is_valid, errors = validator.validate_before_training(df_good)
    validator.display_validation_summary(is_valid, errors, "training")


if __name__ == "__main__":
    demo_validation()
