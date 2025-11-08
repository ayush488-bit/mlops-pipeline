"""
Phase 3: Leakage Checks
Validates that no future information leaks into training.
Critical for production ML systems.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import FEATURE_CONFIG


class LeakageChecker:
    """
    Checks for various types of data leakage that would invalidate the model.
    
    Types of leakage checked:
    1. Target leakage: Features that directly use the target variable
    2. Train-test contamination: Using test data statistics in training
    3. Temporal leakage: Using future information
    4. Feature leakage: Features not available at decision time
    """
    
    def __init__(self, target_column: str = None):
        """
        Initialize leakage checker.
        
        Args:
            target_column: Name of target variable
        """
        self.target_column = target_column or FEATURE_CONFIG["target"]
        self.leakage_found = []
    
    def check_target_leakage(self, df: pd.DataFrame, feature_columns: List[str]) -> bool:
        """
        Check if any feature column contains or is derived from the target.
        
        Args:
            df: DataFrame with features and target
            feature_columns: List of feature column names
            
        Returns:
            True if no leakage found
        """
        print("\n🔍 Checking for target leakage...")
        
        has_leakage = False
        
        # Check 1: Feature name contains target name
        for feature in feature_columns:
            if self.target_column.lower() in feature.lower():
                self.leakage_found.append({
                    'type': 'target_leakage',
                    'feature': feature,
                    'reason': f"Feature name '{feature}' contains target name '{self.target_column}'"
                })
                has_leakage = True
        
        # Check 2: Perfect correlation with target (suspicious)
        if self.target_column in df.columns:
            for feature in feature_columns:
                if feature in df.columns and feature != self.target_column:
                    # Check correlation
                    if df[feature].dtype in [np.float64, np.int64]:
                        corr = df[feature].corr(df[self.target_column])
                        if abs(corr) > 0.99:  # Nearly perfect correlation
                            self.leakage_found.append({
                                'type': 'target_leakage',
                                'feature': feature,
                                'reason': f"Feature has suspiciously high correlation ({corr:.4f}) with target"
                            })
                            has_leakage = True
        
        if not has_leakage:
            print("   ✅ No target leakage detected")
        else:
            print(f"   ❌ Target leakage detected in {len([l for l in self.leakage_found if l['type'] == 'target_leakage'])} features")
        
        return not has_leakage
    
    def check_train_test_contamination(self, 
                                      train_stats: Dict[str, float],
                                      test_stats: Dict[str, float],
                                      used_stats: Dict[str, float]) -> bool:
        """
        Check if test data statistics were used in training transformations.
        
        Args:
            train_stats: Statistics computed on training data
            test_stats: Statistics computed on test data
            used_stats: Statistics actually used in transformation
            
        Returns:
            True if no contamination found
        """
        print("\n🔍 Checking for train-test contamination...")
        
        has_contamination = False
        
        for feature in used_stats.keys():
            if feature in train_stats and feature in test_stats:
                # Check if used stats match train stats (good)
                # or if they match combined train+test stats (bad)
                train_val = train_stats[feature]
                used_val = used_stats[feature]
                
                # Allow small floating point differences
                if not np.isclose(train_val, used_val, rtol=1e-5):
                    self.leakage_found.append({
                        'type': 'train_test_contamination',
                        'feature': feature,
                        'reason': f"Used stats ({used_val:.4f}) don't match training stats ({train_val:.4f})"
                    })
                    has_contamination = True
        
        if not has_contamination:
            print("   ✅ No train-test contamination detected")
        else:
            print(f"   ❌ Train-test contamination detected")
        
        return not has_contamination
    
    def check_temporal_leakage(self, 
                              df: pd.DataFrame,
                              feature_columns: List[str],
                              timestamp_column: str = None) -> bool:
        """
        Check if features use information from the future.
        
        Args:
            df: DataFrame with features
            feature_columns: List of feature column names
            timestamp_column: Name of timestamp column (if exists)
            
        Returns:
            True if no leakage found
        """
        print("\n🔍 Checking for temporal leakage...")
        
        # For house price prediction, we check logical consistency
        # Example: age_years should be <= current year - build year
        
        has_leakage = False
        
        # Check for negative ages (impossible)
        if 'age_years' in feature_columns and 'age_years' in df.columns:
            negative_ages = (df['age_years'] < 0).sum()
            if negative_ages > 0:
                self.leakage_found.append({
                    'type': 'temporal_leakage',
                    'feature': 'age_years',
                    'reason': f"Found {negative_ages} negative age values (impossible)"
                })
                has_leakage = True
        
        # Check for future dates if timestamp exists
        if timestamp_column and timestamp_column in df.columns:
            future_dates = (pd.to_datetime(df[timestamp_column]) > pd.Timestamp.now()).sum()
            if future_dates > 0:
                self.leakage_found.append({
                    'type': 'temporal_leakage',
                    'feature': timestamp_column,
                    'reason': f"Found {future_dates} future timestamps"
                })
                has_leakage = True
        
        if not has_leakage:
            print("   ✅ No temporal leakage detected")
        else:
            print(f"   ❌ Temporal leakage detected")
        
        return not has_leakage
    
    def check_feature_availability(self, 
                                   feature_columns: List[str],
                                   available_at_decision_time: List[str]) -> bool:
        """
        Check if all features will be available at decision time.
        
        Args:
            feature_columns: List of feature column names
            available_at_decision_time: List of features available in production
            
        Returns:
            True if all features available
        """
        print("\n🔍 Checking feature availability at decision time...")
        
        has_unavailable = False
        
        for feature in feature_columns:
            if feature not in available_at_decision_time:
                self.leakage_found.append({
                    'type': 'feature_unavailable',
                    'feature': feature,
                    'reason': f"Feature '{feature}' not available at decision time"
                })
                has_unavailable = True
        
        if not has_unavailable:
            print("   ✅ All features available at decision time")
        else:
            print(f"   ❌ Some features not available at decision time")
        
        return not has_unavailable
    
    def assert_no_leakage(self) -> Tuple[bool, List[Dict]]:
        """
        Assert that no leakage was found. Raises error if leakage detected.
        
        Returns:
            Tuple of (passed, leakage_list)
        """
        if self.leakage_found:
            print("\n" + "=" * 80)
            print("❌ LEAKAGE DETECTED - MODEL INVALID")
            print("=" * 80)
            
            for i, leak in enumerate(self.leakage_found, 1):
                print(f"\n{i}. {leak['type'].upper()}")
                print(f"   Feature: {leak['feature']}")
                print(f"   Reason: {leak['reason']}")
            
            print("\n" + "=" * 80)
            print("⚠️  FIX ALL LEAKAGE ISSUES BEFORE TRAINING")
            print("=" * 80)
            
            return False, self.leakage_found
        else:
            print("\n" + "=" * 80)
            print("✅ NO LEAKAGE DETECTED - SAFE TO TRAIN")
            print("=" * 80)
            return True, []
    
    def run_all_checks(self, 
                      df: pd.DataFrame,
                      feature_columns: List[str],
                      available_at_decision_time: List[str] = None) -> bool:
        """
        Run all leakage checks.
        
        Args:
            df: DataFrame with features and target
            feature_columns: List of feature column names
            available_at_decision_time: List of features available in production
            
        Returns:
            True if all checks pass
        """
        print("\n" + "=" * 80)
        print("RUNNING LEAKAGE CHECKS")
        print("=" * 80)
        
        # Reset leakage list
        self.leakage_found = []
        
        # Run checks
        self.check_target_leakage(df, feature_columns)
        self.check_temporal_leakage(df, feature_columns)
        
        if available_at_decision_time:
            self.check_feature_availability(feature_columns, available_at_decision_time)
        
        # Assert no leakage
        passed, leakage = self.assert_no_leakage()
        
        return passed


def demo_leakage_checks():
    """Demonstrate leakage checking with examples."""
    print("\n" + "=" * 80)
    print("LEAKAGE CHECKER DEMO")
    print("=" * 80)
    
    # Create sample data
    df = pd.DataFrame({
        'square_feet': [1500, 2000, 2500, 3000],
        'bedrooms': [2, 3, 4, 5],
        'bathrooms': [1, 2, 2, 3],
        'age_years': [10, 20, 5, 15],
        'neighborhood_quality': [7, 8, 9, 8],
        'has_garage': [1, 1, 0, 1],
        'price': [300000, 400000, 500000, 600000]
    })
    
    feature_columns = ['square_feet', 'bedrooms', 'bathrooms', 'age_years', 
                      'neighborhood_quality', 'has_garage']
    
    # All features available at decision time (user provides them)
    available_features = feature_columns.copy()
    
    # Run checks
    checker = LeakageChecker(target_column='price')
    passed = checker.run_all_checks(df, feature_columns, available_features)
    
    if passed:
        print("\n✅ All leakage checks passed! Safe to proceed with training.")
    else:
        print("\n❌ Leakage detected! Fix issues before training.")


if __name__ == "__main__":
    demo_leakage_checks()
