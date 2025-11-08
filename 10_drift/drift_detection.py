"""
Phase 10: Drift Detection
Detects when data or model performance changes over time.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, Tuple, List
from datetime import datetime
from pathlib import Path
import json
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PATHS


class DriftDetector:
    """
    Detects three types of drift:
    1. Covariate Shift: Input features change
    2. Concept Drift: Relationship between X and y changes
    3. Label Drift: Target distribution changes
    """
    
    def __init__(self, baseline_data: pd.DataFrame = None):
        """
        Initialize drift detector.
        
        Args:
            baseline_data: Training data to use as baseline
        """
        self.baseline_data = baseline_data
        self.baseline_stats = None
        
        if baseline_data is not None:
            self.compute_baseline_stats()
    
    def compute_baseline_stats(self):
        """Compute statistics on baseline data."""
        self.baseline_stats = {}
        
        for col in self.baseline_data.columns:
            if self.baseline_data[col].dtype in [np.float64, np.int64]:
                self.baseline_stats[col] = {
                    'mean': float(self.baseline_data[col].mean()),
                    'std': float(self.baseline_data[col].std()),
                    'min': float(self.baseline_data[col].min()),
                    'max': float(self.baseline_data[col].max()),
                    'median': float(self.baseline_data[col].median())
                }
    
    def detect_covariate_shift(self,
                               current_data: pd.DataFrame,
                               feature: str,
                               threshold: float = 0.05) -> Tuple[bool, float, str]:
        """
        Detect if feature distribution has changed (Kolmogorov-Smirnov test).
        
        Args:
            current_data: Current production data
            feature: Feature name to check
            threshold: P-value threshold (default 0.05)
            
        Returns:
            Tuple of (drift_detected, p_value, interpretation)
        """
        if self.baseline_data is None:
            return False, 1.0, "No baseline data"
        
        if feature not in self.baseline_data.columns or feature not in current_data.columns:
            return False, 1.0, f"Feature {feature} not found"
        
        # Kolmogorov-Smirnov test
        baseline_values = self.baseline_data[feature].dropna()
        current_values = current_data[feature].dropna()
        
        statistic, p_value = stats.ks_2samp(baseline_values, current_values)
        
        drift_detected = p_value < threshold
        
        if drift_detected:
            interpretation = f"⚠️  DRIFT DETECTED in {feature} (p={p_value:.4f})"
        else:
            interpretation = f"✅ No drift in {feature} (p={p_value:.4f})"
        
        return drift_detected, p_value, interpretation
    
    def detect_all_covariate_shifts(self,
                                    current_data: pd.DataFrame,
                                    threshold: float = 0.05) -> Dict[str, Any]:
        """
        Check all features for covariate shift.
        
        Args:
            current_data: Current production data
            threshold: P-value threshold
            
        Returns:
            Dictionary with drift results for each feature
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'features_checked': [],
            'drifted_features': [],
            'drift_details': {}
        }
        
        numeric_features = [col for col in self.baseline_data.columns 
                          if self.baseline_data[col].dtype in [np.float64, np.int64]]
        
        for feature in numeric_features:
            drift_detected, p_value, interpretation = self.detect_covariate_shift(
                current_data, feature, threshold
            )
            
            results['features_checked'].append(feature)
            results['drift_details'][feature] = {
                'drift_detected': drift_detected,
                'p_value': p_value,
                'interpretation': interpretation
            }
            
            if drift_detected:
                results['drifted_features'].append(feature)
        
        results['drift_detected'] = len(results['drifted_features']) > 0
        results['drift_percentage'] = len(results['drifted_features']) / len(results['features_checked']) * 100
        
        return results
    
    def detect_concept_drift(self,
                            recent_errors: List[float],
                            historical_mae: float,
                            threshold_pct: float = 20.0) -> Tuple[bool, float, str]:
        """
        Detect if model performance has degraded (concept drift).
        
        Args:
            recent_errors: Recent prediction errors
            historical_mae: Historical MAE from training
            threshold_pct: Percentage increase threshold
            
        Returns:
            Tuple of (drift_detected, current_mae, interpretation)
        """
        if not recent_errors:
            return False, 0.0, "No recent errors available"
        
        current_mae = np.mean(np.abs(recent_errors))
        mae_increase_pct = ((current_mae - historical_mae) / historical_mae) * 100
        
        drift_detected = mae_increase_pct > threshold_pct
        
        if drift_detected:
            interpretation = f"⚠️  CONCEPT DRIFT: MAE increased by {mae_increase_pct:.1f}% (from ${historical_mae:,.0f} to ${current_mae:,.0f})"
        else:
            interpretation = f"✅ No concept drift: MAE {mae_increase_pct:+.1f}% (current ${current_mae:,.0f})"
        
        return drift_detected, current_mae, interpretation
    
    def detect_label_drift(self,
                          current_targets: pd.Series,
                          threshold: float = 0.05) -> Tuple[bool, float, str]:
        """
        Detect if target distribution has changed.
        
        Args:
            current_targets: Current target values
            threshold: P-value threshold
            
        Returns:
            Tuple of (drift_detected, p_value, interpretation)
        """
        if self.baseline_data is None or 'price' not in self.baseline_data.columns:
            return False, 1.0, "No baseline target data"
        
        baseline_targets = self.baseline_data['price'].dropna()
        
        # Kolmogorov-Smirnov test
        statistic, p_value = stats.ks_2samp(baseline_targets, current_targets.dropna())
        
        drift_detected = p_value < threshold
        
        baseline_mean = baseline_targets.mean()
        current_mean = current_targets.mean()
        change_pct = ((current_mean - baseline_mean) / baseline_mean) * 100
        
        if drift_detected:
            interpretation = f"⚠️  LABEL DRIFT: Target distribution changed (p={p_value:.4f}, mean {change_pct:+.1f}%)"
        else:
            interpretation = f"✅ No label drift (p={p_value:.4f}, mean {change_pct:+.1f}%)"
        
        return drift_detected, p_value, interpretation
    
    def run_full_drift_analysis(self,
                                current_data: pd.DataFrame,
                                recent_errors: List[float] = None,
                                historical_mae: float = None) -> Dict[str, Any]:
        """
        Run complete drift analysis.
        
        Args:
            current_data: Current production data
            recent_errors: Recent prediction errors
            historical_mae: Historical MAE
            
        Returns:
            Complete drift analysis report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'covariate_shift': None,
            'concept_drift': None,
            'label_drift': None,
            'overall_drift_detected': False,
            'recommendations': []
        }
        
        # 1. Covariate Shift
        print("\n🔍 Checking for covariate shift...")
        covariate_results = self.detect_all_covariate_shifts(current_data)
        report['covariate_shift'] = covariate_results
        
        if covariate_results['drift_detected']:
            report['overall_drift_detected'] = True
            report['recommendations'].append(
                f"Retrain model - {len(covariate_results['drifted_features'])} features drifted"
            )
        
        # 2. Concept Drift
        if recent_errors and historical_mae:
            print("\n🔍 Checking for concept drift...")
            drift_detected, current_mae, interpretation = self.detect_concept_drift(
                recent_errors, historical_mae
            )
            report['concept_drift'] = {
                'drift_detected': drift_detected,
                'current_mae': current_mae,
                'historical_mae': historical_mae,
                'interpretation': interpretation
            }
            
            if drift_detected:
                report['overall_drift_detected'] = True
                report['recommendations'].append("Model performance degraded - retrain immediately")
        
        # 3. Label Drift
        if 'price' in current_data.columns:
            print("\n🔍 Checking for label drift...")
            drift_detected, p_value, interpretation = self.detect_label_drift(current_data['price'])
            report['label_drift'] = {
                'drift_detected': drift_detected,
                'p_value': p_value,
                'interpretation': interpretation
            }
            
            if drift_detected:
                report['overall_drift_detected'] = True
                report['recommendations'].append("Target distribution changed - review data collection")
        
        return report
    
    def display_drift_report(self, report: Dict[str, Any]):
        """Display drift report in readable format."""
        print("\n" + "=" * 80)
        print("DRIFT DETECTION REPORT")
        print("=" * 80)
        
        print(f"\n📅 Timestamp: {report['timestamp']}")
        print(f"\n🚨 Overall Drift: {'DETECTED' if report['overall_drift_detected'] else 'NOT DETECTED'}")
        
        # Covariate Shift
        if report['covariate_shift']:
            cov = report['covariate_shift']
            print(f"\n📊 Covariate Shift:")
            print(f"   Features checked: {len(cov['features_checked'])}")
            print(f"   Drifted features: {len(cov['drifted_features'])}")
            if cov['drifted_features']:
                print(f"   Affected: {', '.join(cov['drifted_features'])}")
        
        # Concept Drift
        if report['concept_drift']:
            con = report['concept_drift']
            print(f"\n📉 Concept Drift:")
            print(f"   {con['interpretation']}")
        
        # Label Drift
        if report['label_drift']:
            lab = report['label_drift']
            print(f"\n🎯 Label Drift:")
            print(f"   {lab['interpretation']}")
        
        # Recommendations
        if report['recommendations']:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "=" * 80)


def demo_drift_detection():
    """Demo drift detection."""
    print("\n" + "=" * 80)
    print("DRIFT DETECTION DEMO")
    print("=" * 80)
    
    # Load baseline data
    data_path = PATHS["data_dir"] / "data_raw_v1.csv"
    if data_path.exists():
        baseline_data = pd.read_csv(data_path)
        print(f"\n📂 Loaded baseline data: {len(baseline_data)} samples")
        
        # Create detector
        detector = DriftDetector(baseline_data)
        
        # Simulate current data with some drift
        current_data = baseline_data.sample(n=500, random_state=42).copy()
        
        # Introduce drift in square_feet (increase by 20%)
        current_data['square_feet'] = current_data['square_feet'] * 1.2
        
        # Run full analysis
        report = detector.run_full_drift_analysis(
            current_data,
            recent_errors=[25000] * 100,  # Simulated errors
            historical_mae=23000
        )
        
        # Display report
        detector.display_drift_report(report)
    else:
        print(f"\n⚠️  Baseline data not found. Run main pipeline first.")


if __name__ == "__main__":
    demo_drift_detection()
