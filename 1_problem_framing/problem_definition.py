"""
Phase 1: Problem Framing & Metrics
This file defines the problem statement, metric ladder, and guardrails.
This is the foundation - everything else builds on this definition.
"""

import json
from pathlib import Path
from typing import Dict, Any


class ProblemFrame:
    """
    Defines the ML problem before any code is written.
    This ensures alignment between business goals and technical implementation.
    """
    
    @staticmethod
    def get_problem_statement() -> Dict[str, str]:
        """
        Define the problem in one clear sentence with all components.
        
        Returns:
            Dictionary with problem components
        """
        return {
            "given": "Historical house data (square feet, bedrooms, bathrooms, age, neighborhood quality, garage)",
            "predict": "Selling price of the house",
            "for": "Real estate platform users (buyers and sellers)",
            "at_time": "Immediately when user views listing (< 500ms)",
            "to_optimize": "Reduce average listing time by 15% and increase user engagement by 20%",
            "full_statement": (
                "Given historical house features, predict the selling price immediately "
                "to help real estate platform users make faster decisions, "
                "thereby reducing listing time by 15%."
            )
        }
    
    @staticmethod
    def get_metric_ladder() -> Dict[str, Any]:
        """
        Define the metric ladder from business to data health.
        Read top-down to understand success, read bottom-up to debug failures.
        
        Returns:
            Dictionary with metric ladder
        """
        return {
            "business_outcome": {
                "primary": "Average listing time reduced by 15%",
                "secondary": [
                    "User engagement (clicks on estimates) increased by 20%",
                    "Conversion rate (listings created) increased by 10%",
                    "User satisfaction score > 4.5/5"
                ]
            },
            "product_metric": {
                "primary": "User clicks on price estimate and proceeds to list house",
                "secondary": [
                    "Time spent on listing page increased",
                    "Price estimate viewed before listing",
                    "User returns to platform within 7 days"
                ]
            },
            "model_metric": {
                "primary": "MAE (Mean Absolute Error) ≤ $50,000",
                "secondary": {
                    "rmse": "RMSE ≤ $75,000",
                    "r2_score": "R² ≥ 0.85",
                    "mape": "MAPE ≤ 15%"
                },
                "slice_metrics": "Performance consistent across price ranges and neighborhoods"
            },
            "data_health": {
                "missing_values": "< 5% missing rate per feature",
                "feature_drift": "Detected and monitored daily",
                "input_distribution": "Monitored by slice (neighborhood, price range)",
                "data_quality": "Schema validation on every batch"
            }
        }
    
    @staticmethod
    def get_guardrails() -> Dict[str, Any]:
        """
        Define hard constraints that must never be violated.
        If any guardrail breaks, system must alert and potentially rollback.
        
        Returns:
            Dictionary with guardrails
        """
        return {
            "north_star": "MAE ≤ $50,000 (primary success metric)",
            "guardrails": {
                "guardrail_1": {
                    "name": "No negative predictions",
                    "rule": "All predictions must be >= $0",
                    "action_if_violated": "Use fallback prediction (neighborhood median)"
                },
                "guardrail_2": {
                    "name": "Latency constraint",
                    "rule": "P99 latency < 500ms",
                    "action_if_violated": "Alert on-call engineer, investigate performance"
                },
                "guardrail_3": {
                    "name": "Error rate",
                    "rule": "< 1% of predictions return NaN or error",
                    "action_if_violated": "Rollback to previous model immediately"
                },
                "guardrail_4": {
                    "name": "Prediction range",
                    "rule": "Predictions must be between $0 and $10M",
                    "action_if_violated": "Clip to range and log anomaly"
                }
            },
            "hard_constraints": [
                "Model must be explainable (linear model preferred)",
                "Training data must be < 90 days old",
                "All features must be available at decision time (no future leakage)",
                "Model must beat baseline by at least 20%"
            ]
        }
    
    @staticmethod
    def get_failure_analysis() -> Dict[str, Any]:
        """
        Define what happens when the model fails and how to respond.
        
        Returns:
            Dictionary with failure scenarios and responses
        """
        return {
            "failure_scenarios": {
                "scenario_1": {
                    "failure": "Model predicts price 50% too high",
                    "impact": "User lists house at inflated price, doesn't sell",
                    "severity": "High - damages user trust",
                    "rollback_path": "Revert to previous model within 5 minutes",
                    "owner": "ML Engineer on-call"
                },
                "scenario_2": {
                    "failure": "Model predicts price 50% too low",
                    "impact": "User underprices house, loses money",
                    "severity": "Critical - financial loss to user",
                    "rollback_path": "Immediate rollback + incident review",
                    "owner": "ML Lead + Product Manager"
                },
                "scenario_3": {
                    "failure": "Model latency > 2 seconds",
                    "impact": "Poor user experience, page abandonment",
                    "severity": "Medium - affects engagement",
                    "rollback_path": "Switch to cached predictions",
                    "owner": "Platform Engineer"
                },
                "scenario_4": {
                    "failure": "Model returns NaN for 5% of requests",
                    "impact": "Users see errors, lose confidence",
                    "severity": "High - breaks user experience",
                    "rollback_path": "Use fallback model (neighborhood median)",
                    "owner": "ML Engineer on-call"
                }
            },
            "monitoring_alerts": [
                "MAE increases by > 20% compared to baseline",
                "Error rate > 1%",
                "Latency P99 > 500ms",
                "Feature drift detected (> 20% change)",
                "Prediction distribution shifts significantly"
            ]
        }
    
    @staticmethod
    def save_to_json(output_path: Path = None):
        """
        Save all problem framing information to JSON file.
        
        Args:
            output_path: Path to save JSON file
        """
        if output_path is None:
            output_path = Path(__file__).parent / "metric_ladder.json"
        
        problem_frame = {
            "problem_statement": ProblemFrame.get_problem_statement(),
            "metric_ladder": ProblemFrame.get_metric_ladder(),
            "guardrails": ProblemFrame.get_guardrails(),
            "failure_analysis": ProblemFrame.get_failure_analysis(),
            "version": "1.0",
            "last_updated": "2025-01-15"
        }
        
        with open(output_path, 'w') as f:
            json.dump(problem_frame, f, indent=2)
        
        print(f"✅ Problem framing saved to: {output_path}")
        return problem_frame
    
    @staticmethod
    def display_summary():
        """Display a human-readable summary of the problem framing."""
        problem = ProblemFrame.get_problem_statement()
        metrics = ProblemFrame.get_metric_ladder()
        guardrails = ProblemFrame.get_guardrails()
        
        print("=" * 80)
        print("PROBLEM FRAMING SUMMARY")
        print("=" * 80)
        print(f"\n📋 PROBLEM STATEMENT:")
        print(f"   {problem['full_statement']}")
        print(f"\n🎯 BUSINESS GOAL:")
        print(f"   {metrics['business_outcome']['primary']}")
        print(f"\n📊 MODEL METRIC:")
        print(f"   {metrics['model_metric']['primary']}")
        print(f"\n🛡️  NORTH STAR GUARDRAIL:")
        print(f"   {guardrails['north_star']}")
        print("\n" + "=" * 80)


if __name__ == "__main__":
    # Display summary
    ProblemFrame.display_summary()
    
    # Save to JSON
    ProblemFrame.save_to_json()
