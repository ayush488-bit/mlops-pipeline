"""
Main Orchestration Script
Runs the complete MLOps pipeline end-to-end.

Usage:
    python main.py --mode train     # Train full pipeline
    python main.py --mode evaluate  # Evaluate existing model
    python main.py --mode serve     # Start serving API (Phase 8)
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import all phases
from config import DATA_CONFIG, PATHS, FEATURE_CONFIG

# Import from each phase using underscore naming
import importlib.util

# Phase 1
spec = importlib.util.spec_from_file_location("problem_definition", 
    Path(__file__).parent / "1_problem_framing" / "problem_definition.py")
problem_def = importlib.util.module_from_spec(spec)
spec.loader.exec_module(problem_def)
ProblemFrame = problem_def.ProblemFrame

# Phase 2
spec = importlib.util.spec_from_file_location("data_collection",
    Path(__file__).parent / "2_data_management" / "data_collection.py")
data_coll = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_coll)
DataCollector = data_coll.DataCollector

spec = importlib.util.spec_from_file_location("data_versioning",
    Path(__file__).parent / "2_data_management" / "data_versioning.py")
data_vers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_vers)
DataVersioner = data_vers.DataVersioner

# Phase 3
spec = importlib.util.spec_from_file_location("feature_engineering",
    Path(__file__).parent / "3_features" / "feature_engineering.py")
feat_eng = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feat_eng)
FeatureEngineer = feat_eng.FeatureEngineer

spec = importlib.util.spec_from_file_location("leakage_checks",
    Path(__file__).parent / "3_features" / "leakage_checks.py")
leak_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(leak_check)
LeakageChecker = leak_check.LeakageChecker

# Phase 4
spec = importlib.util.spec_from_file_location("baseline",
    Path(__file__).parent / "4_model" / "baseline.py")
baseline_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline_mod)
BaselineBuilder = baseline_mod.BaselineBuilder

spec = importlib.util.spec_from_file_location("first_model",
    Path(__file__).parent / "4_model" / "first_model.py")
first_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(first_mod)
FirstModelTrainer = first_mod.FirstModelTrainer

# Phase 5
spec = importlib.util.spec_from_file_location("validation_gates",
    Path(__file__).parent / "5_validation" / "validation_gates.py")
valid_gates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(valid_gates)
DataValidator = valid_gates.DataValidator

# Phase 6
spec = importlib.util.spec_from_file_location("offline_eval",
    Path(__file__).parent / "6_evaluation" / "offline_eval.py")
offline_ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(offline_ev)
OfflineEvaluator = offline_ev.OfflineEvaluator


def phase_1_problem_framing():
    """Phase 1: Define the problem and metrics."""
    print("\n" + "=" * 80)
    print("PHASE 1: PROBLEM FRAMING")
    print("=" * 80)
    
    # Display problem definition
    ProblemFrame.display_summary()
    
    # Save metric ladder
    ProblemFrame.save_to_json()
    
    print("\n✅ Phase 1 complete!")
    return True


def phase_2_data_collection():
    """Phase 2: Collect and version data."""
    print("\n" + "=" * 80)
    print("PHASE 2: DATA COLLECTION & VERSIONING")
    print("=" * 80)
    
    # Initialize collector
    collector = DataCollector(
        n_samples=DATA_CONFIG["n_samples"],
        random_seed=DATA_CONFIG["random_seed"]
    )
    
    # Generate synthetic data
    df = collector.generate_synthetic_data()
    
    # Save with versioning
    data_path, lineage_path = collector.save_data_with_lineage(
        df, version="v1", stage="raw"
    )
    
    # Verify integrity
    collector.verify_data_integrity(df, version="v1", stage="raw")
    
    # Register in version manifest
    versioner = DataVersioner()
    metadata = collector.compute_metadata(df)
    checksum = collector.compute_checksum(df)
    
    versioner.register_version(
        version="v1",
        stage="raw",
        data_path=str(data_path),
        lineage_path=str(lineage_path),
        checksum=checksum,
        metadata=metadata
    )
    
    print("\n✅ Phase 2 complete!")
    return df


def phase_3_feature_engineering(X_train, X_test):
    """Phase 3: Engineer features with leakage checks."""
    print("\n" + "=" * 80)
    print("PHASE 3: FEATURE ENGINEERING")
    print("=" * 80)
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Fit on training data ONLY
    print("\n🔧 Fitting feature engineer on training data...")
    X_train_scaled = fe.fit_transform(X_train)
    
    # Transform test data using training statistics
    print("🔧 Transforming test data using training statistics...")
    X_test_scaled = fe.transform(X_test)
    
    # Display feature statistics
    fe.display_stats()
    
    # Save feature engineer
    fe.save()
    
    # Verify training-serving parity
    print("\n🔍 Verifying training-serving parity...")
    X_train_check = fe.transform(X_train)
    if np.allclose(X_train_scaled.values, X_train_check.values):
        print("✅ Training-serving parity verified!")
    else:
        print("❌ WARNING: Training-serving parity violated!")
    
    print("\n✅ Phase 3 complete!")
    return X_train_scaled, X_test_scaled, fe


def phase_4_baseline_and_model(X_train, X_test, y_train, y_test):
    """Phase 4: Build baseline and train first model."""
    print("\n" + "=" * 80)
    print("PHASE 4: BASELINE & FIRST MODEL")
    print("=" * 80)
    
    # Build baseline
    print("\n📊 Building baseline...")
    baseline = BaselineBuilder()
    baseline.fit(y_train)
    
    # Evaluate baseline
    baseline_metrics = baseline.evaluate(y_test)
    baseline.display_metrics(baseline_metrics)
    
    # Train first model
    print("\n🤖 Training first model...")
    trainer = FirstModelTrainer()
    model = trainer.train(X_train, y_train, version="v1")
    
    # Evaluate model on train set
    print("\n📊 Evaluating on training set...")
    train_metrics = trainer.evaluate(X_train, y_train)
    trainer.display_metrics(train_metrics, "Train")
    
    # Evaluate model on test set
    print("\n📊 Evaluating on test set...")
    test_metrics = trainer.evaluate(X_test, y_test)
    trainer.display_metrics(test_metrics, "Test")
    
    # Compare to baseline
    trainer.compare_to_baseline(test_metrics, baseline_metrics)
    
    # Save model
    trainer.save(version="v1")
    
    print("\n✅ Phase 4 complete!")
    return baseline_metrics, test_metrics, model, trainer


def phase_5_validation(df):
    """Phase 5: Validate data quality."""
    print("\n" + "=" * 80)
    print("PHASE 5: DATA VALIDATION")
    print("=" * 80)
    
    # Initialize validator
    validator = DataValidator()
    
    # Validate before training
    is_valid, errors = validator.validate_before_training(df)
    validator.display_validation_summary(is_valid, errors, "training")
    
    if not is_valid:
        print("\n❌ Data validation failed! Fix errors before proceeding.")
        return False
    
    print("\n✅ Phase 5 complete!")
    return True


def phase_6_evaluation(y_test, y_pred, X_test):
    """Phase 6: Comprehensive offline evaluation."""
    print("\n" + "=" * 80)
    print("PHASE 6: OFFLINE EVALUATION")
    print("=" * 80)
    
    # Initialize evaluator
    evaluator = OfflineEvaluator()
    
    # Overall metrics
    metrics = evaluator.evaluate_regression(y_test, y_pred)
    evaluator.display_metrics(metrics, "Test")
    
    # Evaluate by price range
    range_metrics = evaluator.evaluate_by_price_range(y_test, y_pred)
    
    # Evaluate by neighborhood (if available)
    if 'neighborhood_quality' in X_test.columns:
        slice_metrics = evaluator.evaluate_by_slice(
            y_test, y_pred, X_test, 'neighborhood_quality'
        )
    
    # Save metrics
    evaluator.save_metrics(metrics, model_version="v1", dataset_name="test")
    
    print("\n✅ Phase 6 complete!")
    return metrics


def run_leakage_checks(df, feature_columns):
    """Run comprehensive leakage checks."""
    print("\n" + "=" * 80)
    print("LEAKAGE CHECKS")
    print("=" * 80)
    
    checker = LeakageChecker(target_column='price')
    
    # All features should be available at decision time
    available_features = feature_columns.copy()
    
    # Run all checks
    passed = checker.run_all_checks(df, feature_columns, available_features)
    
    if not passed:
        print("\n❌ Leakage detected! Fix before proceeding.")
        sys.exit(1)
    
    return passed


def main_train():
    """Run complete training pipeline."""
    print("\n" + "=" * 100)
    print(" " * 30 + "MLOPS TRAINING PIPELINE")
    print("=" * 100)
    
    # Phase 1: Problem Framing
    phase_1_problem_framing()
    
    # Phase 2: Data Collection
    df = phase_2_data_collection()
    
    # Phase 5: Validation (before training)
    if not phase_5_validation(df):
        print("\n❌ Pipeline failed at validation stage")
        return
    
    # Prepare features and target
    feature_columns = FEATURE_CONFIG["numeric_features"] + FEATURE_CONFIG["categorical_features"]
    target_column = FEATURE_CONFIG["target"]
    
    X = df[feature_columns]
    y = df[target_column].values
    
    # Run leakage checks
    run_leakage_checks(df, feature_columns)
    
    # Split data
    print("\n📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=DATA_CONFIG["train_test_split"],
        random_state=DATA_CONFIG["random_seed"]
    )
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test:  {len(X_test)} samples")
    
    # Phase 3: Feature Engineering
    X_train_scaled, X_test_scaled, fe = phase_3_feature_engineering(X_train, X_test)
    
    # Phase 4: Baseline and Model
    baseline_metrics, test_metrics, model, trainer = phase_4_baseline_and_model(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # Phase 6: Evaluation
    y_pred = model.predict(X_test_scaled)
    
    # Get original X_test for slicing (before scaling)
    X_test_original = X_test.copy()
    
    phase_6_evaluation(y_test, y_pred, X_test_original)
    
    # Final summary
    print("\n" + "=" * 100)
    print(" " * 35 + "PIPELINE COMPLETE!")
    print("=" * 100)
    print(f"\n✅ Model trained and evaluated successfully!")
    print(f"\n📊 Final Test Metrics:")
    print(f"   MAE:  ${test_metrics['mae']:,.0f}")
    print(f"   RMSE: ${test_metrics['rmse']:,.0f}")
    print(f"   R²:   {test_metrics['r2']:.4f}")
    
    improvement = (baseline_metrics['mae'] - test_metrics['mae']) / baseline_metrics['mae'] * 100
    print(f"\n💡 Model beats baseline by {improvement:.1f}%")
    
    print(f"\n📁 Artifacts saved to:")
    print(f"   Model: {PATHS['artifacts_dir']}/model_v1.pkl")
    print(f"   Features: {PATHS['transformers_dir']}/feature_engineer.pkl")
    print(f"   Metrics: {PATHS['metrics_dir']}/metrics_v1_test.json")
    
    print("\n🚀 Next steps:")
    print("   1. Review metrics and ensure guardrails are met")
    print("   2. Deploy with canary release (Phase 8)")
    print("   3. Monitor performance (Phase 9)")
    print("   4. Set up drift detection (Phase 10)")
    
    print("\n" + "=" * 100)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MLOps Pipeline")
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "serve"],
        default="train",
        help="Pipeline mode"
    )
    
    args = parser.parse_args()
    
    if args.mode == "train":
        main_train()
    elif args.mode == "evaluate":
        print("Evaluation mode - coming soon!")
    elif args.mode == "serve":
        print("Serving mode - coming soon (Phase 8)!")


if __name__ == "__main__":
    main()
