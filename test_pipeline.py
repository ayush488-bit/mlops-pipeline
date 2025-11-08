"""
Test script to verify the MLOps pipeline works end-to-end.
Run this to quickly check if everything is set up correctly.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "=" * 80)
    print("TESTING IMPORTS")
    print("=" * 80)
    
    try:
        import numpy as np
        print("✅ numpy")
    except ImportError as e:
        print(f"❌ numpy: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ pandas")
    except ImportError as e:
        print(f"❌ pandas: {e}")
        return False
    
    try:
        from sklearn.linear_model import LinearRegression
        print("✅ scikit-learn")
    except ImportError as e:
        print(f"❌ scikit-learn: {e}")
        return False
    
    try:
        import joblib
        print("✅ joblib")
    except ImportError as e:
        print(f"❌ joblib: {e}")
        return False
    
    print("\n✅ All core dependencies installed!")
    return True


def test_config():
    """Test that config loads correctly."""
    print("\n" + "=" * 80)
    print("TESTING CONFIGURATION")
    print("=" * 80)
    
    try:
        from config import DATA_CONFIG, PATHS, GUARDRAILS
        print("✅ Config loaded")
        print(f"   Data samples: {DATA_CONFIG['n_samples']}")
        print(f"   Random seed: {DATA_CONFIG['random_seed']}")
        print(f"   MAE threshold: ${GUARDRAILS['mae_threshold']:,}")
        return True
    except Exception as e:
        print(f"❌ Config failed: {e}")
        return False


def test_phase_1():
    """Test Phase 1: Problem Framing."""
    print("\n" + "=" * 80)
    print("TESTING PHASE 1: PROBLEM FRAMING")
    print("=" * 80)
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("problem_definition",
            Path(__file__).parent / "1_problem_framing" / "problem_definition.py")
        problem_def = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(problem_def)
        
        ProblemFrame = problem_def.ProblemFrame
        problem = ProblemFrame.get_problem_statement()
        
        print(f"✅ Problem statement loaded")
        print(f"   Predict: {problem['predict']}")
        return True
    except Exception as e:
        print(f"❌ Phase 1 failed: {e}")
        return False


def test_phase_2():
    """Test Phase 2: Data Collection."""
    print("\n" + "=" * 80)
    print("TESTING PHASE 2: DATA COLLECTION")
    print("=" * 80)
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("data_collection",
            Path(__file__).parent / "2_data_management" / "data_collection.py")
        data_coll = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_coll)
        
        DataCollector = data_coll.DataCollector
        collector = DataCollector(n_samples=100, random_seed=42)
        df = collector.generate_synthetic_data()
        
        print(f"✅ Data generation works")
        print(f"   Generated: {len(df)} samples")
        print(f"   Features: {list(df.columns)}")
        return True
    except Exception as e:
        print(f"❌ Phase 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase_3():
    """Test Phase 3: Feature Engineering."""
    print("\n" + "=" * 80)
    print("TESTING PHASE 3: FEATURE ENGINEERING")
    print("=" * 80)
    
    try:
        import pandas as pd
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("feature_engineering",
            Path(__file__).parent / "3_features" / "feature_engineering.py")
        feat_eng = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(feat_eng)
        
        FeatureEngineer = feat_eng.FeatureEngineer
        
        # Create sample data
        X = pd.DataFrame({
            'square_feet': [1500, 2000, 2500],
            'bedrooms': [2, 3, 4],
            'bathrooms': [1, 2, 2],
            'age_years': [10, 20, 5],
            'neighborhood_quality': [7, 8, 9],
            'has_garage': [1, 1, 0]
        })
        
        fe = FeatureEngineer()
        X_scaled = fe.fit_transform(X)
        
        print(f"✅ Feature engineering works")
        print(f"   Input shape: {X.shape}")
        print(f"   Output shape: {X_scaled.shape}")
        return True
    except Exception as e:
        print(f"❌ Phase 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase_4():
    """Test Phase 4: Model Training."""
    print("\n" + "=" * 80)
    print("TESTING PHASE 4: MODEL TRAINING")
    print("=" * 80)
    
    try:
        import numpy as np
        import pandas as pd
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("first_model",
            Path(__file__).parent / "4_model" / "first_model.py")
        first_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(first_mod)
        
        FirstModelTrainer = first_mod.FirstModelTrainer
        
        # Create sample data
        X_train = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        y_train = np.array([100, 200, 300, 400, 500])
        
        trainer = FirstModelTrainer()
        model = trainer.train(X_train, y_train, version="test")
        
        # Test prediction
        y_pred = trainer.predict(X_train)
        
        print(f"✅ Model training works")
        print(f"   Predictions shape: {y_pred.shape}")
        return True
    except Exception as e:
        print(f"❌ Phase 4 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 100)
    print(" " * 35 + "PIPELINE TEST SUITE")
    print("=" * 100)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Phase 1: Problem Framing", test_phase_1),
        ("Phase 2: Data Collection", test_phase_2),
        ("Phase 3: Feature Engineering", test_phase_3),
        ("Phase 4: Model Training", test_phase_4),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 100)
    print(" " * 40 + "TEST SUMMARY")
    print("=" * 100)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    print("\n" + "=" * 100)
    print(f"Results: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED! Pipeline is ready to use.")
        print("\nNext step: Run the full pipeline with:")
        print("   python main.py --mode train")
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed. Please fix before running the pipeline.")
    
    print("=" * 100)
    
    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
