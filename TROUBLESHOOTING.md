# 🔧 Troubleshooting Guide

Common issues and solutions encountered while building and running this MLOps pipeline.

---

## 📋 Table of Contents

1. [Installation Issues](#installation-issues)
2. [Training Errors](#training-errors)
3. [API Server Issues](#api-server-issues)
4. [Monitoring & Drift Detection](#monitoring--drift-detection)
5. [Database Issues](#database-issues)
6. [Import Errors](#import-errors)
7. [Model Loading Issues](#model-loading-issues)
8. [Performance Issues](#performance-issues)

---

## Installation Issues

### ❌ Problem: `ModuleNotFoundError: No module named 'sklearn'`

**Cause**: scikit-learn not installed or wrong package name

**Solution**:
```bash
pip install scikit-learn pandas numpy scipy joblib
```

**Note**: The package is `scikit-learn` but imported as `sklearn`

---

### ❌ Problem: `ModuleNotFoundError: No module named 'rich'`

**Cause**: Rich library not installed (needed for beautiful output)

**Solution**:
```bash
pip install rich
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

---

### ❌ Problem: Virtual environment not activating

**Cause**: Wrong activation command

**Solution**:
```bash
# Correct command
source venv/bin/activate

# NOT this (that's for conda)
source activate venv
```

**Verify activation**:
```bash
which python
# Should show: /path/to/venv/bin/python
```

---

## Training Errors

### ❌ Problem: `TypeError: __init__() got an unexpected keyword argument 'normalize'`

**Cause**: Using deprecated `normalize` parameter in LinearRegression

**Solution**: Remove from `config.py`:
```python
# WRONG
MODEL_CONFIG = {
    "hyperparams": {
        "fit_intercept": True,
        "normalize": True  # ❌ Deprecated
    }
}

# CORRECT
MODEL_CONFIG = {
    "hyperparams": {
        "fit_intercept": True
    }
}
```

---

### ❌ Problem: `NameError: name 'Tuple' is not defined`

**Cause**: Missing import in typing

**Solution**: Add to imports:
```python
from typing import Dict, Any, List, Tuple  # Add Tuple
```

**Files affected**:
- `6_evaluation/offline_eval.py`
- `6_evaluation/online_eval.py`

---

### ❌ Problem: `PicklingError` when saving feature engineer

**Cause**: Dynamic module loading conflicts with pickle

**Solution**: Save state dictionary instead of object:
```python
# In feature_engineering.py
def save(self, filepath):
    state = {
        'numeric_features': self.numeric_features,
        'categorical_features': self.categorical_features,
        'scaling_method': self.scaling_method,
        'feature_stats': self.feature_stats,
        'is_fitted': self.is_fitted
    }
    joblib.dump(state, filepath)
```

---

### ❌ Problem: `KeyError: 'model_dir'` or `KeyError: 'feature_dir'`

**Cause**: Wrong path names in config

**Solution**: Use correct path names from `config.py`:
```python
# WRONG
model_path = PATHS["model_dir"]
fe_path = PATHS["feature_dir"]

# CORRECT
model_path = PATHS["artifacts_dir"]
fe_path = PATHS["transformers_dir"]
```

**Check `config.py` for available paths**:
```python
PATHS = {
    "project_root": BASE_DIR,
    "data_dir": ...,
    "artifacts_dir": ...,      # ✅ Use this for models
    "transformers_dir": ...,   # ✅ Use this for feature engineer
    "metrics_dir": ...,
}
```

---

## API Server Issues

### ❌ Problem: `[Errno 48] error while attempting to bind on address: address already in use`

**Cause**: Port 8000 already in use

**Solution**:
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Then restart server
python 8_deployment/serve.py
```

**Alternative**: Change port in serve.py:
```python
uvicorn.run(app, host="0.0.0.0", port=8001)  # Use different port
```

---

### ❌ Problem: `AttributeError: 'dict' object has no attribute 'transform'`

**Cause**: Feature engineer loaded as dict instead of FeatureEngineer object

**Solution**: Use custom load method:
```python
# WRONG
feature_engineer = joblib.load(fe_path)

# CORRECT
import importlib.util
spec = importlib.util.spec_from_file_location(
    "feature_engineering",
    Path(__file__).parent.parent / "3_features" / "feature_engineering.py"
)
fe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fe_module)
feature_engineer = fe_module.FeatureEngineer.load(fe_path)
```

---

### ❌ Problem: `AttributeError: 'list' object has no attribute 'columns'`

**Cause**: Feature engineer expects DataFrame, not list

**Solution**:
```python
# WRONG
X = feature_engineer.transform([house_dict])

# CORRECT
house_df = pd.DataFrame([house_dict])
X = feature_engineer.transform(house_df)
```

---

### ❌ Problem: API returns `Internal Server Error` (500)

**Cause**: Check server logs for actual error

**Solution**:
```bash
# Run server in foreground to see errors
python 8_deployment/serve.py

# Or check logs
tail -f /tmp/serve.log
```

**Common causes**:
- Model not loaded correctly
- Wrong data format in request
- Missing features in input
- Numpy type serialization issues

---

## Monitoring & Drift Detection

### ❌ Problem: `KeyError: 'problem_statement'` in dashboard

**Cause**: API returns nested JSON structure

**Solution**: Use `.get()` for safe access:
```python
# WRONG
problem = data['problem_statement']

# CORRECT
problem = data.get('problem_statement', {})
if isinstance(problem, dict):
    st.info(problem.get('problem_statement', 'N/A'))
```

---

### ❌ Problem: Drift detection shows "Not enough data"

**Cause**: Need at least 50 predictions for statistical significance

**Solution**:
```bash
# Generate test predictions
python generate_predictions.py 60

# Then check drift
python 9_monitoring/monitor.py
```

---

### ❌ Problem: `ValueError: numpy types not JSON serializable`

**Cause**: Numpy types (np.int64, np.float64, np.bool_) can't be serialized

**Solution**: Convert to Python types:
```python
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
```

---

### ❌ Problem: Metrics showing $0 after training

**Cause**: API not reloading model after training

**Solution**: Add model reload endpoint and call it:
```python
# In API
@app.post("/api/v1/model/reload")
async def reload_model():
    global model, feature_engineer
    load_model_artifacts()
    return {"status": "success"}

# After training
requests.post("http://localhost:8000/api/v1/model/reload")
```

---

## Database Issues

### ❌ Problem: `database is locked`

**Cause**: Multiple processes accessing SQLite simultaneously

**Solution**:
```bash
# Close all connections
pkill -f serve.py

# Delete and recreate database
rm production.db

# Restart server (will recreate DB)
python 8_deployment/serve.py
```

---

### ❌ Problem: `no such table: predictions`

**Cause**: Database not initialized

**Solution**: Database auto-creates on server startup. Just restart:
```bash
python 8_deployment/serve.py
```

---

### ❌ Problem: Can't query database

**Cause**: Database file doesn't exist

**Solution**:
```bash
# Check if database exists
ls -la production.db

# If missing, start server to create it
python 8_deployment/serve.py

# Make at least one prediction to populate
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"square_feet": 2000, "bedrooms": 3, "bathrooms": 2, "age_years": 10, "neighborhood_quality": 7, "has_garage": 1}'
```

---

## Import Errors

### ❌ Problem: `ModuleNotFoundError` when running scripts from subdirectories

**Cause**: Python can't find parent modules

**Solution**: Add parent to path:
```python
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Now can import from root
from config import PATHS
```

---

### ❌ Problem: `ImportError: cannot import name 'FeatureEngineer'`

**Cause**: Circular imports or wrong import path

**Solution**: Use dynamic import:
```python
import importlib.util

spec = importlib.util.spec_from_file_location(
    "feature_engineering",
    Path(__file__).parent / "3_features" / "feature_engineering.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

FeatureEngineer = module.FeatureEngineer
```

---

## Model Loading Issues

### ❌ Problem: `FileNotFoundError: model_v1.pkl not found`

**Cause**: Model not trained yet

**Solution**:
```bash
# Train model first
python train_beautiful.py

# Verify model exists
ls -la 4_model/artifacts/model_v1.pkl
```

---

### ❌ Problem: Model predictions are all the same

**Cause**: Model not fitted or loaded incorrectly

**Solution**:
```python
# Check if model is fitted
print(hasattr(model, 'coef_'))  # Should be True

# Check coefficients
print(model.coef_)  # Should not be all zeros

# Retrain if needed
python train_beautiful.py
```

---

### ❌ Problem: `TypeError: FeatureEngineer.__init__() got unexpected keyword arguments`

**Cause**: Trying to pass parameters to __init__ that doesn't accept them

**Solution**: Don't pass parameters when loading:
```python
# WRONG
fe = FeatureEngineer(numeric_features=..., categorical_features=...)

# CORRECT
fe = FeatureEngineer()
fe.numeric_features = state['numeric_features']
fe.categorical_features = state['categorical_features']
# ... restore other attributes
```

---

## Performance Issues

### ❌ Problem: Training is very slow

**Cause**: Too many samples or inefficient code

**Solution**: Reduce samples in `config.py`:
```python
DATA_CONFIG = {
    "n_samples": 1000,  # Reduce from 5000 for faster training
}
```

---

### ❌ Problem: API latency > 500ms

**Cause**: Model or feature engineering too slow

**Solution**:
1. Check model complexity
2. Profile code:
```python
import time
start = time.time()
# Your code here
print(f"Took {(time.time() - start) * 1000:.2f}ms")
```

3. Optimize feature engineering
4. Use simpler model for testing

---

### ❌ Problem: Drift detection takes too long

**Cause**: Too many predictions to analyze

**Solution**: Limit query in monitor.py:
```python
# Limit to recent predictions
df = pd.read_sql_query("""
    SELECT * FROM predictions 
    WHERE timestamp >= datetime('now', '-1 hour')  -- Last hour only
    ORDER BY timestamp DESC
    LIMIT 500  -- Max 500 predictions
""", conn)
```

---

## Path Issues

### ❌ Problem: `FileNotFoundError` for config files

**Cause**: Running script from wrong directory

**Solution**: Always run from project root:
```bash
# WRONG
cd 8_deployment
python serve.py

# CORRECT
cd /path/to/linear-regression-prod-template
python 8_deployment/serve.py
```

---

### ❌ Problem: Relative paths not working

**Cause**: Using relative paths instead of absolute

**Solution**: Use Path(__file__).parent:
```python
# WRONG
DB_PATH = "production.db"

# CORRECT
DB_PATH = Path(__file__).parent.parent / "production.db"
```

---

## Terminal Output Issues

### ❌ Problem: Rich output not showing colors

**Cause**: Terminal doesn't support colors or rich not installed

**Solution**:
```bash
# Install rich
pip install rich

# Use terminal that supports colors (iTerm2, modern terminals)
# Or force color output
export FORCE_COLOR=1
```

---

### ❌ Problem: Tables are truncated or wrapped weirdly

**Cause**: Terminal window too narrow

**Solution**:
1. Widen terminal window
2. Or reduce table width in code:
```python
table.add_column("Feature", style="cyan", width=15)  # Set max width
```

---

## Retraining Issues

### ❌ Problem: `python retrain.py check` fails with `UnboundLocalError`

**Cause**: `sys` imported inside function after being used

**Solution**: Import at top of file:
```python
import sys  # At top of file
from pathlib import Path

# Not inside function
```

---

### ❌ Problem: Retraining doesn't update model

**Cause**: Model backup not created or paths wrong

**Solution**:
```bash
# Create backup manually
cp 4_model/artifacts/model_v1.pkl 4_model/artifacts/model_v1_backup.pkl

# Then retrain
python 12_learning/retrain.py force
```

---

## General Debugging Tips

### 🔍 Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 🔍 Check Python Version

```bash
python --version
# Should be Python 3.8+
```

### 🔍 Verify All Imports

```bash
python test_pipeline.py
```

### 🔍 Check File Permissions

```bash
chmod -R u+w .
```

### 🔍 Clear Python Cache

```bash
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
```

---

## 🆘 Still Having Issues?

1. **Check logs**: Look at terminal output carefully
2. **Read error messages**: They usually tell you exactly what's wrong
3. **Verify setup**: Run `python test_pipeline.py`
4. **Check versions**: Ensure compatible package versions
5. **Start fresh**: Delete artifacts and retrain

### Clean Start Commands

```bash
# Remove all generated files
rm -rf 4_model/artifacts/*
rm -rf 3_features/transformers/*
rm -rf 6_evaluation/metrics/*
rm -rf 2_data_management/data/*
rm production.db

# Retrain everything
python train_beautiful.py

# Restart server
python 8_deployment/serve.py
```

---

## 📚 Additional Resources

- **README.md** - Complete documentation
- **QUICKSTART.md** - Quick setup guide
- **config.py** - All configuration options
- **API Docs** - http://localhost:8000/docs (when server running)

---

**Most issues can be solved by reading error messages carefully and checking file paths!** 🎯
