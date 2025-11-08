# 🏠 MLOps Production Pipeline - House Price Prediction

Complete production-grade MLOps system implementing all 12 phases of the machine learning lifecycle with beautiful terminal output.

## 🎯 Overview

**Problem**: Predict house prices using a production-ready ML pipeline  
**Model**: Linear Regression  
**Guardrails**: MAE ≤ $50k, RMSE ≤ $75k, R² ≥ 0.85

---

## 📁 Project Structure

```
linear-regression-prod-template/
├── 1_problem_framing/      # Problem definition & metrics
├── 2_data_management/      # Data collection & versioning
├── 3_features/             # Feature engineering & leakage checks
├── 4_model/                # Baseline & model training
├── 5_validation/           # Data validation & schema
├── 6_evaluation/           # Model evaluation & metrics
├── 7_experiments/          # (Reserved for experiment tracking)
├── 8_deployment/           # Production API server
│   └── serve.py           # FastAPI production server
├── 9_monitoring/           # Performance monitoring & drift detection
│   └── monitor.py         # Beautiful monitoring dashboard
├── 10_drift/               # Drift detection algorithms
├── 11_rollback/            # Health checks & rollback system
│   └── rollback.py        # Rollback automation
├── 12_learning/            # Continuous learning & retraining
│   └── retrain.py         # Auto-retraining system
├── config.py               # Central configuration
├── main.py                 # Core training pipeline
├── train_beautiful.py      # Training with beautiful output ✨
├── generate_predictions.py # Prediction generator for testing
├── run_all_phases.py       # Demo all 12 phases
└── test_pipeline.py        # Verify setup
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Model (Beautiful Output)

```bash
python train_beautiful.py
```

**Output:**
```
🚀 MLOps Training Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%

📊 Model Performance
┌──────────┬─────────┬───────────┬────────┐
│ Metric   │   Value │ Guardrail │ Status │
├──────────┼─────────┼───────────┼────────┤
│ MAE      │ $23,353 │ ≤ $50,000 │   ✅   │
│ RMSE     │ $29,508 │ ≤ $75,000 │   ✅   │
│ R² Score │  0.9500 │    ≥ 0.85 │   ✅   │
└──────────┴─────────┴───────────┴────────┘
```

### 3. Start Production Server

```bash
python 8_deployment/serve.py
```

### 4. Generate Test Predictions

```bash
# In another terminal
python generate_predictions.py 60
```

### 5. Monitor Performance

```bash
python 9_monitoring/monitor.py
```

---

## 🔧 All Commands

### Training
```bash
# Beautiful training output
python train_beautiful.py

# Standard training
python main.py --mode train

# Run all 12 phases demo
python run_all_phases.py

# Test pipeline setup
python test_pipeline.py
```

### Production
```bash
# Start API server
python 8_deployment/serve.py

# Generate test predictions
python generate_predictions.py 60

# Monitor performance (with drift detection)
python 9_monitoring/monitor.py

# Check system health
python 11_rollback/rollback.py check

# Rollback to previous model
python 11_rollback/rollback.py rollback

# Check if retraining needed
python 12_learning/retrain.py check

# Auto-retrain if needed
python 12_learning/retrain.py auto

# Force retraining
python 12_learning/retrain.py force
```

---

## 📊 12 MLOps Phases

### Phase 1: Problem Framing
Define problem, metrics, and guardrails before writing code.

**Files**: `1_problem_framing/problem_definition.py`

### Phase 2: Data Management
Generate/collect data with versioning and lineage tracking.

**Files**: `2_data_management/data_collection.py`, `data_versioning.py`

### Phase 3: Feature Engineering
Transform features with training-serving parity.

**Files**: `3_features/feature_engineering.py`, `leakage_checks.py`

### Phase 4: Model Training
Baseline model + Linear Regression training.

**Files**: `4_model/baseline.py`, `first_model.py`

### Phase 5: Data Validation
Schema validation and quality checks.

**Files**: `5_validation/schema.py`, `validation_gates.py`

### Phase 6: Evaluation
Offline metrics and guardrail validation.

**Files**: `6_evaluation/offline_eval.py`

### Phase 7: Experiments
(Reserved for future experiment tracking)

### Phase 8: Deployment
Production API server with FastAPI.

**Files**: `8_deployment/serve.py`

**Endpoints**:
- `POST /predict` - Make predictions
- `GET /health` - Health check
- `GET /stats` - System statistics
- `GET /docs` - API documentation

### Phase 9: Monitoring
Performance monitoring and drift detection.

**Files**: `9_monitoring/monitor.py`

**Features**:
- Latency metrics (P95, P99)
- Price distribution analysis
- Hourly prediction charts
- **Drift detection** with KS test
- Beautiful table output

### Phase 10: Drift Detection
Statistical drift detection algorithms.

**Files**: `10_drift/drift_detection.py`

**Detects**:
- Covariate shift (input features)
- Concept drift (X→y relationship)
- Label drift (target distribution)

### Phase 11: Rollback
Health checks and automated rollback.

**Files**: `11_rollback/rollback.py`

**Monitors**:
- Latency (P99 < 500ms)
- Prediction anomalies
- Error rates
- Negative predictions

### Phase 12: Continuous Learning
Automated retraining triggers.

**Files**: `12_learning/retrain.py`

**Triggers**:
- Data drift detected
- Model age > 7 days
- Performance degradation
- High anomaly rate

---

## 🎨 Beautiful Output Features

All tools now feature professional terminal output using the `rich` library:

- ✅ **Colored text** (cyan, green, red, yellow)
- ✅ **Beautiful tables** with borders
- ✅ **Progress bars** with spinners
- ✅ **Panels** for important messages
- ✅ **Status indicators** (✅/⚠️/❌)

---

## 📈 Expected Results

After training:

```
MAE:  $23,000 - $45,000
RMSE: $30,000 - $65,000
R²:   0.88 - 0.95
```

Model beats baseline by **~78%**

---

## 🗄️ Data Storage

### Files Created
- `production.db` - SQLite database logging all predictions
- `4_model/artifacts/model_v1.pkl` - Trained model
- `3_features/transformers/feature_engineer.pkl` - Feature transformer
- `6_evaluation/metrics/metrics_v1_test.json` - Model metrics
- `2_data_management/data/` - Training/test data

### Database Schema
```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    square_feet REAL,
    bedrooms INTEGER,
    bathrooms INTEGER,
    age_years REAL,
    neighborhood_quality INTEGER,
    has_garage INTEGER,
    predicted_price REAL,
    latency_ms REAL,
    model_version TEXT
)
```

---

## 🔄 Complete Workflow

```bash
# 1. Train model
python train_beautiful.py

# 2. Start production server (Terminal 1)
python 8_deployment/serve.py

# 3. Generate predictions (Terminal 2)
python generate_predictions.py 60

# 4. Monitor performance
python 9_monitoring/monitor.py

# 5. Check health
python 11_rollback/rollback.py check

# 6. Check if retraining needed
python 12_learning/retrain.py check
```

---

## ⚙️ Configuration

All settings in `config.py`:

```python
# Data
DATA_CONFIG = {
    "n_samples": 5000,
    "random_seed": 42,
    "train_test_split": 0.2
}

# Guardrails
GUARDRAILS = {
    "mae_threshold": 50000,
    "rmse_threshold": 75000,
    "r2_threshold": 0.85,
    "latency_ms": 500
}

# Retraining
RETRAINING_CONFIG = {
    "schedule_days": 7,
    "drift_based_threshold": 0.20,
    "performance_based_threshold": 0.15
}
```

---

## 🛡️ Guardrails

| Metric | Threshold | Action |
|--------|-----------|--------|
| MAE | ≤ $50,000 | Don't deploy |
| RMSE | ≤ $75,000 | Don't deploy |
| R² | ≥ 0.85 | Don't deploy |
| Latency (P99) | < 500ms | Alert |
| Error Rate | < 1% | Rollback |

---

## 🧪 Testing

```bash
# Test all imports and setup
python test_pipeline.py

# Test individual phases
python 2_data_management/data_collection.py
python 3_features/feature_engineering.py
python 4_model/baseline.py
```

---

## 📦 Dependencies

Core requirements:
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning
- `scipy` - Statistical tests
- `joblib` - Model persistence
- `fastapi` - Production API
- `uvicorn` - ASGI server
- `pydantic` - Data validation
- `rich` - Beautiful terminal output ✨

Install all:
```bash
pip install -r requirements.txt
```

---

## 🎯 Key Features

### ✨ Production-Ready
- FastAPI server with automatic docs
- SQLite logging for all predictions
- Health checks and monitoring
- Automated rollback system

### 📊 Monitoring & Observability
- Real-time latency tracking
- Drift detection with statistical tests
- Beautiful terminal dashboards
- Prediction logging and analysis

### 🔄 Continuous Learning
- Automated retraining triggers
- Model versioning and backup
- Performance degradation detection
- Scheduled retraining support

### 🎨 Developer Experience
- Beautiful colored output
- Progress bars and spinners
- Professional table formatting
- Clear status indicators

---

## 🚨 Troubleshooting

### Port Already in Use
```bash
lsof -ti:8000 | xargs kill -9
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### Database Locked
```bash
rm production.db
# Restart server to recreate
```

---

## 📚 Documentation

- **QUICKSTART.md** - Quick setup guide (5 minutes)
- **TROUBLESHOOTING.md** - Common issues and solutions 🔧
- **config.py** - All configuration options
- **API Docs** - http://localhost:8000/docs (when server running)

---

## ✅ Status

**All 12 MLOps phases implemented and working!**

- ✅ Problem Framing
- ✅ Data Management
- ✅ Feature Engineering
- ✅ Model Training
- ✅ Data Validation
- ✅ Model Evaluation
- ✅ Experiments (structure ready)
- ✅ Deployment (FastAPI)
- ✅ Monitoring (with drift detection)
- ✅ Drift Detection
- ✅ Rollback System
- ✅ Continuous Learning

---

**Built with production best practices** | **Beautiful terminal output** | **Complete MLOps lifecycle**
- Demo feature added
