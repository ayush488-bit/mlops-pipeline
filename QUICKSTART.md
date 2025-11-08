# ⚡ Quick Start

## 1. Install (30 seconds)
```bash
pip install -r requirements.txt
```

## 2. Train Model (1 minute)
```bash
python train_beautiful.py
```

**Expected Output:**
```
🚀 MLOps Training Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%

📊 Model Performance
┌──────────┬─────────┬───────────┬────────┐
│ MAE      │ $23,353 │ ≤ $50,000 │   ✅   │
│ RMSE     │ $29,508 │ ≤ $75,000 │   ✅   │
│ R² Score │  0.9500 │    ≥ 0.85 │   ✅   │
└──────────┴─────────┴───────────┴────────┘

✅ All guardrails passed!
```

## 3. Start Production Server
```bash
python 8_deployment/serve.py
```

## 4. Generate Predictions (in another terminal)
```bash
python generate_predictions.py 60
```

## 5. Monitor Performance
```bash
python 9_monitoring/monitor.py
```

**See beautiful tables with:**
- ⏱️  Latency metrics
- 💰 Price predictions
- 🏠 Input features
- 🔍 **Drift detection**

---

## 🎯 What You Get

- ✅ Trained model with 95% R² score
- ✅ Production API on http://localhost:8000
- ✅ 60+ test predictions logged
- ✅ Drift detection working
- ✅ Beautiful terminal output

---

## 🚀 Next Steps

```bash
# Check system health
python 11_rollback/rollback.py check

# Check if retraining needed
python 12_learning/retrain.py check

# View API docs
open http://localhost:8000/docs
```

---

**Total time: 5 minutes** | **All 12 MLOps phases working!**
