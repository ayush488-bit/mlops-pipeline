"""
Production Serving Script
Simple FastAPI server for model predictions with monitoring
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json
import sqlite3

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config import PATHS

# Initialize FastAPI
app = FastAPI(
    title="House Price Prediction API",
    description="Production ML model serving",
    version="1.0.0"
)

# Global model and feature engineer
model = None
feature_engineer = None
model_version = "v1"

# Database for logging
DB_PATH = Path(__file__).parent.parent / "production.db"

def init_db():
    """Initialize SQLite database for logging."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    """)
    
    conn.commit()
    conn.close()

def load_model():
    """Load model and feature engineer."""
    global model, feature_engineer, model_version
    
    model_path = PATHS["artifacts_dir"] / f"model_{model_version}.pkl"
    fe_path = PATHS["transformers_dir"] / "feature_engineer.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not fe_path.exists():
        raise FileNotFoundError(f"Feature engineer not found: {fe_path}")
    
    # Load model
    model = joblib.load(model_path)
    
    # Load feature engineer using custom load method
    # Import FeatureEngineer class
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "feature_engineering",
        Path(__file__).parent.parent / "3_features" / "feature_engineering.py"
    )
    fe_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fe_module)
    
    feature_engineer = fe_module.FeatureEngineer.load(fe_path)
    
    print(f"✅ Loaded model {model_version}")
    print(f"✅ Loaded feature engineer")

# Request/Response models
class HouseFeatures(BaseModel):
    square_feet: float
    bedrooms: int
    bathrooms: int
    age_years: float
    neighborhood_quality: int
    has_garage: int

class PredictionResponse(BaseModel):
    predicted_price: float
    model_version: str
    latency_ms: float
    timestamp: str

# Startup
@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    init_db()
    load_model()
    print("\n🚀 Production server ready!")
    print(f"📊 Database: {DB_PATH}")

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_version": model_version,
        "timestamp": datetime.now().isoformat()
    }

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(features: HouseFeatures):
    """Make a prediction."""
    start_time = datetime.now()
    
    # Convert to DataFrame (feature_engineer expects DataFrame)
    house_df = pd.DataFrame([features.dict()])
    
    # Transform features
    X = feature_engineer.transform(house_df)
    
    # Predict
    prediction = float(model.predict(X)[0])
    
    # Calculate latency
    latency_ms = (datetime.now() - start_time).total_seconds() * 1000
    
    # Log to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            timestamp, square_feet, bedrooms, bathrooms, age_years,
            neighborhood_quality, has_garage, predicted_price, latency_ms, model_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        features.square_feet, features.bedrooms, features.bathrooms,
        features.age_years, features.neighborhood_quality, features.has_garage,
        prediction, latency_ms, model_version
    ))
    conn.commit()
    conn.close()
    
    return PredictionResponse(
        predicted_price=prediction,
        model_version=model_version,
        latency_ms=round(latency_ms, 2),
        timestamp=datetime.now().isoformat()
    )

# Stats endpoint
@app.get("/stats")
async def get_stats():
    """Get prediction statistics."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            AVG(predicted_price) as avg_price,
            AVG(latency_ms) as avg_latency,
            MAX(latency_ms) as max_latency
        FROM predictions
    """)
    
    row = cursor.fetchone()
    conn.close()
    
    return {
        "total_predictions": row[0],
        "avg_predicted_price": round(row[1], 2) if row[1] else 0,
        "avg_latency_ms": round(row[2], 2) if row[2] else 0,
        "max_latency_ms": round(row[3], 2) if row[3] else 0
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*80)
    print("🚀 STARTING PRODUCTION SERVER")
    print("="*80)
    print("\n📍 Endpoints:")
    print("   Health:  http://localhost:8000/health")
    print("   Predict: http://localhost:8000/predict")
    print("   Stats:   http://localhost:8000/stats")
    print("   Docs:    http://localhost:8000/docs")
    print("\n" + "="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
