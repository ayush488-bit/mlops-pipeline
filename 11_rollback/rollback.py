"""
Rollback System
Detect issues and rollback to previous model version
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import shutil

DB_PATH = Path(__file__).parent.parent / "production.db"
ROLLBACK_LOG = Path(__file__).parent.parent / "rollback_log.json"

def check_health():
    """Check system health."""
    print("\n" + "="*80)
    print("🏥 HEALTH CHECK")
    print("="*80 + "\n")
    
    if not DB_PATH.exists():
        print("❌ No production database found")
        return False
    
    conn = sqlite3.connect(DB_PATH)
    
    # Get recent predictions (last hour)
    cutoff = (datetime.now() - timedelta(hours=1)).isoformat()
    
    df = pd.read_sql_query(f"""
        SELECT * FROM predictions 
        WHERE timestamp >= '{cutoff}'
    """, conn)
    
    conn.close()
    
    if len(df) == 0:
        print("ℹ️  No predictions in last hour")
        return True
    
    issues = []
    
    # Check 1: Latency
    p99_latency = df['latency_ms'].quantile(0.99)
    print(f"⏱️  P99 Latency: {p99_latency:.2f}ms")
    if p99_latency > 500:
        issues.append(f"High latency: {p99_latency:.2f}ms > 500ms")
        print("   ⚠️  WARNING: Latency too high")
    else:
        print("   ✅ Latency OK")
    
    # Check 2: Prediction distribution
    avg_price = df['predicted_price'].mean()
    std_price = df['predicted_price'].std()
    print(f"\n💰 Predictions: ${avg_price:,.0f} ± ${std_price:,.0f}")
    
    # Check for anomalies (predictions > 3 std from mean)
    anomalies = df[np.abs(df['predicted_price'] - avg_price) > 3 * std_price]
    if len(anomalies) > len(df) * 0.05:  # More than 5% anomalies
        issues.append(f"Too many anomalies: {len(anomalies)}/{len(df)}")
        print(f"   ⚠️  WARNING: {len(anomalies)} anomalous predictions")
    else:
        print("   ✅ Predictions normal")
    
    # Check 3: Negative predictions
    negative = df[df['predicted_price'] < 0]
    if len(negative) > 0:
        issues.append(f"Negative predictions: {len(negative)}")
        print(f"\n   🚨 CRITICAL: {len(negative)} negative predictions!")
    
    # Check 4: Extreme predictions
    extreme = df[(df['predicted_price'] < 50000) | (df['predicted_price'] > 2000000)]
    if len(extreme) > len(df) * 0.1:  # More than 10% extreme
        issues.append(f"Extreme predictions: {len(extreme)}/{len(df)}")
        print(f"   ⚠️  WARNING: {len(extreme)} extreme predictions")
    
    print()
    
    if issues:
        print("🚨 HEALTH CHECK FAILED")
        for issue in issues:
            print(f"   - {issue}")
        print("\n💡 Recommendation: ROLLBACK to previous model")
        print("="*80 + "\n")
        return False
    else:
        print("✅ HEALTH CHECK PASSED")
        print("="*80 + "\n")
        return True

def rollback_model():
    """Rollback to previous model version."""
    print("\n" + "="*80)
    print("🔄 INITIATING ROLLBACK")
    print("="*80 + "\n")
    
    # Import config to get correct paths
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config import PATHS
    
    # Check if backup exists
    model_dir = PATHS["artifacts_dir"]
    current_model = model_dir / "model_v1.pkl"
    backup_model = model_dir / "model_v1_backup.pkl"
    
    if not backup_model.exists():
        print("❌ No backup model found")
        print("💡 Create backup first: cp model_v1.pkl model_v1_backup.pkl")
        return False
    
    # Create rollback log
    rollback_entry = {
        "timestamp": datetime.now().isoformat(),
        "reason": "Health check failed",
        "rolled_back_from": "v1",
        "rolled_back_to": "v1_backup"
    }
    
    # Log rollback
    if ROLLBACK_LOG.exists():
        with open(ROLLBACK_LOG, 'r') as f:
            log = json.load(f)
    else:
        log = []
    
    log.append(rollback_entry)
    
    with open(ROLLBACK_LOG, 'w') as f:
        json.dump(log, f, indent=2)
    
    # Perform rollback
    print("📦 Backing up current model...")
    shutil.copy(current_model, model_dir / "model_v1_failed.pkl")
    
    print("🔄 Restoring backup model...")
    shutil.copy(backup_model, current_model)
    
    print("✅ Rollback complete!")
    print(f"📝 Logged to: {ROLLBACK_LOG}")
    print("\n💡 Restart server to load restored model:")
    print("   pkill -f serve.py && python serve.py")
    
    print("\n" + "="*80 + "\n")
    return True

def view_rollback_history():
    """View rollback history."""
    print("\n" + "="*80)
    print("📜 ROLLBACK HISTORY")
    print("="*80 + "\n")
    
    if not ROLLBACK_LOG.exists():
        print("ℹ️  No rollback history")
        return
    
    with open(ROLLBACK_LOG, 'r') as f:
        log = json.load(f)
    
    if len(log) == 0:
        print("ℹ️  No rollbacks performed")
        return
    
    for i, entry in enumerate(log, 1):
        print(f"Rollback #{i}")
        print(f"   Time: {entry['timestamp']}")
        print(f"   Reason: {entry['reason']}")
        print(f"   From: {entry['rolled_back_from']}")
        print(f"   To: {entry['rolled_back_to']}")
        print()
    
    print("="*80 + "\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "check":
            healthy = check_health()
            sys.exit(0 if healthy else 1)
        elif sys.argv[1] == "rollback":
            rollback_model()
        elif sys.argv[1] == "history":
            view_rollback_history()
        else:
            print("Usage: python rollback.py [check|rollback|history]")
    else:
        # Default: check health and rollback if needed
        healthy = check_health()
        if not healthy:
            response = input("\n🚨 Health check failed. Rollback? (yes/no): ")
            if response.lower() == 'yes':
                rollback_model()
