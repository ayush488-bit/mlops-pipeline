"""
Continuous Learning / Retraining Script
Automatically retrain model when needed
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json
import shutil

RETRAIN_LOG = Path(__file__).parent.parent / "retrain_log.json"

def should_retrain():
    """Check if retraining is needed."""
    print("\n" + "="*80)
    print("🤔 CHECKING IF RETRAINING IS NEEDED")
    print("="*80 + "\n")
    
    reasons = []
    
    # Check 1: Drift detection
    print("1️⃣  Checking for drift...")
    result = subprocess.run(
        [sys.executable, "9_monitoring/monitor.py"],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True
    )
    
    if "DRIFT DETECTED" in result.stdout:
        reasons.append("Data drift detected")
        print("   ⚠️  Drift detected")
    else:
        print("   ✅ No drift")
    
    # Check 2: Time since last training
    print("\n2️⃣  Checking training age...")
    sys.path.append(str(Path(__file__).parent.parent))
    from config import PATHS
    
    model_path = PATHS["artifacts_dir"] / "model_v1.pkl"
    
    if model_path.exists():
        import time
        age_days = (time.time() - model_path.stat().st_mtime) / 86400
        print(f"   Model age: {age_days:.1f} days")
        
        if age_days > 7:
            reasons.append(f"Model is {age_days:.1f} days old (>7 days)")
            print("   ⚠️  Model is stale")
        else:
            print("   ✅ Model is fresh")
    
    # Check 3: Performance degradation
    print("\n3️⃣  Checking performance...")
    db_path = Path(__file__).parent / "production.db"
    
    if db_path.exists():
        import sqlite3
        import pandas as pd
        from datetime import timedelta
        
        conn = sqlite3.connect(db_path)
        cutoff = (datetime.now() - timedelta(days=1)).isoformat()
        
        df = pd.read_sql_query(f"""
            SELECT predicted_price FROM predictions 
            WHERE timestamp >= '{cutoff}'
        """, conn)
        
        conn.close()
        
        if len(df) > 100:
            # Check for anomalies
            import numpy as np
            mean = df['predicted_price'].mean()
            std = df['predicted_price'].std()
            anomalies = df[np.abs(df['predicted_price'] - mean) > 3 * std]
            
            anomaly_rate = len(anomalies) / len(df)
            print(f"   Anomaly rate: {anomaly_rate*100:.2f}%")
            
            if anomaly_rate > 0.05:
                reasons.append(f"High anomaly rate: {anomaly_rate*100:.1f}%")
                print("   ⚠️  Too many anomalies")
            else:
                print("   ✅ Performance normal")
        else:
            print("   ℹ️  Not enough data")
    else:
        print("   ℹ️  No production data")
    
    print()
    
    if reasons:
        print("🚨 RETRAINING RECOMMENDED")
        for reason in reasons:
            print(f"   - {reason}")
        print("\n" + "="*80 + "\n")
        return True, reasons
    else:
        print("✅ NO RETRAINING NEEDED")
        print("="*80 + "\n")
        return False, []

def retrain_model():
    """Retrain the model."""
    print("\n" + "="*80)
    print("🔄 STARTING RETRAINING")
    print("="*80 + "\n")
    
    # Import config
    sys.path.append(str(Path(__file__).parent.parent))
    from config import PATHS
    
    # Backup current model
    model_dir = PATHS["artifacts_dir"]
    current_model = model_dir / "model_v1.pkl"
    backup_model = model_dir / "model_v1_backup.pkl"
    
    if current_model.exists():
        print("📦 Backing up current model...")
        shutil.copy(current_model, backup_model)
        print(f"   Saved to: {backup_model}")
    
    # Run training pipeline
    print("\n🤖 Running training pipeline...")
    print("="*80)
    
    result = subprocess.run(
        [sys.executable, "main.py", "--mode", "train"],
        cwd=Path(__file__).parent.parent,
        capture_output=False,  # Show output
        text=True
    )
    
    if result.returncode != 0:
        print("\n❌ Training failed!")
        print("   Restoring backup...")
        if backup_model.exists():
            shutil.copy(backup_model, current_model)
            print("   ✅ Backup restored")
        return False
    
    print("\n" + "="*80)
    print("✅ RETRAINING COMPLETE")
    print("="*80 + "\n")
    
    # Log retraining
    retrain_entry = {
        "timestamp": datetime.now().isoformat(),
        "success": True,
        "backup_created": backup_model.exists()
    }
    
    if RETRAIN_LOG.exists():
        with open(RETRAIN_LOG, 'r') as f:
            log = json.load(f)
    else:
        log = []
    
    log.append(retrain_entry)
    
    with open(RETRAIN_LOG, 'w') as f:
        json.dump(log, f, indent=2)
    
    print(f"📝 Logged to: {RETRAIN_LOG}")
    print("\n💡 Restart server to load new model:")
    print("   pkill -f serve.py && python serve.py")
    print()
    
    return True

def view_retrain_history():
    """View retraining history."""
    print("\n" + "="*80)
    print("📜 RETRAINING HISTORY")
    print("="*80 + "\n")
    
    if not RETRAIN_LOG.exists():
        print("ℹ️  No retraining history")
        return
    
    with open(RETRAIN_LOG, 'r') as f:
        log = json.load(f)
    
    if len(log) == 0:
        print("ℹ️  No retraining performed")
        return
    
    for i, entry in enumerate(log, 1):
        print(f"Retraining #{i}")
        print(f"   Time: {entry['timestamp']}")
        print(f"   Success: {'✅' if entry['success'] else '❌'}")
        print(f"   Backup: {'✅' if entry.get('backup_created') else '❌'}")
        print()
    
    print("="*80 + "\n")

def schedule_retraining():
    """Set up scheduled retraining (cron job)."""
    print("\n" + "="*80)
    print("⏰ SCHEDULE AUTOMATIC RETRAINING")
    print("="*80 + "\n")
    
    print("To schedule automatic retraining, add this to your crontab:")
    print()
    print("# Check daily at 2 AM")
    print(f"0 2 * * * cd {Path(__file__).parent} && {sys.executable} retrain.py auto")
    print()
    print("# Check weekly on Sunday at 3 AM")
    print(f"0 3 * * 0 cd {Path(__file__).parent} && {sys.executable} retrain.py auto")
    print()
    print("To edit crontab:")
    print("   crontab -e")
    print()
    print("="*80 + "\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "check":
            # Just check, don't retrain
            needed, reasons = should_retrain()
            sys.exit(0 if not needed else 1)
        
        elif sys.argv[1] == "auto":
            # Automatic mode: check and retrain if needed
            needed, reasons = should_retrain()
            if needed:
                retrain_model()
        
        elif sys.argv[1] == "force":
            # Force retraining
            retrain_model()
        
        elif sys.argv[1] == "history":
            # View history
            view_retrain_history()
        
        elif sys.argv[1] == "schedule":
            # Show scheduling instructions
            schedule_retraining()
        
        else:
            print("Usage: python retrain.py [check|auto|force|history|schedule]")
            print()
            print("Commands:")
            print("  check     - Check if retraining is needed")
            print("  auto      - Check and retrain if needed")
            print("  force     - Force retraining")
            print("  history   - View retraining history")
            print("  schedule  - Show cron job setup")
    
    else:
        # Interactive mode
        needed, reasons = should_retrain()
        if needed:
            response = input("\n🚨 Retraining recommended. Proceed? (yes/no): ")
            if response.lower() == 'yes':
                retrain_model()
