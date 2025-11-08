"""
Production Monitoring Script with Beautiful Output
Analyze predictions and detect issues
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.progress import track

console = Console()
DB_PATH = Path(__file__).parent.parent / "production.db"

def get_predictions(hours=24):
    """Get recent predictions."""
    if not DB_PATH.exists():
        console.print("[bold red]❌ No production database found. Start server first.[/bold red]")
        return None
    
    conn = sqlite3.connect(DB_PATH)
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    
    df = pd.read_sql_query(f"""
        SELECT * FROM predictions 
        WHERE timestamp >= '{cutoff}'
        ORDER BY timestamp DESC
    """, conn)
    
    conn.close()
    return df

def analyze_performance():
    """Analyze model performance."""
    console.print()
    console.rule("[bold cyan]📊 PRODUCTION MONITORING REPORT[/bold cyan]", style="cyan")
    console.print()
    
    df = get_predictions(hours=24)
    
    if df is None or len(df) == 0:
        console.print("[yellow]ℹ️  No predictions in last 24 hours[/yellow]")
        return
    
    # Overview
    console.print(f"[bold]📈 Predictions (Last 24h):[/bold] [cyan]{len(df)}[/cyan]")
    console.print(f"[bold]📅 Time Range:[/bold] {df['timestamp'].min()} → {df['timestamp'].max()}")
    console.print()
    
    # Latency Metrics Table
    latency_table = Table(title="⏱️  Latency Metrics", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    latency_table.add_column("Metric", style="cyan", width=15)
    latency_table.add_column("Value", justify="right", style="green")
    latency_table.add_column("Status", justify="center")
    
    avg_latency = df['latency_ms'].mean()
    median_latency = df['latency_ms'].median()
    p95_latency = df['latency_ms'].quantile(0.95)
    p99_latency = df['latency_ms'].quantile(0.99)
    max_latency = df['latency_ms'].max()
    
    latency_table.add_row("Average", f"{avg_latency:.2f}ms", "")
    latency_table.add_row("Median", f"{median_latency:.2f}ms", "")
    latency_table.add_row("P95", f"{p95_latency:.2f}ms", "")
    latency_table.add_row("P99", f"{p99_latency:.2f}ms", "✅" if p99_latency < 500 else "⚠️")
    latency_table.add_row("Max", f"{max_latency:.2f}ms", "")
    
    console.print(latency_table)
    console.print()
    
    # Price Predictions Table
    price_table = Table(title="💰 Price Predictions", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    price_table.add_column("Metric", style="cyan", width=15)
    price_table.add_column("Value", justify="right", style="green")
    
    price_table.add_row("Average", f"${df['predicted_price'].mean():,.0f}")
    price_table.add_row("Median", f"${df['predicted_price'].median():,.0f}")
    price_table.add_row("Min", f"${df['predicted_price'].min():,.0f}")
    price_table.add_row("Max", f"${df['predicted_price'].max():,.0f}")
    price_table.add_row("Std Dev", f"${df['predicted_price'].std():,.0f}")
    
    console.print(price_table)
    console.print()
    
    # Input Features Table
    features_table = Table(title="🏠 Input Features", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    features_table.add_column("Feature", style="cyan")
    features_table.add_column("Average", justify="right", style="green")
    
    features_table.add_row("Square Feet", f"{df['square_feet'].mean():,.0f}")
    features_table.add_row("Bedrooms", f"{df['bedrooms'].mean():.1f}")
    features_table.add_row("Bathrooms", f"{df['bathrooms'].mean():.1f}")
    features_table.add_row("Age (years)", f"{df['age_years'].mean():.1f}")
    features_table.add_row("Quality (1-10)", f"{df['neighborhood_quality'].mean():.1f}")
    features_table.add_row("With Garage", f"{df['has_garage'].mean()*100:.1f}%")
    
    console.print(features_table)
    console.print()
    
    # Predictions by hour
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    hourly = df.groupby('hour').size()
    
    if len(hourly) > 0:
        console.print("[bold]📊 Predictions by Hour[/bold]")
        for hour, count in hourly.items():
            bar_length = int(count / hourly.max() * 40)
            bar = "█" * bar_length
            console.print(f"   [cyan]{hour:02d}:00[/cyan]  [green]{bar}[/green] {count}")
        console.print()

def check_drift():
    """Check for data drift."""
    console.print()
    console.rule("[bold yellow]🔍 DRIFT DETECTION[/bold yellow]", style="yellow")
    console.print()
    
    df = get_predictions(hours=24)
    
    if df is None or len(df) < 50:
        console.print(Panel(
            f"[yellow]Need at least 50 predictions for drift detection.\nCurrent: {len(df) if df is not None else 0}[/yellow]",
            title="ℹ️  Insufficient Data",
            border_style="yellow"
        ))
        return
    
    # Load training data
    train_data_path = Path(__file__).parent.parent / "2_data_management" / "data" / "data_raw_v1.csv"
    
    if not train_data_path.exists():
        console.print("[bold red]❌ Training data not found[/bold red]")
        return
    
    train_df = pd.read_csv(train_data_path)
    
    # Drift detection table
    drift_table = Table(title="Drift Analysis", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    drift_table.add_column("Feature", style="cyan", width=20)
    drift_table.add_column("P-Value", justify="right", style="white")
    drift_table.add_column("Status", justify="center", width=10)
    drift_table.add_column("Details", style="dim")
    
    features = ['square_feet', 'bedrooms', 'bathrooms', 'age_years', 'neighborhood_quality', 'has_garage']
    drift_detected = False
    
    from scipy.stats import ks_2samp
    
    for feature in features:
        statistic, pvalue = ks_2samp(train_df[feature], df[feature])
        
        if pvalue < 0.05:
            status = "[bold red]⚠️  DRIFT[/bold red]"
            details = f"Train: {train_df[feature].mean():.1f} → Prod: {df[feature].mean():.1f}"
            drift_detected = True
            style = "red"
        else:
            status = "[green]✅ OK[/green]"
            details = f"Stable"
            style = "green"
        
        drift_table.add_row(feature, f"{pvalue:.4f}", status, details, style=style)
    
    console.print(drift_table)
    console.print()
    
    if drift_detected:
        console.print(Panel(
            "[bold red]🚨 Drift detected in one or more features!\n\n"
            "Recommended action: Consider retraining the model[/bold red]",
            title="⚠️  Action Required",
            border_style="red"
        ))
    else:
        console.print(Panel(
            "[bold green]✅ All features are stable - no retraining needed[/bold green]",
            title="✓ System Healthy",
            border_style="green"
        ))
    
    console.print()

if __name__ == "__main__":
    try:
        analyze_performance()
        check_drift()
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error: {str(e)}[/bold red]")
