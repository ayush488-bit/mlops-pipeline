"""
Beautiful Training Pipeline Wrapper
Runs main.py with rich progress indicators
"""

import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich import box
import json
import time

console = Console()

def run_training():
    """Run training pipeline with beautiful output."""
    
    console.print()
    console.print(Panel.fit(
        "[bold cyan]🚀 MLOps Training Pipeline[/bold cyan]\n"
        "[dim]Complete 12-phase machine learning system[/dim]",
        border_style="cyan"
    ))
    console.print()
    
    phases = [
        ("📋 Problem Framing", 5),
        ("📊 Data Collection", 10),
        ("🔍 Data Validation", 15),
        ("🛡️ Leakage Checks", 20),
        ("🔧 Feature Engineering", 30),
        ("📏 Baseline Model", 40),
        ("🤖 Model Training", 60),
        ("📊 Model Evaluation", 80),
        ("✅ Guardrail Checks", 90),
        ("💾 Saving Artifacts", 100),
    ]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("[cyan]Training...", total=100)
        
        # Run actual training
        process = subprocess.Popen(
            [sys.executable, "main.py", "--mode", "train"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path(__file__).parent
        )
        
        phase_idx = 0
        output_lines = []
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            
            if line:
                output_lines.append(line.strip())
                
                # Update progress based on output
                if "PHASE 1" in line or "Problem Framing" in line:
                    phase_idx = 0
                elif "PHASE 2" in line or "Data Collection" in line:
                    phase_idx = 1
                elif "PHASE 5" in line or "VALIDATION" in line:
                    phase_idx = 2
                elif "LEAKAGE" in line:
                    phase_idx = 3
                elif "PHASE 3" in line or "FEATURE" in line:
                    phase_idx = 4
                elif "BASELINE" in line:
                    phase_idx = 5
                elif "TRAINING MODEL" in line:
                    phase_idx = 6
                elif "Evaluating" in line:
                    phase_idx = 7
                elif "guardrails" in line.lower():
                    phase_idx = 8
                elif "Saved" in line:
                    phase_idx = 9
                
                if phase_idx < len(phases):
                    desc, target = phases[phase_idx]
                    progress.update(task, description=f"[cyan]{desc}", completed=target)
        
        progress.update(task, completed=100, description="[green]✅ Complete!")
        
        # Wait for process to finish
        process.wait()
        
        # Capture any remaining output
        stderr = process.stderr.read()
    
    console.print()
    
    if process.returncode == 0:
        # Parse and display results
        console.rule("[bold green]✅ Training Complete![/bold green]", style="green")
        console.print()
        
        # Try to read metrics
        metrics_path = Path(__file__).parent / "6_evaluation" / "metrics" / "metrics_v1_test.json"
        
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                data = json.load(f)
                metrics = data.get('metrics', {})
            
            # Create results table
            results_table = Table(title="📊 Model Performance", box=box.ROUNDED, show_header=True, header_style="bold magenta")
            results_table.add_column("Metric", style="cyan", width=20)
            results_table.add_column("Value", justify="right", style="green")
            results_table.add_column("Guardrail", justify="right", style="yellow")
            results_table.add_column("Status", justify="center")
            
            mae = metrics.get('mae', 0)
            rmse = metrics.get('rmse', 0)
            r2 = metrics.get('r2', 0)
            
            mae_pass = mae <= 50000
            rmse_pass = rmse <= 75000
            r2_pass = r2 >= 0.85
            
            results_table.add_row(
                "MAE", 
                f"${mae:,.0f}", 
                "≤ $50,000",
                "✅" if mae_pass else "❌"
            )
            results_table.add_row(
                "RMSE", 
                f"${rmse:,.0f}", 
                "≤ $75,000",
                "✅" if rmse_pass else "❌"
            )
            results_table.add_row(
                "R² Score", 
                f"{r2:.4f}", 
                "≥ 0.85",
                "✅" if r2_pass else "❌"
            )
            
            console.print(results_table)
            console.print()
            
            if mae_pass and rmse_pass and r2_pass:
                console.print(Panel(
                    "[bold green]✅ All guardrails passed! Model is ready for deployment.[/bold green]",
                    title="Success",
                    border_style="green"
                ))
            else:
                console.print(Panel(
                    "[bold red]⚠️  Some guardrails failed. Review model before deployment.[/bold red]",
                    title="Warning",
                    border_style="red"
                ))
        
        console.print()
        console.print("[bold cyan]📁 Artifacts saved:[/bold cyan]")
        console.print("   [green]•[/green] Model: 4_model/artifacts/model_v1.pkl")
        console.print("   [green]•[/green] Features: 3_features/transformers/feature_engineer.pkl")
        console.print("   [green]•[/green] Metrics: 6_evaluation/metrics/metrics_v1_test.json")
        console.print()
        
        console.print("[bold cyan]🚀 Next steps:[/bold cyan]")
        console.print("   [yellow]1.[/yellow] python 8_deployment/serve.py")
        console.print("   [yellow]2.[/yellow] python generate_predictions.py 60")
        console.print("   [yellow]3.[/yellow] python 9_monitoring/monitor.py")
        console.print()
        
    else:
        console.print(Panel(
            f"[bold red]❌ Training failed with exit code {process.returncode}[/bold red]",
            title="Error",
            border_style="red"
        ))
        if stderr:
            console.print("\n[red]Error output:[/red]")
            console.print(stderr)

if __name__ == "__main__":
    try:
        run_training()
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error: {str(e)}[/bold red]")
