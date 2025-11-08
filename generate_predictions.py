"""
Generate sample predictions for testing monitoring and drift detection
"""

import requests
import random
import time
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

API_URL = "http://localhost:8000/predict"

def generate_predictions(n=60):
    """Generate n predictions with varied house features."""
    
    console.print("\n[bold cyan]🏠 Generating Sample Predictions[/bold cyan]\n")
    
    predictions = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        
        task = progress.add_task(f"[cyan]Making {n} predictions...", total=n)
        
        for i in range(n):
            # Generate varied house features
            house = {
                "square_feet": random.randint(800, 4500),
                "bedrooms": random.randint(1, 6),
                "bathrooms": random.randint(1, 4),
                "age_years": random.randint(0, 50),
                "neighborhood_quality": random.randint(1, 10),
                "has_garage": random.choice([0, 1])
            }
            
            try:
                response = requests.post(API_URL, json=house, timeout=5)
                if response.status_code == 200:
                    result = response.json()
                    predictions.append({
                        "id": i + 1,
                        "price": result["predicted_price"],
                        "latency": result["latency_ms"]
                    })
                else:
                    console.print(f"[red]✗[/red] Prediction {i+1} failed: {response.status_code}")
            except requests.exceptions.ConnectionError:
                console.print("\n[bold red]❌ Error: API server not running![/bold red]")
                console.print("[yellow]Start server with:[/yellow] python 8_deployment/serve.py\n")
                return
            except Exception as e:
                console.print(f"[red]✗[/red] Error: {str(e)}")
            
            progress.update(task, advance=1)
            time.sleep(0.05)  # Small delay to avoid overwhelming server
    
    # Summary
    console.print(f"\n[bold green]✅ Generated {len(predictions)} predictions![/bold green]\n")
    
    if predictions:
        avg_price = sum(p["price"] for p in predictions) / len(predictions)
        avg_latency = sum(p["latency"] for p in predictions) / len(predictions)
        
        console.print(f"[cyan]📊 Summary:[/cyan]")
        console.print(f"   Average Price: [green]${avg_price:,.0f}[/green]")
        console.print(f"   Average Latency: [green]{avg_latency:.2f}ms[/green]")
        console.print(f"   Min Price: [yellow]${min(p['price'] for p in predictions):,.0f}[/yellow]")
        console.print(f"   Max Price: [yellow]${max(p['price'] for p in predictions):,.0f}[/yellow]")
        
        console.print(f"\n[bold cyan]💡 Now run:[/bold cyan]")
        console.print(f"   [green]python 9_monitoring/monitor.py[/green]")
        console.print(f"   [green]python 11_rollback/rollback.py check[/green]\n")

if __name__ == "__main__":
    import sys
    
    n = 60
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except:
            pass
    
    generate_predictions(n)
