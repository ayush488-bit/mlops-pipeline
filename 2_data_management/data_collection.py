"""
Phase 2: Data Collection & Management
Generates synthetic house price data with realistic correlations.
Implements data versioning and lineage tracking.
"""

import numpy as np
import pandas as pd
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_CONFIG, PATHS


class DataCollector:
    """
    Generates synthetic house price data with realistic relationships.
    Implements data versioning and lineage tracking for reproducibility.
    """
    
    def __init__(self, n_samples: int = None, random_seed: int = None):
        """
        Initialize data collector.
        
        Args:
            n_samples: Number of house samples to generate
            random_seed: Random seed for reproducibility
        """
        self.n_samples = n_samples or DATA_CONFIG["n_samples"]
        self.random_seed = random_seed or DATA_CONFIG["random_seed"]
        np.random.seed(self.random_seed)
        
    def generate_synthetic_data(self) -> pd.DataFrame:
        """
        Generate synthetic house price data with realistic correlations.
        
        Features:
        - square_feet: 500 to 5000 sq ft
        - bedrooms: 1 to 6 bedrooms
        - bathrooms: 1 to 4 bathrooms
        - age_years: 0 to 100 years old
        - neighborhood_quality: 1 to 10 rating
        - has_garage: 0 or 1 (binary)
        - price: Target variable (realistic formula)
        
        Returns:
            DataFrame with house features and prices
        """
        print(f"🏗️  Generating {self.n_samples} synthetic house records...")
        
        # Generate base features with realistic distributions
        square_feet = np.random.normal(2000, 600, self.n_samples)
        square_feet = np.clip(square_feet, 500, 5000)
        
        # Bedrooms correlate with square feet
        bedrooms = np.round(1 + (square_feet / 800) + np.random.normal(0, 0.5, self.n_samples))
        bedrooms = np.clip(bedrooms, 1, 6).astype(int)
        
        # Bathrooms correlate with bedrooms
        bathrooms = np.round(bedrooms * 0.75 + np.random.normal(0, 0.3, self.n_samples))
        bathrooms = np.clip(bathrooms, 1, 4).astype(int)
        
        # Age is random but affects price
        age_years = np.random.exponential(20, self.n_samples)
        age_years = np.clip(age_years, 0, 100)
        
        # Neighborhood quality (1-10 scale)
        neighborhood_quality = np.random.choice(range(1, 11), self.n_samples, 
                                               p=[0.05, 0.08, 0.10, 0.12, 0.15, 
                                                  0.15, 0.12, 0.10, 0.08, 0.05])
        
        # Has garage (70% have garage)
        has_garage = np.random.choice([0, 1], self.n_samples, p=[0.3, 0.7])
        
        # Generate price with realistic formula
        # Base price from square footage
        base_price = square_feet * 150
        
        # Adjustments
        bedroom_adjustment = bedrooms * 15000
        bathroom_adjustment = bathrooms * 20000
        age_penalty = age_years * -800  # Older houses worth less
        neighborhood_bonus = neighborhood_quality * 25000
        garage_bonus = has_garage * 30000
        
        # Add some random noise (market fluctuations)
        noise = np.random.normal(0, 30000, self.n_samples)
        
        # Final price
        price = (base_price + bedroom_adjustment + bathroom_adjustment + 
                age_penalty + neighborhood_bonus + garage_bonus + noise)
        
        # Ensure prices are positive and realistic
        price = np.clip(price, 50000, 5000000)
        
        # Create DataFrame
        df = pd.DataFrame({
            'square_feet': square_feet,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'age_years': age_years,
            'neighborhood_quality': neighborhood_quality,
            'has_garage': has_garage,
            'price': price
        })
        
        # Round numeric columns for cleaner data
        df['square_feet'] = df['square_feet'].round(0)
        df['age_years'] = df['age_years'].round(1)
        df['price'] = df['price'].round(0)
        
        print(f"✅ Generated {len(df)} house records")
        print(f"\n📊 Data Summary:")
        print(df.describe())
        
        return df
    
    def compute_checksum(self, df: pd.DataFrame) -> str:
        """
        Compute SHA256 checksum of dataframe for integrity verification.
        
        Args:
            df: DataFrame to compute checksum for
            
        Returns:
            Hexadecimal checksum string
        """
        # Convert dataframe to bytes and compute hash
        df_bytes = pd.util.hash_pandas_object(df).values.tobytes()
        checksum = hashlib.sha256(df_bytes).hexdigest()
        return checksum
    
    def compute_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute metadata about the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with metadata
        """
        metadata = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "missing_rate": (df.isnull().sum() / len(df)).to_dict(),
            "numeric_stats": {
                col: {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "median": float(df[col].median())
                }
                for col in df.select_dtypes(include=[np.number]).columns
            }
        }
        return metadata
    
    def save_data_with_lineage(self, df: pd.DataFrame, version: str, stage: str) -> Tuple[Path, Path]:
        """
        Save data with versioning and lineage tracking.
        
        Args:
            df: DataFrame to save
            version: Version identifier (e.g., "v1", "v2")
            stage: Data stage (e.g., "raw", "cleaned", "train", "test")
            
        Returns:
            Tuple of (data_path, lineage_path)
        """
        data_dir = PATHS["data_dir"]
        
        # Create filenames
        data_filename = f"data_{stage}_{version}.csv"
        lineage_filename = f"lineage_{stage}_{version}.json"
        
        data_path = data_dir / data_filename
        lineage_path = data_dir / lineage_filename
        
        # Save data
        df.to_csv(data_path, index=False)
        print(f"💾 Saved data to: {data_path}")
        
        # Compute checksum and metadata
        checksum = self.compute_checksum(df)
        metadata = self.compute_metadata(df)
        
        # Create lineage record
        lineage = {
            "version": version,
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            "data_path": str(data_path),
            "checksum": checksum,
            "random_seed": self.random_seed,
            "n_samples": self.n_samples,
            "source": "synthetic_generation",
            "metadata": metadata
        }
        
        # Save lineage
        with open(lineage_path, 'w') as f:
            json.dump(lineage, f, indent=2)
        print(f"📝 Saved lineage to: {lineage_path}")
        
        return data_path, lineage_path
    
    def load_data(self, version: str, stage: str) -> pd.DataFrame:
        """
        Load data from a specific version and stage.
        
        Args:
            version: Version identifier
            stage: Data stage
            
        Returns:
            DataFrame
        """
        data_dir = PATHS["data_dir"]
        data_filename = f"data_{stage}_{version}.csv"
        data_path = data_dir / data_filename
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        print(f"📂 Loaded data from: {data_path}")
        return df
    
    def verify_data_integrity(self, df: pd.DataFrame, version: str, stage: str) -> bool:
        """
        Verify data integrity by comparing checksums.
        
        Args:
            df: DataFrame to verify
            version: Version identifier
            stage: Data stage
            
        Returns:
            True if integrity check passes
        """
        data_dir = PATHS["data_dir"]
        lineage_filename = f"lineage_{stage}_{version}.json"
        lineage_path = data_dir / lineage_filename
        
        if not lineage_path.exists():
            print(f"⚠️  Lineage file not found: {lineage_path}")
            return False
        
        # Load lineage
        with open(lineage_path, 'r') as f:
            lineage = json.load(f)
        
        # Compute current checksum
        current_checksum = self.compute_checksum(df)
        stored_checksum = lineage["checksum"]
        
        if current_checksum == stored_checksum:
            print(f"✅ Data integrity verified for {version}/{stage}")
            return True
        else:
            print(f"❌ Data integrity check FAILED for {version}/{stage}")
            print(f"   Expected: {stored_checksum}")
            print(f"   Got: {current_checksum}")
            return False


def main():
    """Main function to demonstrate data collection."""
    print("=" * 80)
    print("PHASE 2: DATA COLLECTION & VERSIONING")
    print("=" * 80)
    
    # Initialize collector
    collector = DataCollector(n_samples=5000, random_seed=42)
    
    # Generate data
    df = collector.generate_synthetic_data()
    
    # Save with versioning
    data_path, lineage_path = collector.save_data_with_lineage(df, version="v1", stage="raw")
    
    # Verify integrity
    collector.verify_data_integrity(df, version="v1", stage="raw")
    
    print("\n" + "=" * 80)
    print("✅ Data collection complete!")
    print("=" * 80)
    
    return df


if __name__ == "__main__":
    main()
