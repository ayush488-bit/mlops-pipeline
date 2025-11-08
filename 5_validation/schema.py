"""
Phase 5: Data Schema
Defines the contract for data structure and validation rules.
"""

from typing import Dict, Any

# Data schema - this is the contract between data producers and consumers
SCHEMA = {
    "square_feet": {
        "type": "float",
        "min": 500,
        "max": 10000,
        "required": True,
        "description": "House size in square feet"
    },
    "bedrooms": {
        "type": "int",
        "min": 1,
        "max": 10,
        "required": True,
        "description": "Number of bedrooms"
    },
    "bathrooms": {
        "type": "int",
        "min": 1,
        "max": 8,
        "required": True,
        "description": "Number of bathrooms"
    },
    "age_years": {
        "type": "float",
        "min": 0,
        "max": 200,
        "required": True,
        "description": "Age of house in years"
    },
    "neighborhood_quality": {
        "type": "int",
        "min": 1,
        "max": 10,
        "required": True,
        "description": "Neighborhood quality rating (1-10)"
    },
    "has_garage": {
        "type": "int",
        "min": 0,
        "max": 1,
        "required": True,
        "description": "Whether house has garage (0=no, 1=yes)"
    },
    "price": {
        "type": "float",
        "min": 10000,
        "max": 10000000,
        "required": True,
        "description": "House price in dollars (target variable)"
    }
}


def get_feature_schema() -> Dict[str, Any]:
    """Get schema for features only (excluding target)."""
    return {k: v for k, v in SCHEMA.items() if k != "price"}


def get_target_schema() -> Dict[str, Any]:
    """Get schema for target variable."""
    return {"price": SCHEMA["price"]}


def display_schema():
    """Display schema in readable format."""
    print("\n" + "=" * 80)
    print("DATA SCHEMA")
    print("=" * 80)
    
    for field, rules in SCHEMA.items():
        print(f"\n📋 {field}")
        print(f"   Type: {rules['type']}")
        print(f"   Range: [{rules['min']}, {rules['max']}]")
        print(f"   Required: {rules['required']}")
        print(f"   Description: {rules['description']}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    display_schema()
