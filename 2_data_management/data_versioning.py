"""
Phase 2: Data Versioning
Manages artifact storage and versioning for reproducibility.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PATHS


class DataVersioner:
    """
    Manages versioning of data, models, and features.
    Maintains a manifest of all versions for reproducibility.
    """
    
    def __init__(self, manifest_path: Path = None):
        """
        Initialize data versioner.
        
        Args:
            manifest_path: Path to manifest file
        """
        if manifest_path is None:
            manifest_path = PATHS["data_dir"] / "manifest.json"
        
        self.manifest_path = manifest_path
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> Dict[str, Any]:
        """Load manifest from file or create new one."""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        else:
            return {"versions": {}, "latest": None}
    
    def _save_manifest(self):
        """Save manifest to file."""
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
    
    def register_version(self, 
                        version: str,
                        stage: str,
                        data_path: str,
                        lineage_path: str,
                        checksum: str,
                        metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Register a new data version in the manifest.
        
        Args:
            version: Version identifier
            stage: Data stage
            data_path: Path to data file
            lineage_path: Path to lineage file
            checksum: Data checksum
            metadata: Additional metadata
            
        Returns:
            Version record
        """
        version_key = f"{version}_{stage}"
        
        version_record = {
            "version": version,
            "stage": stage,
            "data_path": data_path,
            "lineage_path": lineage_path,
            "checksum": checksum,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.manifest["versions"][version_key] = version_record
        self.manifest["latest"] = version_key
        self._save_manifest()
        
        print(f"📝 Registered version: {version_key}")
        return version_record
    
    def get_version(self, version: str, stage: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific version.
        
        Args:
            version: Version identifier
            stage: Data stage
            
        Returns:
            Version record or None
        """
        version_key = f"{version}_{stage}"
        return self.manifest["versions"].get(version_key)
    
    def get_latest(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest version.
        
        Returns:
            Latest version record or None
        """
        latest_key = self.manifest.get("latest")
        if latest_key:
            return self.manifest["versions"].get(latest_key)
        return None
    
    def list_versions(self, stage: str = None) -> List[Dict[str, Any]]:
        """
        List all versions, optionally filtered by stage.
        
        Args:
            stage: Optional stage filter
            
        Returns:
            List of version records
        """
        versions = list(self.manifest["versions"].values())
        
        if stage:
            versions = [v for v in versions if v["stage"] == stage]
        
        # Sort by creation time
        versions.sort(key=lambda x: x["created_at"], reverse=True)
        
        return versions
    
    def display_versions(self, stage: str = None):
        """
        Display all versions in a readable format.
        
        Args:
            stage: Optional stage filter
        """
        versions = self.list_versions(stage)
        
        if not versions:
            print("No versions found.")
            return
        
        print("\n" + "=" * 80)
        print("DATA VERSIONS")
        print("=" * 80)
        
        for v in versions:
            print(f"\n📦 Version: {v['version']} | Stage: {v['stage']}")
            print(f"   Created: {v['created_at']}")
            print(f"   Data: {v['data_path']}")
            print(f"   Checksum: {v['checksum'][:16]}...")
            if v.get('metadata'):
                print(f"   Rows: {v['metadata'].get('n_rows', 'N/A')}")
                print(f"   Columns: {v['metadata'].get('n_columns', 'N/A')}")
        
        print("\n" + "=" * 80)
    
    def compare_versions(self, version1: str, stage1: str, 
                        version2: str, stage2: str):
        """
        Compare two versions and show differences.
        
        Args:
            version1: First version identifier
            stage1: First stage
            version2: Second version identifier
            stage2: Second stage
        """
        v1 = self.get_version(version1, stage1)
        v2 = self.get_version(version2, stage2)
        
        if not v1 or not v2:
            print("❌ One or both versions not found")
            return
        
        print("\n" + "=" * 80)
        print(f"COMPARING: {version1}/{stage1} vs {version2}/{stage2}")
        print("=" * 80)
        
        print(f"\n📅 Created:")
        print(f"   {version1}/{stage1}: {v1['created_at']}")
        print(f"   {version2}/{stage2}: {v2['created_at']}")
        
        if v1.get('metadata') and v2.get('metadata'):
            m1 = v1['metadata']
            m2 = v2['metadata']
            
            print(f"\n📊 Rows:")
            print(f"   {version1}/{stage1}: {m1.get('n_rows', 'N/A')}")
            print(f"   {version2}/{stage2}: {m2.get('n_rows', 'N/A')}")
            
            print(f"\n📋 Columns:")
            print(f"   {version1}/{stage1}: {m1.get('n_columns', 'N/A')}")
            print(f"   {version2}/{stage2}: {m2.get('n_columns', 'N/A')}")
        
        print("\n" + "=" * 80)


def main():
    """Main function to demonstrate versioning."""
    print("=" * 80)
    print("DATA VERSIONING DEMO")
    print("=" * 80)
    
    versioner = DataVersioner()
    
    # Display all versions
    versioner.display_versions()
    
    # Get latest version
    latest = versioner.get_latest()
    if latest:
        print(f"\n📌 Latest version: {latest['version']}/{latest['stage']}")


if __name__ == "__main__":
    main()
