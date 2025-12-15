#!/usr/bin/env python3
"""
Script to check conda environments and required file paths.

This script verifies that:
1. Required conda environments exist (relion, helicon, cryosift_env)
2. Required files exist (relion exe, cryosift weights, cryosift evaluator script)
"""

import sys
import subprocess
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from cryoagent.config.config_loader import ConfigLoader


def check_conda_available() -> bool:
    """Check if conda is available in the system."""
    try:
        result = subprocess.run(
            ["conda", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_conda_envs() -> List[str]:
    """Get list of all available conda environments."""
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return []
        
        # Parse conda env list output
        # Format: env_name    /path/to/env
        envs = []
        for line in result.stdout.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Extract environment name (first column)
            parts = line.split()
            if parts:
                env_name = parts[0]
                # Skip base environment path entries
                if env_name and not env_name.startswith('/'):
                    envs.append(env_name)
        return envs
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"⚠️  Error getting conda environments: {e}")
        return []


def conda_env_exists(env_name: str, available_envs: Optional[List[str]] = None) -> bool:
    """Check if a conda environment exists."""
    if available_envs is None:
        available_envs = get_conda_envs()
    return env_name in available_envs


def check_file_exists(file_path: str) -> Tuple[bool, str]:
    """Check if a file exists and return status with message."""
    path = Path(file_path)
    if path.exists():
        if path.is_file():
            return True, f"✅ File exists: {file_path}"
        elif path.is_dir():
            return False, f"⚠️  Path exists but is a directory (not a file): {file_path}"
        else:
            return False, f"⚠️  Path exists but is not a regular file: {file_path}"
    else:
        return False, f"❌ File not found: {file_path}"


def load_raw_config(config_path: str) -> Dict[str, Any]:
    """Load raw JSON config for accessing fields not in Pydantic models."""
    with open(config_path, 'r') as f:
        return json.load(f)


def test_conda_environments(config, raw_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Test if required conda environments exist."""
    print("🐍 Testing Conda Environments")
    print("=" * 50)
    
    if not check_conda_available():
        print("❌ Conda is not available in the system")
        print("   Please install Anaconda or Miniconda first")
        return False, []
    
    print("✅ Conda is available")
    print()
    
    # Get all available environments once
    available_envs = get_conda_envs()
    if not available_envs:
        print("⚠️  No conda environments found")
        print("   This might indicate conda is not properly initialized")
        return False, []
    
    all_passed = True
    results = []
    
    # Check relion environment - access from raw config since it's not in Pydantic model
    relion_backend = raw_config.get("relion", {}).get("backend_execution", {})
    relion_env = relion_backend.get("conda_env", "relion-5.0")
    print(f"🔍 Checking relion conda environment: '{relion_env}'")
    if conda_env_exists(relion_env, available_envs):
        print(f"   ✅ Environment '{relion_env}' exists")
        results.append(f"✅ relion environment '{relion_env}' exists")
    else:
        print(f"   ❌ Environment '{relion_env}' not found")
        results.append(f"❌ relion environment '{relion_env}' not found")
        all_passed = False
    print()
    
    # Check helicon environment - access from raw config
    transition_config = raw_config.get("transition", {})
    micrograph_helicon = transition_config.get("micrograph_conversion", {}).get("helicon", {})
    particle_helicon = transition_config.get("particle_conversion", {}).get("helicon", {})
    
    # Use the first one found (they should be the same)
    helicon_env = micrograph_helicon.get("conda_env") or particle_helicon.get("conda_env") or "helicon"
    
    print(f"🔍 Checking helicon conda environment: '{helicon_env}'")
    if conda_env_exists(helicon_env, available_envs):
        print(f"   ✅ Environment '{helicon_env}' exists")
        results.append(f"✅ helicon environment '{helicon_env}' exists")
    else:
        print(f"   ❌ Environment '{helicon_env}' not found")
        results.append(f"❌ helicon environment '{helicon_env}' not found")
        all_passed = False
    print()
    
    # Check cryosift_env environment - access from raw config
    cryosift_config = raw_config.get("cryosift", {})
    cryosift_env = cryosift_config.get("cryosift_env", "magellon2DAssess")
    print(f"🔍 Checking cryosift_env conda environment: '{cryosift_env}'")
    if conda_env_exists(cryosift_env, available_envs):
        print(f"   ✅ Environment '{cryosift_env}' exists")
        results.append(f"✅ cryosift_env environment '{cryosift_env}' exists")
    else:
        print(f"   ❌ Environment '{cryosift_env}' not found")
        results.append(f"❌ cryosift_env environment '{cryosift_env}' not found")
        all_passed = False
    print()
    
    return all_passed, results


def test_file_paths(config, raw_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Test if required file paths exist."""
    print("📁 Testing Required Files")
    print("=" * 50)
    
    all_passed = True
    results = []
    
    # Check relion executable
    relion_exe = config.relion.relion_exe
    print(f"🔍 Checking relion executable: '{relion_exe}'")
    exists, message = check_file_exists(relion_exe)
    print(f"   {message}")
    results.append(message)
    if not exists:
        all_passed = False
    print()
    
    # Check cryosift weights path - access from raw config
    cryosift_config = raw_config.get("cryosift", {})
    cryosift_weights = cryosift_config.get("cryosift_weights_path", "")
    print(f"🔍 Checking cryosift weights path: '{cryosift_weights}'")
    if cryosift_weights:
        exists, message = check_file_exists(cryosift_weights)
        print(f"   {message}")
        results.append(message)
        if not exists:
            all_passed = False
    else:
        print("   ❌ cryosift_weights_path not configured")
        results.append("❌ cryosift_weights_path not configured")
        all_passed = False
    print()
    
    # Check cryosift evaluator script path - access from raw config
    cryosift_evaluator = cryosift_config.get("cryosift_evaluator_script_path", "")
    print(f"🔍 Checking cryosift evaluator script path: '{cryosift_evaluator}'")
    if cryosift_evaluator:
        exists, message = check_file_exists(cryosift_evaluator)
        print(f"   {message}")
        results.append(message)
        if not exists:
            all_passed = False
    else:
        print("   ❌ cryosift_evaluator_script_path not configured")
        results.append("❌ cryosift_evaluator_script_path not configured")
        all_passed = False
    print()
    
    return all_passed, results


def main() -> int:
    """Main function to run all checks."""
    print("🚀 Conda Environments and Files Check")
    print("=" * 60)
    print()
    
    try:
        # Load configuration
        print("📋 Loading configuration...")
        config_path = "configs/master_config.json"
        config_loader = ConfigLoader(config_path)
        config = config_loader.load_config()
        
        # Also load raw JSON for accessing fields not in Pydantic models
        raw_config = load_raw_config(config_path)
        
        print("✅ Configuration loaded successfully!")
        print()
    except FileNotFoundError:
        print("❌ Configuration file not found: configs/master_config.json")
        print("   Please ensure configs/master_config.json exists in the project root")
        return 1
    except Exception as exc:
        print(f"❌ Failed to load configuration: {exc}")
        return 1
    
    # Test conda environments
    env_success, env_results = test_conda_environments(config, raw_config)
    
    # Test file paths
    file_success, file_results = test_file_paths(config, raw_config)
    
    # Summary
    print()
    print("📊 Test Results Summary")
    print("=" * 60)
    
    print("\n🐍 Conda Environments:")
    for result in env_results:
        print(f"   {result}")
    
    print("\n📁 Required Files:")
    for result in file_results:
        print(f"   {result}")
    
    print()
    if env_success and file_success:
        print("🎉 ALL TESTS PASSED!")
        print("   ✅ All conda environments exist")
        print("   ✅ All required files are found")
        print("   ✅ System is ready for workflow execution")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        if not env_success:
            print("   ❌ Some conda environments are missing")
            print("      Run 'bash install_all_envs.sh' to install missing environments")
        if not file_success:
            print("   ❌ Some required files are missing")
            print("      Please check the file paths in configs/master_config.json")
        return 1


if __name__ == "__main__":
    sys.exit(main())

