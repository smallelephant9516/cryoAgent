#!/usr/bin/env python3
"""Verify CryoSift configuration: conda env, weights, and evaluator script."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent))


def get_conda_envs() -> List[str]:
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
        envs = []
        for line in result.stdout.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts and not parts[0].startswith("/"):
                envs.append(parts[0])
        return envs
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def load_raw_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    print("🧪 Testing CryoSift Connection")
    print("=" * 50)

    config_path = "configs/master_config.json"
    try:
        raw = load_raw_config(config_path)
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        return 1
    except Exception as exc:
        print(f"❌ Failed to load configuration: {exc}")
        return 1

    cryosift = raw.get("cryosift") or {}
    cryosift_env = cryosift.get("cryosift_env", "magellon2DAssess")
    weights = cryosift.get("cryosift_weights_path", "")
    evaluator = cryosift.get("cryosift_evaluator_script_path", "")

    all_ok = True

    print(f"🔍 Checking CryoSift conda environment: '{cryosift_env}'")
    available = get_conda_envs()
    if cryosift_env in available:
        print(f"   ✅ Environment '{cryosift_env}' exists")
    else:
        print(f"   ❌ Environment '{cryosift_env}' not found")
        print("      Run: bash install.sh  (step for magellon2DAssess / CryoSift)")
        all_ok = False
    print()

    print(f"🔍 Checking CryoSift weights: '{weights}'")
    if weights and Path(weights).is_file():
        print(f"   ✅ Weights file found")
    elif not weights:
        print("   ❌ cryosift_weights_path not configured in master_config.json")
        all_ok = False
    else:
        print(f"   ❌ Weights file not found: {weights}")
        all_ok = False
    print()

    print(f"🔍 Checking CryoSift evaluator script: '{evaluator}'")
    if evaluator and Path(evaluator).is_file():
        print(f"   ✅ Evaluator script found")
    elif not evaluator:
        print("   ❌ cryosift_evaluator_script_path not configured in master_config.json")
        all_ok = False
    else:
        print(f"   ❌ Evaluator script not found: {evaluator}")
        all_ok = False
    print()

    if all_ok:
        print("✅ CryoSift connection checks passed")
        return 0

    print("❌ CryoSift connection checks failed")
    print("   Configure cryosift.* paths in configs/master_config.json and install the CryoSift conda env")
    return 1


if __name__ == "__main__":
    sys.exit(main())
