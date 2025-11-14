#!/usr/bin/env python3
"""
Run the 2x2x2 CryoSPARC/RELION pipeline matrix in one shot.

This helper script automates the generation of per-combination master configs,
ensures dedicated RELION working directories exist, executes the CryoAgent
workflow for every backend combination (ccc, ccr, ..., rrr), and collects the
result JSON files inside combo-specific sub-folders under the outputs tree.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set


STAGE_NAMES = ["preprocessing", "particle_picking", "reconstruction"]
LETTER_TO_GROUP = {"c": "cryosparc", "r": "relion"}
DEFAULT_COMBOS = [
    "ccc",
    "ccr",
    "crc",
    "crr",
    "rcc",
    "rcr",
    "rrc",
    "rrr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all CryoSPARC/RELION pipeline combinations automatically."
    )
    parser.add_argument(
        "--master-config",
        default="configs/master_config.json",
        help="Path to the base master configuration JSON.",
    )
    parser.add_argument(
        "--workflow-script",
        default="cryoagent_workflow.py",
        help="Entry script that wraps CryoAgentMasterWorkflow.",
    )
    parser.add_argument(
        "--relion-base-dir",
        help=(
            "Parent directory that will contain combo-specific RELION working dirs. "
            "Defaults to the parent of relion.relion_dir from the base config."
        ),
    )
    parser.add_argument(
        "--output-base-dir",
        default="outputs/test_combos",
        help="Root folder for storing combo specific outputs (inside outputs/).",
    )
    parser.add_argument(
        "--experiment-name",
        help="Logical experiment label used under the output base dir. Defaults to relion base name.",
    )
    parser.add_argument(
        "--generated-config-dir",
        default="configs/generated_combos",
        help="Directory to store auto-generated per-combo master configs.",
    )
    parser.add_argument(
        "--combos",
        nargs="*",
        default=DEFAULT_COMBOS,
        help="Subset of combos to run (each string must be 3 letters of c/r).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only prepare configs/directories without launching the workflows.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass --verbose to cryoagent_workflow.py for each run.",
    )
    parser.add_argument(
        "--keep-original-outputs",
        action="store_true",
        help="Copy result JSONs to combo folders instead of moving them.",
    )
    return parser.parse_args()


def load_master_config(config_path: Path) -> Dict:
    with config_path.open("r") as handle:
        return json.load(handle)


def snapshot_files(root: Path) -> Set[Path]:
    if not root.exists():
        return set()
    return {path.relative_to(root) for path in root.rglob("*") if path.is_file()}


def move_new_outputs(
    root: Path, before: Set[Path], destination: Path, keep_original: bool = False
) -> List[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    after = snapshot_files(root)
    new_paths = sorted(after - before)
    relocated: List[Path] = []

    for rel_path in new_paths:
        src = root / rel_path
        dest = destination / rel_path.name
        counter = 1
        while dest.exists():
            dest = destination / f"{rel_path.stem}_{counter}{rel_path.suffix}"
            counter += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        if keep_original:
            dest.write_bytes(src.read_bytes())
        else:
            src.replace(dest)
        relocated.append(dest)
    return relocated


def copy_outputs_to_destination(source_files: List[Path], destination: Path) -> List[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []
    for src in source_files:
        if not src.exists():
            continue
        dest = destination / src.name
        counter = 1
        while dest.exists():
            dest = destination / f"{src.stem}_{counter}{src.suffix}"
            counter += 1
        shutil.copy2(src, dest)
        copied.append(dest)
    return copied


def validate_combo(code: str) -> str:
    normalized = code.strip().lower()
    if len(normalized) != len(STAGE_NAMES):
        raise ValueError(f"Combo '{code}' must have {len(STAGE_NAMES)} letters.")
    for letter in normalized:
        if letter not in LETTER_TO_GROUP:
            raise ValueError(f"Combo '{code}' contains invalid letter '{letter}'.")
    return normalized


def build_combo_config(
    base_config: Dict,
    combo: str,
    relion_base: Path,
) -> Dict:
    updated = copy.deepcopy(base_config)
    relion_dir = relion_base / combo
    relion_dir.mkdir(parents=True, exist_ok=True)
    updated.setdefault("relion", {})
    updated["relion"]["relion_dir"] = str(relion_dir)

    for stage_idx, stage_name in enumerate(STAGE_NAMES):
        letter = combo[stage_idx]
        agent_group = LETTER_TO_GROUP[letter]
        for stage in updated["master_workflow"]["stages"]:
            if stage["name"] == stage_name:
                stage["agent_group"] = agent_group
                break
        else:
            raise KeyError(f"Stage '{stage_name}' not found in master config.")

    run_metadata = updated.setdefault("run_matrix_metadata", {})
    run_metadata["combo"] = combo
    run_metadata["stage_backend_sequence"] = {
        stage: LETTER_TO_GROUP[combo[idx]] for idx, stage in enumerate(STAGE_NAMES)
    }
    run_metadata["relion_dir"] = str(relion_dir)
    return updated


def write_combo_config(config: Dict, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as handle:
        json.dump(config, handle, indent=2)


def determine_relion_base(args: argparse.Namespace, base_config: Dict, root_dir: Path) -> Path:
    if args.relion_base_dir:
        candidate = Path(args.relion_base_dir).expanduser()
        if not candidate.is_absolute():
            candidate = root_dir / candidate
        return candidate.resolve()
    relion_dir = base_config.get("relion", {}).get("relion_dir")
    if not relion_dir:
        raise ValueError("relion.relion_dir missing from master config; please pass --relion-base-dir.")
    relion_path = Path(relion_dir).expanduser().resolve()
    return relion_path.parent


def determine_experiment_name(args: argparse.Namespace, relion_base: Path) -> str:
    if args.experiment_name:
        return args.experiment_name
    return relion_base.name


def run_workflow(script: Path, config_path: Path, verbose: bool = False, dry_run: bool = False) -> int:
    cmd = [sys.executable, str(script), "--config", str(config_path)]
    if verbose:
        cmd.append("--verbose")
    if dry_run:
        cmd.append("--dry-run")
    proc = subprocess.run(cmd)
    return proc.returncode


def main() -> None:
    args = parse_args()

    root_dir = Path(__file__).resolve().parent

    master_config_path = Path(args.master_config).expanduser()
    if not master_config_path.is_absolute():
        master_config_path = (root_dir / master_config_path).resolve()
    else:
        master_config_path = master_config_path.resolve()

    workflow_script = Path(args.workflow_script).expanduser()
    if not workflow_script.is_absolute():
        workflow_script = (root_dir / workflow_script).resolve()
    else:
        workflow_script = workflow_script.resolve()

    base_config = load_master_config(master_config_path)
    relion_base = determine_relion_base(args, base_config, root_dir)
    experiment_name = determine_experiment_name(args, relion_base)

    outputs_root = (root_dir / "outputs").resolve()
    outputs_root.mkdir(exist_ok=True)

    output_base_input = Path(args.output_base_dir).expanduser()
    if not output_base_input.is_absolute():
        output_base_input = root_dir / output_base_input
    combo_outputs_base = output_base_input.resolve()
    combo_outputs_base.mkdir(parents=True, exist_ok=True)

    generated_dir_input = Path(args.generated_config_dir).expanduser()
    if not generated_dir_input.is_absolute():
        generated_dir_input = root_dir / generated_dir_input
    generated_dir = generated_dir_input.resolve()
    generated_dir.mkdir(parents=True, exist_ok=True)

    normalized_combos = [validate_combo(code) for code in args.combos]

    combo_config_paths: Dict[str, Path] = {}
    combo_output_dirs: Dict[str, Path] = {}

    for combo in normalized_combos:
        combo_config = build_combo_config(base_config, combo, relion_base)
        combo_config_path = generated_dir / f"master_config_{combo}.json"
        write_combo_config(combo_config, combo_config_path)
        combo_config_paths[combo] = combo_config_path

        combo_output_dir = combo_outputs_base / experiment_name / combo
        combo_output_dir.mkdir(parents=True, exist_ok=True)
        combo_output_dirs[combo] = combo_output_dir

    grouped_combos: Dict[str, List[str]] = {}
    prefix_order: List[str] = []
    for combo in normalized_combos:
        prefix = combo[:2]
        if prefix not in grouped_combos:
            grouped_combos[prefix] = []
            prefix_order.append(prefix)
        grouped_combos[prefix].append(combo)

    summary: Dict[str, Dict[str, object]] = {}

    for prefix in prefix_order:
        combos_for_prefix = grouped_combos[prefix]
        primary_combo = combos_for_prefix[0]
        replicas = combos_for_prefix[1:]

        print(f"\n=== 🚀 Launching group {prefix.upper()} via {primary_combo.upper()} ===")

        before_snapshot = snapshot_files(outputs_root)
        result_code = 0
        relocated_files: List[Path] = []

        if not args.dry_run:
            result_code = run_workflow(
                workflow_script,
                combo_config_paths[primary_combo],
                verbose=args.verbose,
                dry_run=False,
            )
        else:
            print("Dry run enabled; skipping workflow execution.")

        relocated_files = move_new_outputs(
            outputs_root,
            before_snapshot,
            combo_output_dirs[primary_combo],
            keep_original=args.keep_original_outputs,
        )

        summary[primary_combo] = {
            "return_code": result_code,
            "relocated_files": relocated_files,
            "executed": True,
            "copied_from": None,
        }

        if result_code != 0:
            print(f"⚠️ Combo {primary_combo.upper()} completed with errors (exit code {result_code}).")
        else:
            print(
                f"✅ Combo {primary_combo.upper()} finished. Stored {len(relocated_files)} outputs in {combo_output_dirs[primary_combo]}."
            )

        for replica_combo in replicas:
            copied_files = copy_outputs_to_destination(
                relocated_files,
                combo_output_dirs[replica_combo],
            )
            summary[replica_combo] = {
                "return_code": 0,
                "relocated_files": copied_files,
                "executed": False,
                "copied_from": primary_combo,
            }
            print(
                f"📁 Copied outputs from {primary_combo.upper()} to {replica_combo.upper()} ({len(copied_files)} files)."
            )

    print("\n=== 📦 Run Matrix Summary ===")
    for combo, info in summary.items():
        status = "OK" if info["return_code"] == 0 else f"exit={info['return_code']}"
        files = info["relocated_files"]
        if info.get("executed"):
            mode_desc = "executed"
        else:
            copied_from = info.get("copied_from")
            mode_desc = f"copied from {copied_from.upper()}" if copied_from else "copied"
        print(f"- {combo.upper()}: {status}, {mode_desc}, outputs: {len(files)}")
        for path in files:
            print(f"    • {path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
    except Exception as exc:
        print(f"\nFatal error: {exc}")
        sys.exit(1)

