#!/usr/bin/env python3
"""
Align and compare two density maps using CryoAlign workflow.

This script orchestrates the full alignment workflow:
1. Flip target map
2. Run CryoAlign in docker
3. Transform map using eman2
4. Fitmap using chimerax
5. Calculate FSC to determine resolution

Usage:
    source_map.mrc: the source map the map that stay fixed
    target_map.mrc: the target map the map that is going to be aligned to the source map
    python align_and_compare_maps.py <source_map.mrc> <target_map.mrc> [options]

Output:
    Prints the resolution in Angstroms (the smaller the better).

Environment Variables:
    CRYOALIGN_DOCKER_CONTAINER: Docker container name (default: "cryo2")
    CRYOALIGN_DOCKER_MOUNT_PREFIX: Path prefix for docker mounts (default: "/data")
    CRYOALIGN_TRANSFORM_SCRIPT: Path to Transform_map.py script
    CHIMERAX_CMD: ChimeraX command (default: "chimerax")
    EMAN2_CONDA_ENV: Name of the eman2 conda environment (recommended)
    EMAN2_PYTHON: Python interpreter with eman2 (default: "python")
"""

import argparse
import json
import os
import shlex
import sys
import subprocess
import tempfile
import shutil
import re
from pathlib import Path


def get_env_var(name, default=None, required=False):
    """Get environment variable with optional default."""
    value = os.environ.get(name, default)
    if required and value is None:
        raise ValueError(f"Required environment variable {name} is not set")
    return value


def run_command(cmd, check=True, shell=False, env=None, cwd=None, quiet=False):
    """Run a shell command and return the result."""
    if not quiet:
        print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        result = subprocess.run(
            cmd,
            check=check,
            shell=shell,
            capture_output=True,
            text=True,
            env=env,
            cwd=cwd
        )
        if not quiet:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}", file=sys.stderr)
        if e.stdout:
            print(e.stdout, file=sys.stderr)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        raise


def extract_resolution_from_fsc_output(output_text):
    """Extract resolution value from FSC calculation output."""
    # Look for "Resolution: X.XX Å" pattern
    pattern = r'Resolution:\s*([\d.]+)\s*Å'
    match = re.search(pattern, output_text)
    if match:
        return float(match.group(1))
    
    # Alternative pattern: just a number followed by Å
    pattern2 = r'([\d.]+)\s*Å'
    matches = re.findall(pattern2, output_text)
    if matches:
        # Return the first resolution value found
        return float(matches[0])
    
    raise ValueError(f"Could not extract resolution from output: {output_text}")


def run_alignment_workflow(
    source_map, target_map, data_dir, output_dir, source_map_name, target_map_name,
    docker_container, docker_mount_prefix, docker_mount_prefix_val,
    transform_map_script, chimerax_cmd, eman2_conda_env, fitmap_script,
    cal_fsc_script, voxel_size, alg_type, source_contour_level, target_contour_level
):
    """
    Run alignment workflow (steps 2-5): CryoAlign -> Transform -> Fitmap -> FSC.
    
    Returns:
        resolution (float): Resolution in Angstroms
    """
    # Step 2: Run CryoAlign in docker
    print(f"Step 2: Running CryoAlign...", end=" ", flush=True)
    
    # Map data_dir to docker path
    data_dir_str = str(data_dir)
    if data_dir_str.startswith("/mnt/"):
        parts = data_dir_str.split("/")
        if len(parts) > 3:
            mount_point = "/".join(parts[:3])
            rel_path = data_dir_str[len(mount_point):]
            data_dir_docker = f"{docker_mount_prefix_val}{rel_path}"
        else:
            data_dir_docker = f"{docker_mount_prefix_val}{data_dir_str}"
    else:
        data_dir_docker = f"{docker_mount_prefix_val}{data_dir_str}"
    
    # Run CryoAlign in docker
    cryoalign_cmd_str = (
        f"cd /CryoAlign2/bin && ./CryoAlign "
        f"--data_dir {data_dir_docker} "
        f"--source_map {source_map_name} "
        f"--target_map {target_map_name} "
        f"--voxel_size {voxel_size} "
        f"--alg_type {alg_type} "
        f"--source_contour_level {source_contour_level} "
        f"--target_contour_level {target_contour_level}"
    )
    
    cryoalign_cmd = [
        "docker", "exec", "-i", docker_container,
        "bash", "-c", cryoalign_cmd_str
    ]
    
    run_command(cryoalign_cmd, quiet=True)
    print("Done")
    
    # Step 3: Generate transformed map (in eman2 environment)
    print(f"Step 3: Generating transformed map...", end=" ", flush=True)
    # CryoAlign outputs RT file in data_dir: source_map_target_map_RT.npy
    rt_file = None
    possible_rt_names = [
        f"{Path(source_map_name).stem}_{Path(target_map_name).stem}_RT.npy",
        f"{source_map_name.replace('.mrc', '')}_{target_map_name.replace('.mrc', '')}_RT.npy",
    ]
    
    for rt_name in possible_rt_names:
        rt_candidate = data_dir / rt_name
        if rt_candidate.exists():
            rt_file = rt_candidate
            break
    
    if rt_file is None:
        files_in_dir = list(data_dir.glob("*RT.npy"))
        if files_in_dir:
            rt_file = files_in_dir[0]
        else:
            raise FileNotFoundError(
                f"RT file not found in {data_dir}. "
                f"CryoAlign may have failed. Expected pattern: *RT.npy"
            )
    
    # Transform_map.py outputs - save to output directory
    transformed_map = output_dir / f"{Path(target_map_name).stem}_trans.map"
    
    # Validate transform_map_script
    if transform_map_script is None:
        # Try default location
        default_transform = Path("/home/daoyi/Github/cryoAlign2/script/Transform_map.py")
        if default_transform.exists():
            transform_map_script = str(default_transform)
        else:
            raise ValueError(
                "Transform_map.py script not found. "
                "Please set CRYOALIGN_TRANSFORM_SCRIPT environment variable or --transform_map_script argument."
            )
    
    # Run in eman2 conda environment
    rt_filename = rt_file.name
    target_map_path = data_dir / target_map_name
    
    if eman2_conda_env:
        # Verify conda environment exists
        try:
            info_result = subprocess.run(
                ["conda", "info", "--envs"],
                capture_output=True,
                text=True,
                check=True
            )
            env_lines = [line.strip() for line in info_result.stdout.split('\n') 
                       if line.strip() and not line.startswith('#')]
            env_names = []
            for line in env_lines:
                parts = line.split()
                if parts:
                    env_name = parts[0]
                    if env_name == '*':
                        if len(parts) > 1:
                            env_name = Path(parts[1]).name
                        else:
                            continue
                    env_names.append(env_name)
            
            if eman2_conda_env not in env_names:
                raise ValueError(
                    f"Eman2 conda environment '{eman2_conda_env}' not found. "
                    f"Available environments: {', '.join(env_names)}"
                )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            pass  # Continue anyway
        
        # Use conda run to execute in the specified environment (same pattern as relion_tools.py)
        cmd_str = f"python {shlex.quote(transform_map_script)} {shlex.quote(str(target_map_path))} {shlex.quote(rt_filename)}"
        conda_cmd = [
            "conda", "run", "-n", eman2_conda_env,
            "bash", "-c",
            f"cd {shlex.quote(str(data_dir))} && {cmd_str}"
        ]
        run_command(conda_cmd, quiet=True)
    else:
        # Fallback to default python
        run_command([
            "python",
            transform_map_script,
            str(target_map_path),
            rt_filename
        ], cwd=str(data_dir), quiet=True)
    print("Done")
    
    # Transform_map.py creates output in data_dir, move it to output_dir
    transformed_map_temp = data_dir / f"{Path(target_map_name).stem}_trans.map"
    if not transformed_map_temp.exists():
        alt_transformed = data_dir / f"{Path(target_map_name).stem}.map"
        if alt_transformed.exists():
            transformed_map_temp = alt_transformed
        else:
            raise FileNotFoundError(
                f"Transformed map not found in {data_dir}. "
                f"Transform_map.py may have failed."
            )
    
    # Move transformed map to output directory
    if transformed_map_temp.exists():
        shutil.move(str(transformed_map_temp), str(transformed_map))
    
    # Step 4: Fitmap using chimerax
    print(f"Step 4: Fitting maps using ChimeraX...", end=" ", flush=True)
    fitted_map = output_dir / f"{Path(target_map_name).stem}_trans_fitmap.mrc"
    run_command([
        chimerax_cmd,
        "--nogui",
        "--script",
        f"{fitmap_script} {source_map} {transformed_map} {fitted_map}"
    ], quiet=True)
    print("Done")
    
    # Step 5: Calculate FSC
    print(f"Step 5: Calculating FSC...", end=" ", flush=True)
    result = run_command([
        sys.executable,
        cal_fsc_script,
        str(source_map),
        str(fitted_map)
    ], quiet=True)
    
    # Extract resolution from output
    output_text = result.stdout + result.stderr
    resolution = extract_resolution_from_fsc_output(output_text)
    print(f"Done - Resolution: {resolution} Å")
    
    return resolution


def main():
    parser = argparse.ArgumentParser(
        description="Align and compare two density maps using CryoAlign workflow"
    )
    parser.add_argument("source_map", help="Source density map (MRC format)")
    parser.add_argument("target_map", help="Target density map (MRC format)")
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=5.0,
        help="Voxel size in Angstroms (default: 5.0)"
    )
    parser.add_argument(
        "--source_contour_level",
        type=float,
        default=0.01,
        help="Source contour level for CryoAlign (default: 0.01)"
    )
    parser.add_argument(
        "--target_contour_level",
        type=float,
        default=0.01,
        help="Target contour level for CryoAlign (default: 0.01)"
    )
    parser.add_argument(
        "--alg_type",
        type=str,
        default="global",
        choices=["global", "local"],
        help="CryoAlign algorithm type (default: global)"
    )
    parser.add_argument(
        "--docker_container",
        type=str,
        default=None,
        help="Docker container name for CryoAlign (default: from CRYOALIGN_DOCKER_CONTAINER env var or 'cryo2')"
    )
    parser.add_argument(
        "--docker_mount_prefix",
        type=str,
        default=None,
        help="Prefix for mounting paths in docker (default: from CRYOALIGN_DOCKER_MOUNT_PREFIX env var or '/data')"
    )
    parser.add_argument(
        "--flip_map_script",
        type=str,
        default=None,
        help="Path to flip_map.py script (default: auto-detect)"
    )
    parser.add_argument(
        "--transform_map_script",
        type=str,
        default='/home/daoyi/Github/CryoAlign2/script/Transform_map.py',
        help="Path to Transform_map.py script (default: from CRYOALIGN_TRANSFORM_SCRIPT env var or auto-detect)"
    )
    parser.add_argument(
        "--fitmap_script",
        type=str,
        default=None,
        help="Path to fitmap_chimerax.py script (default: auto-detect)"
    )
    parser.add_argument(
        "--cal_fsc_script",
        type=str,
        default=None,
        help="Path to cal_fsc.py script (default: auto-detect)"
    )
    parser.add_argument(
        "--chimerax_cmd",
        type=str,
        default=None,
        help="ChimeraX command (default: from CHIMERAX_CMD env var or 'chimerax')"
    )
    parser.add_argument(
        "--eman2_conda_env",
        type=str,
        default='eman2',
        help="Name of the eman2 conda environment (default: from EMAN2_CONDA_ENV env var)"
    )
    parser.add_argument(
        "--eman2_env",
        type=str,
        default=None,
        help="Command to activate eman2 environment (legacy, use --eman2_conda_env instead; default: from EMAN2_ENV env var)"
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=None,
        help="Working directory for intermediate files (default: temporary directory)"
    )
    parser.add_argument(
        "--keep_work_dir",
        action="store_true",
        help="Keep working directory after completion (default: delete)"
    )
    
    args = parser.parse_args()
    
    # Get environment variables with defaults
    docker_container = args.docker_container or get_env_var("CRYOALIGN_DOCKER_CONTAINER", "cryo2")
    docker_mount_prefix = args.docker_mount_prefix or get_env_var("CRYOALIGN_DOCKER_MOUNT_PREFIX", "/data")
    transform_map_script = args.transform_map_script or get_env_var("CRYOALIGN_TRANSFORM_SCRIPT")
    chimerax_cmd = args.chimerax_cmd or get_env_var("CHIMERAX_CMD", "chimerax")
    eman2_conda_env = args.eman2_conda_env or get_env_var("EMAN2_CONDA_ENV")
    eman2_env = args.eman2_env or get_env_var("EMAN2_ENV")  # Legacy support
    
    # Auto-detect script paths
    script_dir = Path(__file__).parent
    # Look for alignment tools in cryoagent/tools/alignment_tools/
    if (script_dir / "cryoagent" / "tools" / "alignment_tools").exists():
        alignment_tools_dir = script_dir / "cryoagent" / "tools" / "alignment_tools"
    elif (script_dir / "tools" / "alignment_tools").exists():
        alignment_tools_dir = script_dir / "tools" / "alignment_tools"
    else:
        alignment_tools_dir = script_dir
    
    flip_map_script = args.flip_map_script or str(alignment_tools_dir / "filp_map.py")
    fitmap_script = args.fitmap_script or str(alignment_tools_dir / "fitmap_chimerax.py")
    cal_fsc_script = args.cal_fsc_script or str(alignment_tools_dir / "cal_fsc.py")
    
    # Validate utility scripts exist
    if not Path(flip_map_script).exists():
        raise FileNotFoundError(f"Flip map script not found: {flip_map_script}")
    if not Path(fitmap_script).exists():
        raise FileNotFoundError(f"Fitmap script not found: {fitmap_script}")
    if not Path(cal_fsc_script).exists():
        raise FileNotFoundError(f"FSC calculation script not found: {cal_fsc_script}")
    
    # Validate input files
    source_map = Path(args.source_map).resolve()
    target_map = Path(args.target_map).resolve()
    
    if not source_map.exists():
        raise FileNotFoundError(f"Source map not found: {source_map}")
    if not target_map.exists():
        raise FileNotFoundError(f"Target map not found: {target_map}")
    
    # Create working directory
    # Use the parent directory of source map to ensure docker can access it
    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup_work_dir = False
    else:
        # Create work directory in the same location as source map (for docker access)
        # This ensures the directory is accessible to docker if source is in /mnt/
        source_parent = source_map.parent
        work_dir = source_parent / f"cryoalign_work_{source_map.stem}_{target_map.stem}"
        if work_dir.exists():
            # Clean up old work directory
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup_work_dir = not args.keep_work_dir
    
    # Setup data directory
    data_dir = source_map.parent
    source_map_name = source_map.name
    
    # Create output subfolder for generated files
    output_dir = data_dir / f"alignment_output_{source_map.stem}_{target_map.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create results file in output directory
    results_file = output_dir / f"alignment_results_{source_map.stem}_{target_map.stem}.txt"
    
    try:
        # First run: Non-flipped target map (steps 2-5)
        print("\n" + "="*60)
        print("RUN 1: Alignment with NON-FLIPPED target map")
        print("="*60)
        
        target_map_name = target_map.name
        target_map_in_data = data_dir / target_map_name
        
        # Copy target map to data directory if needed
        if not target_map_in_data.exists() or target_map_in_data.stat().st_mtime < target_map.stat().st_mtime:
            shutil.copy2(target_map, target_map_in_data)
        
        resolution_non_flipped = run_alignment_workflow(
            source_map, target_map_in_data, data_dir, output_dir, source_map_name, target_map_name,
            docker_container, docker_mount_prefix, docker_mount_prefix,
            transform_map_script, chimerax_cmd, eman2_conda_env, fitmap_script,
            cal_fsc_script, args.voxel_size, args.alg_type,
            args.source_contour_level, args.target_contour_level
        )
        
        print(f"\nNon-flipped target map resolution: {resolution_non_flipped} Å")
        
        # Save to file
        with open(results_file, 'w') as f:
            f.write(f"Alignment Results: {source_map_name} vs {target_map_name}\n")
            f.write("="*60 + "\n")
            f.write(f"Non-flipped target map: {resolution_non_flipped} Å\n")
        
        # Second run: Flipped target map (steps 1-5)
        print("\n" + "="*60)
        print("RUN 2: Alignment with FLIPPED target map")
        print("="*60)
        
        # Step 1: Flip target map
        print("Step 1: Flipping target map...", end=" ", flush=True)
        target_flipped = output_dir / f"{target_map.stem}_flip.mrc"
        run_command([
            sys.executable,
            flip_map_script,
            str(target_map),
            str(target_flipped)
        ], quiet=True)
        print("Done")
        
        target_flipped_name = target_flipped.name
        target_flipped_in_data = data_dir / target_flipped_name
        
        # Copy flipped map to data directory (CryoAlign needs it there)
        if not target_flipped_in_data.exists() or target_flipped_in_data.stat().st_mtime < target_flipped.stat().st_mtime:
            shutil.copy2(target_flipped, target_flipped_in_data)
        
        resolution_flipped = run_alignment_workflow(
            source_map, target_flipped_in_data, data_dir, output_dir, source_map_name, target_flipped_name,
            docker_container, docker_mount_prefix, docker_mount_prefix,
            transform_map_script, chimerax_cmd, eman2_conda_env, fitmap_script,
            cal_fsc_script, args.voxel_size, args.alg_type,
            args.source_contour_level, args.target_contour_level
        )
        
        print(f"\nFlipped target map resolution: {resolution_flipped} Å")
        
        # Append to file
        with open(results_file, 'a') as f:
            f.write(f"Flipped target map: {resolution_flipped} Å\n")
            f.write("="*60 + "\n")
        
        # Compare and determine best resolution
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        
        if resolution_non_flipped < resolution_flipped:
            best_resolution = resolution_non_flipped
            best_method = "Non-flipped"
        else:
            best_resolution = resolution_flipped
            best_method = "Flipped"
        
        print(f"Non-flipped: {resolution_non_flipped} Å")
        print(f"Flipped:     {resolution_flipped} Å")
        print(f"\nBest resolution: {best_resolution} Å ({best_method})")
        
        # Append comparison to file
        with open(results_file, 'a') as f:
            f.write(f"\nBest resolution: {best_resolution} Å ({best_method})\n")
        
        print(f"\nResults saved to: {results_file}")
        
        return best_resolution
        
    finally:
        if cleanup_work_dir and work_dir.exists():
            print(f"\nCleaning up working directory: {work_dir}")
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    try:
        resolution = main()
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

