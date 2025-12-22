#!/usr/bin/env python3
"""
Compare all density maps in a folder and create a resolution relationship matrix.

This script finds all *_volume.mrc files in a given folder, compares all pairs
using align_and_compare_maps.py, and creates a triangular matrix showing the
best resolution between each pair.

Usage:
    python compare_all_densities.py <folder_path> [options]

Output:
    - Prints progress for each comparison
    - Creates a resolution matrix file (CSV format)
    - Creates a summary text file
"""

import argparse
import csv
import itertools
import re
import subprocess
import sys
from pathlib import Path


def find_volume_maps(folder_path):
    """Find all *_volume.mrc files in the given folder."""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    volume_maps = sorted(folder.glob("*_volume.mrc"))
    if not volume_maps:
        raise ValueError(f"No *_volume.mrc files found in {folder_path}")
    
    return volume_maps


def run_comparison(source_map, target_map, align_script, **kwargs):
    """
    Run align_and_compare_maps.py for a pair of maps.
    
    Returns:
        float: Best resolution in Angstroms
    """
    cmd = [sys.executable, str(align_script), str(source_map), str(target_map)]
    
    # Add optional arguments
    if kwargs.get("voxel_size"):
        cmd.extend(["--voxel_size", str(kwargs["voxel_size"])])
    if kwargs.get("source_contour_level"):
        cmd.extend(["--source_contour_level", str(kwargs["source_contour_level"])])
    if kwargs.get("target_contour_level"):
        cmd.extend(["--target_contour_level", str(kwargs["target_contour_level"])])
    if kwargs.get("alg_type"):
        cmd.extend(["--alg_type", kwargs["alg_type"]])
    if kwargs.get("docker_container"):
        cmd.extend(["--docker_container", kwargs["docker_container"]])
    if kwargs.get("docker_mount_prefix"):
        cmd.extend(["--docker_mount_prefix", kwargs["docker_mount_prefix"]])
    if kwargs.get("transform_map_script"):
        cmd.extend(["--transform_map_script", kwargs["transform_map_script"]])
    if kwargs.get("fitmap_script"):
        cmd.extend(["--fitmap_script", kwargs["fitmap_script"]])
    if kwargs.get("cal_fsc_script"):
        cmd.extend(["--cal_fsc_script", kwargs["cal_fsc_script"]])
    if kwargs.get("chimerax_cmd"):
        cmd.extend(["--chimerax_cmd", kwargs["chimerax_cmd"]])
    if kwargs.get("eman2_conda_env"):
        cmd.extend(["--eman2_conda_env", kwargs["eman2_conda_env"]])
    if kwargs.get("keep_work_dir"):
        cmd.append("--keep_work_dir")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extract resolution from output
        # The script prints "Best resolution: X.XX Å (Method)" at the end
        output = result.stdout + result.stderr
        # First try: "Best resolution: X.XX Å"
        pattern = r'Best resolution:\s*([\d.]+)\s*Å'
        match = re.search(pattern, output)
        if match:
            return float(match.group(1))
        # Second try: any "X.XX Å" pattern, prefer the last one
        pattern2 = r'([\d.]+)\s*Å'
        matches = re.findall(pattern2, output)
        if matches:
            # Get the last one (should be the best resolution)
            return float(matches[-1])
        # Third try: look for resolution in the return value if script was modified
        # Check if there's a numeric value at the end of output
        pattern3 = r'(\d+\.\d+)\s*$'
        match3 = re.search(pattern3, output.strip())
        if match3:
            return float(match3.group(1))
        raise ValueError(f"Could not extract resolution from output:\n{output}")
    except subprocess.CalledProcessError as e:
        print(f"Error comparing {source_map.name} vs {target_map.name}:", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        raise


def create_resolution_matrix(volume_maps, align_script, output_dir, **kwargs):
    """
    Compare all pairs of volume maps and create a resolution matrix.
    
    Args:
        volume_maps: List of Path objects for volume maps
        align_script: Path to align_and_compare_maps.py
        output_dir: Directory to save results
        **kwargs: Additional arguments to pass to align_and_compare_maps.py
    
    Returns:
        dict: Dictionary mapping (map1, map2) -> resolution
    """
    n_maps = len(volume_maps)
    n_combinations = n_maps * (n_maps - 1) // 2
    
    print(f"Found {n_maps} volume maps")
    print(f"Will perform {n_combinations} pairwise comparisons")
    print()
    
    results = {}
    combination_num = 0
    
    # Generate all pairwise combinations
    for map1, map2 in itertools.combinations(volume_maps, 2):
        combination_num += 1
        print(f"[{combination_num}/{n_combinations}] Comparing {map1.name} vs {map2.name}...", end=" ", flush=True)
        
        try:
            resolution = run_comparison(map1, map2, align_script, **kwargs)
            results[(map1, map2)] = resolution
            print(f"Done - Resolution: {resolution:.2f} Å")
        except Exception as e:
            print(f"Failed: {e}", file=sys.stderr)
            results[(map1, map2)] = None
    
    return results


def save_matrix(volume_maps, results, output_dir):
    """
    Save resolution matrix to CSV and text files.
    
    Args:
        volume_maps: List of Path objects for volume maps
        results: Dictionary mapping (map1, map2) -> resolution
        output_dir: Directory to save output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_maps = len(volume_maps)
    map_names = [m.name for m in volume_maps]
    
    # Create matrix (symmetric, with diagonal as 0 or map name)
    matrix = [[None] * n_maps for _ in range(n_maps)]
    
    # Fill matrix with results
    for (map1, map2), resolution in results.items():
        i = volume_maps.index(map1)
        j = volume_maps.index(map2)
        matrix[i][j] = resolution
        matrix[j][i] = resolution  # Symmetric
    
    # Set diagonal to 0 (same map)
    for i in range(n_maps):
        matrix[i][i] = 0.0
    
    # Save as CSV
    csv_file = output_dir / "resolution_matrix.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header row
        writer.writerow([''] + map_names)
        # Data rows
        for i, map_name in enumerate(map_names):
            row = [map_name] + [
                f"{matrix[i][j]:.2f}" if matrix[i][j] is not None else "N/A"
                for j in range(n_maps)
            ]
            writer.writerow(row)
    
    # Save as formatted text file
    txt_file = output_dir / "resolution_matrix.txt"
    with open(txt_file, 'w') as f:
        f.write("Resolution Matrix (Best Resolution in Angstroms)\n")
        f.write("=" * 80 + "\n\n")
        
        # Calculate column widths
        max_name_len = max(len(name) for name in map_names)
        col_width = max(max_name_len, 10)
        
        # Header
        f.write(f"{'':<{max_name_len}}  ")
        for name in map_names:
            f.write(f"{name:<{col_width}}  ")
        f.write("\n")
        f.write("-" * (max_name_len + (col_width + 2) * n_maps) + "\n")
        
        # Data rows
        for i, map_name in enumerate(map_names):
            f.write(f"{map_name:<{max_name_len}}  ")
            for j in range(n_maps):
                if matrix[i][j] is not None:
                    if i == j:
                        f.write(f"{'0.00':<{col_width}}  ")
                    else:
                        f.write(f"{matrix[i][j]:.2f} Å{'':<{col_width-6}}  ")
                else:
                    f.write(f"{'N/A':<{col_width}}  ")
            f.write("\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Note: Lower values indicate better alignment (higher resolution)\n")
        f.write("Diagonal values are 0 (same map)\n")
    
    # Save summary statistics
    summary_file = output_dir / "comparison_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Pairwise Comparison Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total maps: {n_maps}\n")
        f.write(f"Total comparisons: {len(results)}\n\n")
        
        # List all pairs with resolutions
        f.write("Pairwise Resolutions:\n")
        f.write("-" * 80 + "\n")
        for (map1, map2), resolution in sorted(results.items()):
            if resolution is not None:
                f.write(f"{map1.name} vs {map2.name}: {resolution:.2f} Å\n")
            else:
                f.write(f"{map1.name} vs {map2.name}: FAILED\n")
        
        # Statistics
        valid_resolutions = [r for r in results.values() if r is not None]
        if valid_resolutions:
            f.write("\n" + "-" * 80 + "\n")
            f.write("Statistics:\n")
            f.write(f"  Best (lowest) resolution: {min(valid_resolutions):.2f} Å\n")
            f.write(f"  Worst (highest) resolution: {max(valid_resolutions):.2f} Å\n")
            f.write(f"  Average resolution: {sum(valid_resolutions)/len(valid_resolutions):.2f} Å\n")
            f.write(f"  Successful comparisons: {len(valid_resolutions)}/{len(results)}\n")
    
    print(f"\nResults saved to:")
    print(f"  - CSV matrix: {csv_file}")
    print(f"  - Text matrix: {txt_file}")
    print(f"  - Summary: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare all density maps in a folder and create a resolution matrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python compare_all_densities.py /path/to/cryosparc/project

  # With custom parameters
  python compare_all_densities.py /path/to/folder --voxel_size 3.0 --alg_type local

  # Keep intermediate work directories
  python compare_all_densities.py /path/to/folder --keep_work_dir
        """
    )
    
    parser.add_argument(
        "folder",
        help="Folder containing *_volume.mrc files to compare"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: <folder>/comparison_results)"
    )
    parser.add_argument(
        "--align_script",
        type=str,
        default=None,
        help="Path to align_and_compare_maps.py (default: auto-detect)"
    )
    
    # Pass through arguments for align_and_compare_maps.py
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
        help="Docker container name for CryoAlign"
    )
    parser.add_argument(
        "--docker_mount_prefix",
        type=str,
        default=None,
        help="Prefix for mounting paths in docker"
    )
    parser.add_argument(
        "--transform_map_script",
        type=str,
        default=None,
        help="Path to Transform_map.py script"
    )
    parser.add_argument(
        "--fitmap_script",
        type=str,
        default=None,
        help="Path to fitmap_chimerax.py script"
    )
    parser.add_argument(
        "--cal_fsc_script",
        type=str,
        default=None,
        help="Path to cal_fsc.py script"
    )
    parser.add_argument(
        "--chimerax_cmd",
        type=str,
        default=None,
        help="ChimeraX command"
    )
    parser.add_argument(
        "--eman2_conda_env",
        type=str,
        default='eman2',
        help="Name of the eman2 conda environment"
    )
    parser.add_argument(
        "--keep_work_dir",
        action="store_true",
        help="Keep working directories after completion"
    )
    
    args = parser.parse_args()
    
    # Find volume maps
    folder_path = Path(args.folder).resolve()
    volume_maps = find_volume_maps(folder_path)
    
    # Find align script
    if args.align_script:
        align_script = Path(args.align_script).resolve()
    else:
        # Auto-detect: look in same directory as this script
        script_dir = Path(__file__).parent
        align_script = script_dir / "align_and_compare_maps.py"
        if not align_script.exists():
            raise FileNotFoundError(
                f"align_and_compare_maps.py not found. "
                f"Please specify --align_script or ensure it's in {script_dir}"
            )
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = folder_path / "comparison_results"
    
    # Prepare kwargs for align_and_compare_maps.py
    kwargs = {
        "voxel_size": args.voxel_size,
        "source_contour_level": args.source_contour_level,
        "target_contour_level": args.target_contour_level,
        "alg_type": args.alg_type,
        "docker_container": args.docker_container,
        "docker_mount_prefix": args.docker_mount_prefix,
        "transform_map_script": args.transform_map_script,
        "fitmap_script": args.fitmap_script,
        "cal_fsc_script": args.cal_fsc_script,
        "chimerax_cmd": args.chimerax_cmd,
        "eman2_conda_env": args.eman2_conda_env,
        "keep_work_dir": args.keep_work_dir,
    }
    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    
    # Run comparisons
    print("=" * 80)
    print("Density Map Comparison")
    print("=" * 80)
    print(f"Folder: {folder_path}")
    print(f"Align script: {align_script}")
    print()
    
    results = create_resolution_matrix(volume_maps, align_script, output_dir, **kwargs)
    
    # Save matrix
    print("\n" + "=" * 80)
    print("Creating resolution matrix...")
    print("=" * 80)
    save_matrix(volume_maps, results, output_dir)
    
    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

