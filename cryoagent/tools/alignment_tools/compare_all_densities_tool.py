"""Tool wrapper for comparing all density maps in a folder.

This module provides a LangChain tool wrapper for the compare_all_densities.py script,
which performs pairwise comparisons of all density maps in a folder and creates
resolution matrices and clustering results.

Example usage in an agent:
    from cryoagent.tools.alignment_tools import CompareAllDensitiesTool
    
    # Create the tool
    tool = CompareAllDensitiesTool.create_compare_all_densities_tool()
    
    # Add to agent's tools list
    tools = [tool, ...]
    
    # Agent can then use it with:
    # - Simple: "/path/to/folder"
    # - JSON: '{"folder": "/path/to/folder", "voxel_size": 3.0}'
    # - Key-value: "folder=/path/to/folder,voxel_size=3.0"
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from langchain.tools import Tool


class CompareAllDensitiesTool:
    """Tool factory for comparing all density maps in a folder."""
    
    @staticmethod
    def create_compare_all_densities_tool(
        compare_script: Optional[str] = None,
        align_script: Optional[str] = None,
        default_voxel_size: float = 5.0,
        default_alg_type: str = "global"
    ) -> Tool:
        """
        Create a tool for comparing all density maps in a folder.
        
        Args:
            compare_script: Path to compare_all_densities.py script (auto-detected if None)
            align_script: Path to align_and_compare_maps.py script (auto-detected if None)
            default_voxel_size: Default voxel size in Angstroms
            default_alg_type: Default algorithm type (global/local)
        
        Returns:
            LangChain Tool instance
        """
        def _compare_all_densities_tool(input_str: str) -> str:
            """
            Compare all density maps in a folder and create a resolution matrix.
            
            Input format (JSON string or key=value pairs):
            {
                "folder": "/path/to/folder",
                "output_dir": "/path/to/output" (optional),
                "voxel_size": 5.0 (optional),
                "source_contour_level": optional (if not provided, RMS will be calculated automatically),
                "target_contour_level": optional (if not provided, RMS will be calculated automatically),
                "alg_type": "global" (optional, "global" or "local"),
                "docker_container": "cryo2" (optional),
                "docker_mount_prefix": "/data" (optional),
                "resolution_threshold": 20.0 (optional),
                "n_clusters": null (optional, auto-detect if None),
                "cluster_method": "spectral" (optional),
                "keep_work_dir": false (optional),
                "no_rms_threshold": false (optional),
                "class_resolutions": optional (list of dicts with class_id and resolution_angstroms for filtering),
                "resolution_filter_threshold": 12.0 (optional, default 12.0 Å - clusters with worse resolution are filtered out)
            }
            
            Or simple format:
            - "folder=/path/to/folder" or just "/path/to/folder"
            
            Returns:
                String describing the results and output file locations
            """
            try:
                # Parse input

                print("input_str", input_str)

                params = _parse_tool_input(input_str)
                
                # Get folder path (required)
                folder = params.get("folder") or params.get("folder_path") or input_str.strip()
                if not folder:
                    return "❌ Error: 'folder' parameter is required"
                
                
                
                folder_path = Path(folder).resolve()

                print("folder", folder, "folder_path", folder_path)

                if not folder_path.exists():
                    return f"❌ Error: Folder does not exist: {folder_path}"
                if not folder_path.is_dir():
                    return f"❌ Error: Path is not a directory: {folder_path}"
                
                # Find compare_all_densities.py script
                # First try local copy in alignment_tools folder, then project root, then current directory
                if compare_script:
                    compare_script_path = Path(compare_script).resolve()
                else:
                    # Try local copy in alignment_tools folder first
                    local_script = Path(__file__).parent / "compare_all_densities.py"
                    if local_script.exists():
                        compare_script_path = local_script.resolve()
                    else:
                        # Fallback to project root
                        script_dir = Path(__file__).parent.parent.parent.parent
                        compare_script_path = script_dir / "compare_all_densities.py"
                        if not compare_script_path.exists():
                            # Try current directory
                            compare_script_path = Path("compare_all_densities.py")
                            if not compare_script_path.exists():
                                return f"❌ Error: compare_all_densities.py not found. Please specify compare_script parameter."
                
                # Build command
                cmd = [sys.executable, str(compare_script_path), str(folder_path)]
                
                # Add optional parameters
                if params.get("output_dir"):
                    cmd.extend(["--output_dir", str(params["output_dir"])])
                
                # Auto-detect align_script if not provided
                detected_align_script = None
                if params.get("align_script"):
                    detected_align_script = Path(params["align_script"]).resolve()
                elif align_script:
                    detected_align_script = Path(align_script).resolve()
                else:
                    # Auto-detect: try same directory as compare_all_densities.py first (alignment_tools folder)
                    align_script_candidate = compare_script_path.parent / "align_and_compare_maps.py"
                    if align_script_candidate.exists():
                        detected_align_script = align_script_candidate.resolve()
                    else:
                        # Fallback: try project root
                        project_root = Path(__file__).parent.parent.parent.parent
                        align_script_candidate = project_root / "align_and_compare_maps.py"
                        if align_script_candidate.exists():
                            detected_align_script = align_script_candidate.resolve()
                
                if detected_align_script:
                    cmd.extend(["--align_script", str(detected_align_script)])
                
                if params.get("voxel_size"):
                    cmd.extend(["--voxel_size", str(params["voxel_size"])])
                elif default_voxel_size:
                    cmd.extend(["--voxel_size", str(default_voxel_size)])
                
                # Only pass contour levels if explicitly provided
                # If not provided, the underlying script will calculate RMS automatically
                if params.get("source_contour_level") is not None:
                    cmd.extend(["--source_contour_level", str(params["source_contour_level"])])
                
                if params.get("target_contour_level") is not None:
                    cmd.extend(["--target_contour_level", str(params["target_contour_level"])])
                
                if params.get("alg_type"):
                    cmd.extend(["--alg_type", params["alg_type"]])
                elif default_alg_type:
                    cmd.extend(["--alg_type", default_alg_type])
                
                if params.get("docker_container"):
                    cmd.extend(["--docker_container", params["docker_container"]])
                
                if params.get("docker_mount_prefix"):
                    cmd.extend(["--docker_mount_prefix", params["docker_mount_prefix"]])
                
                if params.get("transform_map_script"):
                    cmd.extend(["--transform_map_script", str(params["transform_map_script"])])
                
                if params.get("fitmap_script"):
                    cmd.extend(["--fitmap_script", str(params["fitmap_script"])])
                
                if params.get("cal_fsc_script"):
                    cmd.extend(["--cal_fsc_script", str(params["cal_fsc_script"])])
                
                if params.get("chimerax_cmd"):
                    cmd.extend(["--chimerax_cmd", params["chimerax_cmd"]])
                
                if params.get("eman2_conda_env"):
                    cmd.extend(["--eman2_conda_env", params["eman2_conda_env"]])
                
                if params.get("resolution_threshold"):
                    cmd.extend(["--resolution_threshold", str(params["resolution_threshold"])])
                
                if params.get("n_clusters"):
                    cmd.extend(["--n_clusters", str(params["n_clusters"])])
                
                if params.get("cluster_method"):
                    cmd.extend(["--cluster_method", params["cluster_method"]])
                
                if params.get("keep_work_dir"):
                    cmd.append("--keep_work_dir")
                
                if params.get("no_rms_threshold"):
                    cmd.append("--no_rms_threshold")
                
                # Run the script with timeout to prevent hanging
                # The comparison can take a while, so use a reasonable timeout (1 hour)
                timeout_seconds = 3600
                
                # Use subprocess.run with capture_output for better performance
                # Run from alignment_tools directory to ensure scripts find each other
                script_working_dir = compare_script_path.parent
                
                try:
                    # Use unbuffered output to ensure we capture all print statements
                    env = os.environ.copy()
                    env['PYTHONUNBUFFERED'] = '1'
                    
                    # Run the script and capture output
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        cwd=str(script_working_dir),  # Run from alignment_tools folder
                        timeout=timeout_seconds,
                        env=env,  # Pass environment with PYTHONUNBUFFERED
                    )
                    
                    # Combine stdout and stderr - show full output
                    output = result.stdout + result.stderr
                    
                    # Debug: Log if output is empty (this shouldn't happen)
                    if not output.strip():
                        output = f"[Warning: Script produced no output. Return code: {result.returncode}]\n"
                        
                except subprocess.TimeoutExpired:
                    error_msg = f"❌ Error: compare_all_densities.py timed out after {timeout_seconds} seconds"
                    return error_msg
                except Exception as e:
                    error_msg = f"❌ Error running compare_all_densities.py: {str(e)}"
                    return error_msg
                
                # Extract output directory from output or use default
                output_dir = params.get("output_dir")
                if not output_dir:
                    output_dir = folder_path / "comparison_results"
                else:
                    output_dir = Path(output_dir)
                
                # Check if key output files were created (even if return code is non-zero)
                # The script might have warnings but still produce valid results
                key_files = [
                    "resolution_matrix.csv",
                    "normalized_matrix.csv",
                    "comparison_summary.txt"
                ]
                files_exist = all((output_dir / f).exists() for f in key_files)
                
                if result.returncode != 0 and not files_exist:
                    return f"❌ Error running compare_all_densities.py (exit code {result.returncode}):\n{output}"
                
                # Parse clustering results if available
                cluster_info = ""
                cluster_file = output_dir / "clusters.txt"
                cluster_csv_file = output_dir / "clusters.csv"
                num_clusters = 0
                cluster_groups = {}
                
                # Extract pairwise comparison resolutions from output for LLM analysis
                pairwise_results = []
                import re
                # Pattern: "[1/3] Comparing map1 vs map2... Done - Resolution: X.XX Å"
                resolution_pattern = r'\[(\d+)/(\d+)\]\s+Comparing\s+([^\s]+)\s+vs\s+([^\s]+)\.\.\.\s+Done\s+-\s+Resolution:\s+([\d.]+)\s+Å'
                matches = re.findall(resolution_pattern, output)
                for match in matches:
                    idx, total, map1, map2, resolution = match
                    pairwise_results.append({
                        'map1': map1,
                        'map2': map2,
                        'resolution': float(resolution)
                    })
                
                if cluster_csv_file.exists():
                    # Read clustering results from CSV
                    import csv
                    clusters = {}
                    with open(cluster_csv_file, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            map_name = row.get('map_name', '')
                            cluster_id = row.get('cluster_id', '-1')
                            clusters[map_name] = int(cluster_id)
                    
                    # Count clusters and group maps
                    if clusters:
                        cluster_groups = {}
                        for map_name, cluster_id in clusters.items():
                            if cluster_id not in cluster_groups:
                                cluster_groups[cluster_id] = []
                            cluster_groups[cluster_id].append(map_name)
                        
                        num_clusters = len(cluster_groups)
                        cluster_info = f"\n\nClustering Results:\n"
                        cluster_info += f"  Number of clusters (groups): {num_clusters}\n"
                        for cluster_id in sorted(cluster_groups.keys()):
                            maps = cluster_groups[cluster_id]
                            cluster_info += f"  Cluster {cluster_id}: {len(maps)} map(s) - {', '.join(maps)}\n"
                
                # Process class resolutions and filter clusters if provided
                class_resolutions_param = params.get("class_resolutions")
                resolution_filter_threshold = params.get("resolution_filter_threshold", 12.0)
                filtered_cluster_groups = cluster_groups.copy()
                cluster_resolutions = {}
                
                if class_resolutions_param and cluster_groups:
                    # Parse class resolutions (can be JSON string or list)
                    if isinstance(class_resolutions_param, str):
                        try:
                            import json
                            class_resolutions_data = json.loads(class_resolutions_param)
                        except:
                            class_resolutions_data = class_resolutions_param
                    else:
                        class_resolutions_data = class_resolutions_param
                    
                    # Convert to list if it's a dict with 'classes' key
                    if isinstance(class_resolutions_data, dict) and 'classes' in class_resolutions_data:
                        class_resolutions_list = class_resolutions_data['classes']
                    elif isinstance(class_resolutions_data, list):
                        class_resolutions_list = class_resolutions_data
                    else:
                        class_resolutions_list = []
                    
                    # Create mapping from class_id to resolution
                    class_id_to_resolution = {}
                    for class_data in class_resolutions_list:
                        if isinstance(class_data, dict):
                            class_id = class_data.get('class_id')
                            resolution = class_data.get('resolution_angstroms')
                            if class_id is not None and resolution is not None:
                                class_id_to_resolution[int(class_id)] = float(resolution)
                    
                    # Map map names to class IDs and find best resolution per cluster
                    # Map names are like "J785_class_00_00042_volume.mrc" -> extract "00" -> class_id 0
                    import re
                    filtered_cluster_groups = {}
                    cluster_resolutions = {}
                    
                    for cluster_id, maps in cluster_groups.items():
                        cluster_resolution_list = []
                        for map_name in maps:
                            # Extract class number from map name (e.g., "class_00" -> 0)
                            class_match = re.search(r'class_(\d+)', map_name)
                            if class_match:
                                class_id = int(class_match.group(1))
                                if class_id in class_id_to_resolution:
                                    cluster_resolution_list.append(class_id_to_resolution[class_id])
                        
                        if cluster_resolution_list:
                            # Use best (lowest) resolution in cluster to represent quality
                            best_resolution = min(cluster_resolution_list)
                            cluster_resolutions[cluster_id] = best_resolution
                            
                            # Filter out clusters with resolution worse than threshold
                            if best_resolution <= resolution_filter_threshold:
                                filtered_cluster_groups[cluster_id] = maps
                
                # Start with a prominent summary section for easy LLM parsing
                # This ensures key information is immediately visible
                result_msg = f"{'='*80}\n"
                result_msg += f"DENSITY COMPARISON SUMMARY (KEY RESULTS)\n"
                result_msg += f"{'='*80}\n"
                
                # Add clustering results with resolutions and filtering FIRST
                if cluster_info:
                    result_msg += cluster_info
                    
                    # Add resolution information for each cluster
                    if cluster_resolutions:
                        result_msg += "\nCluster Resolutions (best resolution in each cluster):\n"
                        for cluster_id in sorted(cluster_groups.keys()):
                            if cluster_id in cluster_resolutions:
                                resolution = cluster_resolutions[cluster_id]
                                status = "✓ KEPT" if resolution <= resolution_filter_threshold else "✗ FILTERED (resolution > 12.0 Å)"
                                result_msg += f"  Cluster {cluster_id}: {resolution:.2f} Å - {status}\n"
                        
                        # Show filtered results summary
                        num_filtered = len(cluster_groups) - len(filtered_cluster_groups)
                        num_total = len(cluster_groups)
                        num_kept = len(filtered_cluster_groups)
                        
                        result_msg += f"\n📊 FILTERING SUMMARY:\n"
                        result_msg += f"  Total clusters found: {num_total}\n"
                        result_msg += f"  Clusters KEPT (resolution ≤ {resolution_filter_threshold} Å): {num_kept}\n"
                        result_msg += f"  Clusters FILTERED OUT (resolution > {resolution_filter_threshold} Å): {num_filtered}\n"
                        
                        if num_filtered > 0:
                            # Show which clusters were filtered
                            filtered_cluster_ids = [cid for cid in cluster_groups.keys() if cid not in filtered_cluster_groups]
                            result_msg += f"\n  Filtered clusters: {', '.join([f'Cluster {cid}' for cid in sorted(filtered_cluster_ids)])}\n"
                            
                            # Show which clusters remain
                            if filtered_cluster_groups:
                                result_msg += f"\n  Remaining clusters (for refinement):\n"
                                for cluster_id in sorted(filtered_cluster_groups.keys()):
                                    maps = filtered_cluster_groups[cluster_id]
                                    resolution = cluster_resolutions.get(cluster_id, 0.0)
                                    # Extract class IDs from map names
                                    class_ids = []
                                    for map_name in maps:
                                        class_match = re.search(r'class_(\d+)', map_name)
                                        if class_match:
                                            class_ids.append(int(class_match.group(1)))
                                    class_ids_str = ', '.join([f'Class {cid}' for cid in sorted(class_ids)])
                                    result_msg += f"    Cluster {cluster_id}: {class_ids_str} (resolution: {resolution:.2f} Å)\n"
                        else:
                            result_msg += f"\n  All clusters passed resolution filter (≤ {resolution_filter_threshold} Å)\n"
                
                # Add pairwise comparison results
                if pairwise_results:
                    result_msg += f"\nPairwise Comparison Results:\n"
                    for pair in pairwise_results:
                        result_msg += f"  {pair['map1']} vs {pair['map2']}: {pair['resolution']:.2f} Å\n"
                
                result_msg += f"\n{'='*80}\n"
                result_msg += f"FULL SCRIPT OUTPUT (for detailed progress)\n"
                result_msg += f"{'='*80}\n"
                
                # Then add the full script output for detailed progress
                if output.strip():
                    if result.returncode != 0:
                        result_msg += f"⚠️ Script warnings/errors (exit code {result.returncode}):\n{output}\n\n"
                    else:
                        # Put full script output after summary
                        result_msg += f"{output}\n\n"
                else:
                    # If no output captured, indicate this
                    result_msg += f"[Note: Script completed but produced no stdout/stderr output. Return code: {result.returncode}]\n\n"
                
                # Add final summary section (duplicate key info at end for visibility)
                result_msg += f"\n{'='*80}\n"
                result_msg += f"SUMMARY (repeated for visibility)\n"
                result_msg += f"{'='*80}\n"
                result_msg += f"✅ Successfully compared all density maps in {folder_path}\n"
                
                # Add pairwise comparison results (factual data for LLM to report)
                if pairwise_results:
                    result_msg += "Pairwise Comparison Results:\n"
                    for pair in pairwise_results:
                        result_msg += f"  {pair['map1']} vs {pair['map2']}: {pair['resolution']:.2f} Å\n"
                    result_msg += "\n"
                
                # Add output directory info
                result_msg += f"\nOutput directory: {output_dir}\n"
                result_msg += f"Results saved to: resolution_matrix.csv, clusters.csv, comparison_summary.txt\n"
                
                return result_msg
                
            except Exception as e:
                return f"❌ Error: {str(e)}\n{type(e).__name__}"
        
        return Tool(
            name="compare_all_densities",
            description=(
                "Compare all density maps (*_volume.mrc files) in a folder and create a resolution relationship matrix. "
                "This tool performs pairwise comparisons using align_and_compare_maps.py and creates: "
                "1) A resolution matrix showing best resolution between each pair, "
                "2) A normalized matrix (2*pixel/Resolution), "
                "3) Graph clustering results to group similar maps. "
                "Required parameter: 'folder' (path to folder containing *_volume.mrc files). "
                "Optional parameters: output_dir, voxel_size, alg_type (global/local), "
                "source_contour_level, target_contour_level, docker_container, docker_mount_prefix, "
                "resolution_threshold, n_clusters, cluster_method (spectral/agglomerative/louvain), "
                "keep_work_dir, no_rms_threshold. "
                "Returns paths to output files including CSV matrices, summary, and clustering results."
            ),
            func=_compare_all_densities_tool
        )


def _parse_tool_input(input_str: str) -> Dict[str, Any]:
    """
    Parse tool input string into parameters.
    
    Supports multiple formats:
    - JSON format: {"key": "value"}
    - Key=value pairs: key=value,key2=value2
    - Simple path: /path/to/folder
    """
    import shlex
    
    input_str = input_str.strip()
    
    # Case 1: JSON format
    if input_str.startswith("{") and input_str.endswith("}"):
        try:
            return json.loads(input_str)
        except json.JSONDecodeError:
            pass
    
    # Case 2: Simple path (no special characters that suggest key=value)
    if not ("=" in input_str or ":" in input_str):
        return {"folder": input_str}
    
    # Case 3: Key/value pairs
    params = {}
    normalized = input_str.replace("\n", " ").strip()
    if normalized:
        # Try splitting by comma first, then by space
        parts = []
        if "," in normalized:
            parts = [p.strip() for p in normalized.split(",")]
        else:
            # Use shlex to handle quoted values
            try:
                parts = shlex.split(normalized)
            except ValueError:
                parts = normalized.split()
        
        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                
                # Convert boolean strings
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                # Convert numeric strings
                elif value.replace(".", "").replace("-", "").isdigit():
                    try:
                        if "." in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass
                # Convert null/None
                elif value.lower() in ("null", "none"):
                    value = None
                
                params[key] = value
            else:
                # If no =, treat as folder path
                if "folder" not in params:
                    params["folder"] = part
    
    return params

