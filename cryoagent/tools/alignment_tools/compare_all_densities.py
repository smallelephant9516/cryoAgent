#!/usr/bin/env python3
"""
Compare all density maps in a folder and create a resolution relationship matrix.

This script finds all *_volume.mrc files in a given folder, compares all pairs
using align_and_compare_maps.py, and creates a triangular matrix showing the
best resolution between each pair.

Additionally, the script:
1. Estimates pixel size from MRC file headers
2. Calculates normalized matrix: (2*pixel)/Resolution for each edge
3. Builds a graph from the similarity matrix
4. Clusters the graph into groups

Usage:
    python compare_all_densities.py <folder_path> [options]

Output:
    - Prints progress for each comparison
    - Creates a resolution matrix file (CSV format)
    - Creates a normalized matrix file (2*pixel/Resolution)
    - Creates a summary text file
    - Creates clustering results (if clustering succeeds)
"""

import argparse
import csv
import itertools
import re
import subprocess
import sys
from pathlib import Path
import numpy as np

try:
    import mrcfile
except ImportError:
    mrcfile = None

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    from sklearn.cluster import SpectralClustering
    from sklearn.cluster import AgglomerativeClustering
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


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


def estimate_pixel_size(map_path):
    """
    Estimate pixel size (voxel size) from MRC file header.
    
    Args:
        map_path: Path to MRC file
    
    Returns:
        float: Pixel size in Angstroms
    """
    if mrcfile is None:
        raise ImportError(
            "mrcfile is required to estimate pixel size. "
            "Install it with: pip install mrcfile"
        )
    
    try:
        with mrcfile.open(str(map_path), mode='r', permissive=True) as mrc:
            # Get voxel size (usually stored as a tuple or object with x, y, z)
            voxel_size = mrc.voxel_size
            if hasattr(voxel_size, 'x'):
                # mrcfile returns a VoxelSize object
                pixel_size = float(voxel_size.x)
            elif isinstance(voxel_size, (list, tuple, np.ndarray)):
                # If it's a sequence, take the first element
                pixel_size = float(voxel_size[0])
            else:
                # If it's a scalar
                pixel_size = float(voxel_size)
            
            # Validate pixel size is reasonable (typically 0.5-10 Angstroms)
            if pixel_size <= 0 or pixel_size > 100:
                raise ValueError(f"Invalid pixel size from MRC header: {pixel_size} Å")
            
            return pixel_size
    except Exception as e:
        raise ValueError(f"Could not read pixel size from {map_path}: {e}")


def estimate_average_pixel_size(volume_maps):
    """
    Estimate average pixel size across all maps.
    
    If maps have different pixel sizes, we use the average.
    This is used for the (2*pixel)/Resolution calculation.
    
    Args:
        volume_maps: List of Path objects for volume maps
    
    Returns:
        float: Average pixel size in Angstroms
    """
    pixel_sizes = []
    for map_path in volume_maps:
        try:
            pixel_size = estimate_pixel_size(map_path)
            pixel_sizes.append(pixel_size)
        except Exception as e:
            print(f"Warning: Could not estimate pixel size for {map_path.name}: {e}", file=sys.stderr)
    
    if not pixel_sizes:
        raise ValueError("Could not estimate pixel size from any map files")
    
    avg_pixel_size = np.mean(pixel_sizes)
    if len(set(pixel_sizes)) > 1:
        print(f"Warning: Maps have different pixel sizes: {pixel_sizes}")
        print(f"Using average pixel size: {avg_pixel_size:.4f} Å", file=sys.stderr)
    else:
        print(f"Estimated pixel size: {avg_pixel_size:.4f} Å")
    
    return avg_pixel_size


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
    if kwargs.get("no_rms_threshold"):
        cmd.append("--no_rms_threshold")
    
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


def calculate_normalized_matrix(volume_maps, results, pixel_size, resolution_threshold=None):
    """
    Calculate normalized matrix: (2*pixel)/Resolution for each edge.
    
    This metric represents the Nyquist frequency ratio, which is useful
    for comparing alignments across different pixel sizes.
    
    Edges with resolution worse than resolution_threshold are excluded (set to None).
    
    Args:
        volume_maps: List of Path objects for volume maps
        results: Dictionary mapping (map1, map2) -> resolution
        pixel_size: Pixel size in Angstroms
        resolution_threshold: Maximum resolution in Angstroms to include (None = no filter)
    
    Returns:
        dict: Dictionary mapping (map1, map2) -> normalized_value (None if filtered out)
    """
    normalized_results = {}
    filtered_count = 0
    for (map1, map2), resolution in results.items():
        if resolution is not None and resolution > 0:
            # Filter by resolution threshold
            if resolution_threshold is not None and resolution > resolution_threshold:
                normalized_results[(map1, map2)] = None
                filtered_count += 1
            else:
                normalized_value = (2 * pixel_size) / resolution
                normalized_results[(map1, map2)] = normalized_value
        else:
            normalized_results[(map1, map2)] = None
    
    if resolution_threshold is not None and filtered_count > 0:
        print(f"Filtered out {filtered_count} edges with resolution > {resolution_threshold} Å")
    
    return normalized_results


def build_graph(volume_maps, normalized_results, resolution_threshold=None):
    """
    Build a graph from the normalized similarity matrix.
    
    Nodes represent maps, edges represent comparisons with weights = (2*pixel)/Resolution.
    Higher values indicate better alignment (higher resolution relative to pixel size).
    
    Only edges with valid (non-None) weights are included in the graph.
    
    Args:
        volume_maps: List of Path objects for volume maps
        normalized_results: Dictionary mapping (map1, map2) -> normalized_value
        resolution_threshold: Maximum resolution threshold used (for logging purposes)
    
    Returns:
        networkx.Graph or dict: Graph object (or adjacency dict if networkx unavailable)
    """
    n_maps = len(volume_maps)
    map_names = [m.name for m in volume_maps]
    
    if HAS_NETWORKX:
        G = nx.Graph()
        # Add nodes
        for map_name in map_names:
            G.add_node(map_name)
        
        # Add edges with weights
        for (map1, map2), weight in normalized_results.items():
            if weight is not None:
                name1 = map1.name
                name2 = map2.name
                G.add_edge(name1, name2, weight=weight)
        
        return G
    else:
        # Fallback: return adjacency dictionary
        adjacency = {}
        for i, name1 in enumerate(map_names):
            adjacency[name1] = {}
            for j, name2 in enumerate(map_names):
                if i == j:
                    adjacency[name1][name2] = 1.0  # Self-connection
                else:
                    # Find the edge in normalized_results
                    map1 = volume_maps[i]
                    map2 = volume_maps[j]
                    # Create key in consistent order (use str comparison for determinism)
                    key = (map1, map2) if str(map1) <= str(map2) else (map2, map1)
                    weight = normalized_results.get(key)
                    adjacency[name1][name2] = weight if weight is not None else 0.0
        
        return adjacency


def cluster_graph(graph, volume_maps, n_clusters=None, method='spectral'):
    """
    Cluster the graph into groups.
    
    Args:
        graph: NetworkX graph or adjacency dictionary
        volume_maps: List of Path objects for volume maps (for ordering)
        n_clusters: Number of clusters (None for auto-detection)
        method: Clustering method ('spectral', 'agglomerative', 'louvain')
    
    Returns:
        dict: Dictionary mapping map_name -> cluster_id
    """
    map_names = [m.name for m in volume_maps]
    
    if isinstance(graph, dict):
        # Convert adjacency dict to numpy matrix
        n = len(map_names)
        adjacency_matrix = np.zeros((n, n))
        for i, name1 in enumerate(map_names):
            for j, name2 in enumerate(map_names):
                adjacency_matrix[i, j] = graph[name1].get(name2, 0.0)
        
        # Make symmetric and handle missing values
        adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)
        
    elif HAS_NETWORKX:
        # Convert NetworkX graph to adjacency matrix
        n = len(map_names)
        adjacency_matrix = np.zeros((n, n))
        for i, name1 in enumerate(map_names):
            for j, name2 in enumerate(map_names):
                if graph.has_edge(name1, name2):
                    adjacency_matrix[i, j] = graph[name1][name2].get('weight', 1.0)
        
        # Make symmetric
        adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)
        
        # Try Louvain community detection if networkx is available and method is 'louvain'
        if method == 'louvain':
            try:
                try:
                    import community.community_louvain as community_louvain
                except ImportError:
                    # Try alternative import name
                    import community_louvain
                partition = community_louvain.best_partition(graph)
                # Convert to cluster_id mapping
                clusters = {}
                for name in map_names:
                    clusters[name] = partition.get(name, -1)
                return clusters
            except (ImportError, AttributeError):
                print("Warning: python-louvain not available, falling back to spectral clustering", file=sys.stderr)
                method = 'spectral'
    else:
        raise ValueError("Cannot cluster graph: networkx not available and graph is not a dict")
    
    # Check if we have valid edges
    max_weight = adjacency_matrix.max()
    if max_weight <= 0:
        raise ValueError("No valid edges found in graph - cannot perform clustering")
    
    # Count connected components first (needed for auto-detection and special handling)
    n_connected_components = 1
    
    if HAS_NETWORKX and isinstance(graph, nx.Graph):
        # Use NetworkX to find connected components
        connected_components = list(nx.connected_components(graph))
        n_connected_components = len(connected_components)
    else:
        # Count connected components from adjacency matrix using simple DFS
        # Create a binary adjacency matrix (0 = no edge, 1 = has edge)
        binary_adj = (adjacency_matrix > 0).astype(int)
        # Remove self-connections for component detection
        np.fill_diagonal(binary_adj, 0)
        
        # Simple DFS to find connected components
        visited = np.zeros(n, dtype=bool)
        components_list = []
        
        def dfs(start):
            component = []
            stack = [start]
            visited[start] = True
            while stack:
                node = stack.pop()
                component.append(node)
                for neighbor in range(n):
                    if binary_adj[node, neighbor] > 0 and not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            return component
        
        for i in range(n):
            if not visited[i]:
                component = dfs(i)
                components_list.append(component)
        
        n_connected_components = len(components_list)
        connected_components = [set(map_names[i] for i in comp) for comp in components_list]
    
    # Auto-detect number of clusters if not specified
    if n_clusters is None:
        # Use heuristic: sqrt(n/2) or at least number of connected components
        heuristic_clusters = max(2, int(np.sqrt(len(map_names) / 2)))
        # Need at least as many clusters as connected components
        # (can't have fewer clusters than disconnected components)
        n_clusters = max(n_connected_components, heuristic_clusters)
        # But don't exceed number of maps
        n_clusters = min(n_clusters, len(map_names))
        
        print(f"Detected {n_connected_components} connected component(s) in graph")
        print(f"Auto-detected number of clusters: {n_clusters} (at least {n_connected_components} required for disconnected components)")
    
    # Ensure n_clusters is at least the number of connected components
    # (this is a constraint: we can't have fewer clusters than disconnected components)
    if n_clusters < n_connected_components:
        print(f"Warning: n_clusters ({n_clusters}) is less than number of connected components ({n_connected_components})")
        print(f"Adjusting n_clusters to {n_connected_components}")
        n_clusters = n_connected_components
    
    # Apply clustering method
    if method == 'spectral' and HAS_SKLEARN:
        # Spectral clustering on similarity matrix
        # Convert to similarity (normalize)
        similarity_matrix = adjacency_matrix / (max_weight + 1e-10)
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42
        )
        labels = clustering.fit_predict(similarity_matrix)
        
    elif method == 'agglomerative' and HAS_SKLEARN:
        # Agglomerative clustering
        # Convert to distance matrix (inverse of similarity)
        distance_matrix = max_weight - adjacency_matrix
        # Ensure non-negative distances
        distance_matrix = np.maximum(distance_matrix, 0.0)
        # Add small value to diagonal to avoid zero distances
        np.fill_diagonal(distance_matrix, 0.0)
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            linkage='average'
        )
        labels = clustering.fit_predict(distance_matrix)
        
    else:
        raise ValueError(f"Clustering method '{method}' not available or sklearn not installed")
    
    # Create mapping from map name to cluster ID
    clusters = {}
    for i, name in enumerate(map_names):
        clusters[name] = int(labels[i])
    
    return clusters


def save_matrix(volume_maps, results, output_dir, pixel_size=None, normalized_results=None, clusters=None, resolution_threshold=None):
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
    
    # Save as CSV (lower triangle only)
    csv_file = output_dir / "resolution_matrix.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header row
        writer.writerow([''] + map_names)
        # Data rows (lower triangle only)
        for i, map_name in enumerate(map_names):
            row = [map_name] + [
                "0.00" if i == j else (f"{matrix[i][j]:.2f}" if (matrix[i][j] is not None and i > j) else "")
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
        f.write(f"Total comparisons: {len(results)}\n")
        if resolution_threshold is not None:
            f.write(f"Resolution threshold: {resolution_threshold} Å (edges with worse resolution excluded from graph)\n")
        f.write("\n")
        
        # List all pairs with resolutions
        f.write("Pairwise Resolutions:\n")
        f.write("-" * 80 + "\n")
        filtered_count = 0
        for (map1, map2), resolution in sorted(results.items()):
            if resolution is not None:
                filtered = (resolution_threshold is not None and resolution > resolution_threshold)
                status = " (FILTERED)" if filtered else ""
                f.write(f"{map1.name} vs {map2.name}: {resolution:.2f} Å{status}\n")
                if filtered:
                    filtered_count += 1
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
            if resolution_threshold is not None and filtered_count > 0:
                included_count = len(valid_resolutions) - filtered_count
                f.write(f"  Edges included in graph: {included_count}/{len(valid_resolutions)} (filtered out {filtered_count} edges with resolution > {resolution_threshold} Å)\n")
    
    # Save normalized matrix if available
    if normalized_results is not None and pixel_size is not None:
        normalized_csv_file = output_dir / "normalized_matrix.csv"
        normalized_txt_file = output_dir / "normalized_matrix.txt"
        
        # Create normalized matrix
        normalized_matrix = [[None] * n_maps for _ in range(n_maps)]
        for (map1, map2), value in normalized_results.items():
            i = volume_maps.index(map1)
            j = volume_maps.index(map2)
            normalized_matrix[i][j] = value
            normalized_matrix[j][i] = value  # Symmetric
        
        # Set diagonal
        for i in range(n_maps):
            normalized_matrix[i][i] = 1.0  # Self-similarity
        
        # Save normalized CSV
        with open(normalized_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([''] + map_names)
            for i, map_name in enumerate(map_names):
                row = [map_name] + [
                    "1.00" if i == j else (f"{normalized_matrix[i][j]:.4f}" if (normalized_matrix[i][j] is not None and i > j) else "")
                    for j in range(n_maps)
                ]
                writer.writerow(row)
        
        # Save normalized text file
        with open(normalized_txt_file, 'w') as f:
            f.write("Normalized Matrix (2*pixel/Resolution)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Pixel size: {pixel_size:.4f} Å\n")
            f.write("Higher values indicate better alignment (higher resolution relative to pixel size)\n\n")
            
            max_name_len = max(len(name) for name in map_names)
            col_width = max(max_name_len, 12)
            
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
                    if normalized_matrix[i][j] is not None:
                        if i == j:
                            f.write(f"{'1.00':<{col_width}}  ")
                        else:
                            f.write(f"{normalized_matrix[i][j]:.4f}{'':<{col_width-7}}  ")
                    else:
                        f.write(f"{'N/A':<{col_width}}  ")
                f.write("\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("Note: Normalized value = (2 * pixel_size) / resolution\n")
            f.write("Diagonal values are 1.0 (same map)\n")
        
        print(f"  - Normalized CSV matrix: {normalized_csv_file}")
        print(f"  - Normalized text matrix: {normalized_txt_file}")
    
    # Save clustering results if available
    if clusters is not None:
        cluster_file = output_dir / "clusters.txt"
        with open(cluster_file, 'w') as f:
            f.write("Graph Clustering Results\n")
            f.write("=" * 80 + "\n\n")
            
            # Group maps by cluster
            cluster_groups = {}
            for map_name, cluster_id in clusters.items():
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(map_name)
            
            # Write clusters
            for cluster_id in sorted(cluster_groups.keys()):
                f.write(f"Cluster {cluster_id} ({len(cluster_groups[cluster_id])} maps):\n")
                for map_name in sorted(cluster_groups[cluster_id]):
                    f.write(f"  - {map_name}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write(f"Total clusters: {len(cluster_groups)}\n")
            f.write(f"Total maps: {len(clusters)}\n")
        
        # Also save as CSV for easy import
        cluster_csv_file = output_dir / "clusters.csv"
        with open(cluster_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['map_name', 'cluster_id'])
            for map_name in map_names:
                writer.writerow([map_name, clusters.get(map_name, -1)])
        
        print(f"  - Clustering results: {cluster_file}")
        print(f"  - Clustering CSV: {cluster_csv_file}")
    
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
    parser.add_argument(
        "--no_rms_threshold",
        action="store_true",
        help="Disable automatic RMS calculation and use manual contour levels"
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=None,
        help="Number of clusters for graph clustering (default: auto-detect)"
    )
    parser.add_argument(
        "--cluster_method",
        type=str,
        default="spectral",
        choices=["spectral", "agglomerative", "louvain"],
        help="Clustering method (default: spectral)"
    )
    parser.add_argument(
        "--resolution_threshold",
        type=float,
        default=20.0,
        help="Maximum resolution in Angstroms to include in graph (edges with worse resolution are excluded) (default: 20.0)"
    )
    
    args = parser.parse_args()
    
    # Find volume maps
    folder_path = Path(args.folder).resolve()
    volume_maps = find_volume_maps(folder_path)
    
    # Find align script
    if args.align_script:
        align_script = Path(args.align_script).resolve()
    else:
        # Auto-detect: look in same directory as this script first
        script_dir = Path(__file__).parent
        align_script = script_dir / "align_and_compare_maps.py"
        if not align_script.exists():
            # Fallback: look in project root (parent of alignment_tools)
            project_root = script_dir.parent.parent.parent
            align_script = project_root / "align_and_compare_maps.py"
            if not align_script.exists():
                # Fallback: look in current directory
                align_script = Path("align_and_compare_maps.py")
                if not align_script.exists():
                    raise FileNotFoundError(
                        f"align_and_compare_maps.py not found. "
                        f"Please specify --align_script or ensure it's in {script_dir}, {project_root}, or current directory"
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
        "no_rms_threshold": args.no_rms_threshold,
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
    
    # Estimate pixel size from maps
    print("=" * 80)
    print("Estimating pixel size from MRC files...")
    print("=" * 80)
    try:
        pixel_size = estimate_average_pixel_size(volume_maps)
    except Exception as e:
        print(f"Warning: Could not estimate pixel size: {e}", file=sys.stderr)
        print(f"Using provided voxel_size as pixel size: {kwargs.get('voxel_size', 5.0)}", file=sys.stderr)
        pixel_size = kwargs.get('voxel_size', 5.0)
    
    # Run pairwise comparisons
    print("\n" + "=" * 80)
    print("Running pairwise comparisons...")
    print("=" * 80)
    results = create_resolution_matrix(volume_maps, align_script, output_dir, **kwargs)
    
    # Calculate normalized matrix
    print("\n" + "=" * 80)
    print("Calculating normalized matrix (2*pixel/Resolution)...")
    print(f"Resolution threshold: {args.resolution_threshold} Å (edges with worse resolution will be excluded)")
    print("=" * 80)
    normalized_results = calculate_normalized_matrix(volume_maps, results, pixel_size, 
                                                     resolution_threshold=args.resolution_threshold)
    
    # Build graph
    print("\n" + "=" * 80)
    print("Building graph from similarity matrix...")
    print("=" * 80)
    graph = build_graph(volume_maps, normalized_results, resolution_threshold=args.resolution_threshold)
    if HAS_NETWORKX:
        print(f"Graph nodes: {len(graph.nodes())}")
        print(f"Graph edges: {len(graph.edges())}")
    else:
        print("Graph built (using adjacency dictionary)")
    
    # Cluster graph
    print("\n" + "=" * 80)
    print("Clustering graph...")
    print("=" * 80)
    try:
        clusters = cluster_graph(graph, volume_maps, n_clusters=args.n_clusters, method=args.cluster_method)
        print(f"Clustering completed: {len(set(clusters.values()))} clusters found")
        # Print cluster summary
        cluster_counts = {}
        for map_name, cluster_id in clusters.items():
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        for cluster_id, count in sorted(cluster_counts.items()):
            print(f"  Cluster {cluster_id}: {count} maps")
    except Exception as e:
        print(f"Warning: Clustering failed: {e}", file=sys.stderr)
        print("Continuing without clustering results...", file=sys.stderr)
        clusters = None
    
    # Save matrix and results
    print("\n" + "=" * 80)
    print("Creating resolution matrix and saving results...")
    print("=" * 80)
    save_matrix(volume_maps, results, output_dir, pixel_size=pixel_size, 
                normalized_results=normalized_results, clusters=clusters,
                resolution_threshold=args.resolution_threshold)
    
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

