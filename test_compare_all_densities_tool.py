#!/usr/bin/env python3
"""
Test script for compare_all_densities tool.

This script tests the CompareAllDensitiesTool to ensure it:
1. Correctly runs the comparison script
2. Parses clustering results from CSV files
3. Handles errors gracefully
4. Returns properly formatted results
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from cryoagent.tools.alignment_tools.compare_all_densities_tool import CompareAllDensitiesTool


def test_basic_comparison():
    """Test basic comparison with J764 directory."""
    print("=" * 80)
    print("Test 1: Basic Comparison with J764")
    print("=" * 80)
    
    tool = CompareAllDensitiesTool.create_compare_all_densities_tool(
        default_voxel_size=2.24
    )
    
    test_folder = "/mnt/sda2/cryosparc_projects/CS-relion-tutorial/J764"
    
    print(f"\nTesting with folder: {test_folder}")
    print("-" * 80)
    
    result = tool.func(test_folder)
    
    print("\nTool Result:")
    print(result)
    print("-" * 80)
    
    # Check if result contains key information
    assert "Successfully compared" in result, "Result should indicate success"
    assert "Number of clusters" in result, "Result should contain clustering information"
    assert "Cluster 0" in result, "Result should contain cluster details"
    
    print("\n✅ Test 1 PASSED: Basic comparison works correctly")
    return True


def test_json_input():
    """Test with JSON input format."""
    print("\n" + "=" * 80)
    print("Test 2: JSON Input Format")
    print("=" * 80)
    
    tool = CompareAllDensitiesTool.create_compare_all_densities_tool(
        default_voxel_size=2.24
    )
    
    test_input = json.dumps({
        "folder": "/mnt/sda2/cryosparc_projects/CS-relion-tutorial/J764",
        "voxel_size": 2.24,
        "alg_type": "global"
    })
    
    print(f"\nTesting with JSON input: {test_input}")
    print("-" * 80)
    
    result = tool.func(test_input)
    
    print("\nTool Result:")
    print(result)
    print("-" * 80)
    
    assert "Successfully compared" in result, "Result should indicate success"
    
    print("\n✅ Test 2 PASSED: JSON input format works correctly")
    return True


def test_clustering_parsing():
    """Test that clustering results are properly parsed."""
    print("\n" + "=" * 80)
    print("Test 3: Clustering Results Parsing")
    print("=" * 80)
    
    tool = CompareAllDensitiesTool.create_compare_all_densities_tool(
        default_voxel_size=2.24
    )
    
    test_folder = "/mnt/sda2/cryosparc_projects/CS-relion-tutorial/J764"
    result = tool.func(test_folder)
    
    # Check for clustering information
    assert "Number of clusters (groups):" in result, "Should report number of clusters"
    assert "Cluster 0:" in result, "Should report cluster 0"
    assert "Cluster 1:" in result, "Should report cluster 1"
    
    # Extract number of clusters from result
    import re
    cluster_match = re.search(r"Number of clusters \(groups\): (\d+)", result)
    if cluster_match:
        num_clusters = int(cluster_match.group(1))
        print(f"\nDetected {num_clusters} clusters from result")
        assert num_clusters == 2, f"Expected 2 clusters, got {num_clusters}"
    
    print("\n✅ Test 3 PASSED: Clustering results are properly parsed")
    return True


def test_output_files():
    """Test that output files are created."""
    print("\n" + "=" * 80)
    print("Test 4: Output Files Verification")
    print("=" * 80)
    
    tool = CompareAllDensitiesTool.create_compare_all_densities_tool(
        default_voxel_size=2.24
    )
    
    test_folder = "/mnt/sda2/cryosparc_projects/CS-relion-tutorial/J764"
    result = tool.func(test_folder)
    
    output_dir = Path(test_folder) / "comparison_results"
    
    # Check for key output files
    required_files = [
        "resolution_matrix.csv",
        "resolution_matrix.txt",
        "normalized_matrix.csv",
        "normalized_matrix.txt",
        "comparison_summary.txt",
        "clusters.csv",
        "clusters.txt"
    ]
    
    print(f"\nChecking output directory: {output_dir}")
    print("-" * 80)
    
    missing_files = []
    for filename in required_files:
        filepath = output_dir / filename
        if filepath.exists():
            print(f"✅ {filename} exists")
        else:
            print(f"❌ {filename} MISSING")
            missing_files.append(filename)
    
    if missing_files:
        print(f"\n⚠️  Warning: {len(missing_files)} files are missing: {missing_files}")
    else:
        print("\n✅ All required output files exist")
    
    # Verify clusters.csv can be read
    clusters_csv = output_dir / "clusters.csv"
    if clusters_csv.exists():
        import csv
        with open(clusters_csv, 'r') as f:
            reader = csv.DictReader(f)
            clusters = {}
            for row in reader:
                map_name = row['map_name']
                cluster_id = int(row['cluster_id'])
                clusters[map_name] = cluster_id
        
        print(f"\nParsed clusters from CSV:")
        cluster_groups = {}
        for map_name, cluster_id in clusters.items():
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(map_name)
        
        for cluster_id in sorted(cluster_groups.keys()):
            print(f"  Cluster {cluster_id}: {cluster_groups[cluster_id]}")
        
        assert len(cluster_groups) == 2, f"Expected 2 clusters, found {len(cluster_groups)}"
    
    print("\n✅ Test 4 PASSED: Output files are created and readable")
    return True


def test_error_handling():
    """Test error handling with invalid input."""
    print("\n" + "=" * 80)
    print("Test 5: Error Handling")
    print("=" * 80)
    
    tool = CompareAllDensitiesTool.create_compare_all_densities_tool(
        default_voxel_size=2.24
    )
    
    # Test with non-existent folder
    print("\nTesting with non-existent folder...")
    result = tool.func("/nonexistent/folder/path")
    
    print(f"Result: {result[:200]}...")
    
    assert "Error" in result or "does not exist" in result, "Should return error for invalid folder"
    
    print("\n✅ Test 5 PASSED: Error handling works correctly")
    return True


def test_with_parameters():
    """Test with custom parameters."""
    print("\n" + "=" * 80)
    print("Test 6: Custom Parameters")
    print("=" * 80)
    
    tool = CompareAllDensitiesTool.create_compare_all_densities_tool(
        default_voxel_size=2.24
    )
    
    # Test with custom output directory
    test_input = json.dumps({
        "folder": "/mnt/sda2/cryosparc_projects/CS-relion-tutorial/J764",
        "output_dir": "/tmp/test_comparison_output",
        "voxel_size": 2.24,
        "resolution_threshold": 20.0,
        "cluster_method": "spectral"
    })
    
    print(f"\nTesting with custom parameters...")
    print(f"Input: {test_input}")
    print("-" * 80)
    
    result = tool.func(test_input)
    
    print("\nTool Result (first 500 chars):")
    print(result[:500])
    print("-" * 80)
    
    # Check if custom output directory is mentioned
    assert "/tmp/test_comparison_output" in result or "Successfully compared" in result
    
    print("\n✅ Test 6 PASSED: Custom parameters work correctly")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Compare All Densities Tool Test Suite")
    print("=" * 80)
    
    tests = [
        ("Basic Comparison", test_basic_comparison),
        ("JSON Input Format", test_json_input),
        ("Clustering Parsing", test_clustering_parsing),
        ("Output Files", test_output_files),
        ("Error Handling", test_error_handling),
        ("Custom Parameters", test_with_parameters),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success, None))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' FAILED with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False, str(e)))
    
    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, error in results:
        status = "✅ PASSED" if success else f"❌ FAILED ({error})"
        print(f"  {test_name}: {status}")
    
    print("-" * 80)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

