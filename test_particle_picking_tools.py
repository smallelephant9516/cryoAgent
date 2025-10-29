#!/usr/bin/env python3
"""Test script for particle picking tools."""

import os
import sys
import json
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cryoagent.config.config_loader import ConfigLoader, RELIONSettings
from cryoagent.tools.relion_tools import RELIONTools


def test_particle_picking_tools():
    """Test all particle picking tools."""
    print("🧪 Testing Particle Picking Tools")
    print("=" * 50)
    
    # Load configuration
    config_loader = ConfigLoader(
        config_path="configs/master_config.json",
        master_config_path="configs/master_config.json"
    )
    settings = config_loader.get_relion_settings()
    
    # Initialize RELION tools
    relion_tools = RELIONTools(settings, config_loader)
    
    # Test data paths (you'll need to adjust these)
    test_micrographs_star = "Select/job001/micrographs.star"  # From micrograph selection
    test_particles_star = "Extract/job001/particles.star"   # From particle extraction
    test_optimiser_star = "Class2D/job001/run_optimiser.star"  # From 2D classification
    
    print("\n1. Testing Blob Picker Tool")
    print("-" * 30)
    try:
        # Test blob picker with minimal parameters
        result = relion_tools.blob_picker(
            input_star=test_micrographs_star,
            output_dir="AutoPick",
            particle_diameter=200.0,
            angpix=1.0,
            LoG=True,
            LoG_diam_min=100.0,
            LoG_diam_max=300.0,
            wait_for_completion=False,  # Don't wait for actual completion in test
            timeout=30
        )
        print(f"✅ Blob picker test passed: {result['job_type']}")
        print(f"   Output directory: {result['output_dir']}")
    except Exception as e:
        print(f"❌ Blob picker test failed: {e}")
    
    print("\n2. Testing Particle Extraction Tool")
    print("-" * 30)
    try:
        # Test particle extraction
        result = relion_tools.particle_extraction(
            input_star=test_micrographs_star,
            output_dir="Extract",
            coord_suffix="_autopick.star",
            coord_list = result['output_dir'],
            extract_size=128,
            norm=True,
            wait_for_completion=False,
            timeout=30
        )
        print(f"✅ Particle extraction test passed: {result['job_type']}")
        print(f"   Output directory: {result['output_dir']}")
    except Exception as e:
        print(f"❌ Particle extraction test failed: {e}")
        print(test_micrographs_star)
    
    print("\n3. Testing 2D Classification Tool")
    print("-" * 30)
    try:
        # Test 2D classification
        result = relion_tools.classification_2d(
            input_star=test_particles_star,
            output_dir="Class2D",
            K=10,  # Small number for testing
            iter=5,  # Few iterations for testing
            particle_diameter=200.0,
            angpix=1.0,
            wait_for_completion=False,
            timeout=30
        )
        print(f"✅ 2D classification test passed: {result['job_type']}")
        print(f"   Output directory: {result['output_dir']}")
    except Exception as e:
        print(f"❌ 2D classification test failed: {e}")
    
    print("\n4. Testing Auto 2D Selection Tool")
    print("-" * 30)
    try:
        # Test auto 2D selection
        result = relion_tools.auto_2d_selection(
            input_opt=test_optimiser_star,
            output_dir="Select2",
            min_score=0.5,
            max_score=999.0,
            auto_select=True,
            wait_for_completion=False,
            timeout=30
        )
        print(f"✅ Auto 2D selection test passed: {result['job_type']}")
        print(f"   Output directory: {result['output_dir']}")
    except Exception as e:
        print(f"❌ Auto 2D selection test failed: {e}")
    
    print("\n5. Testing Configuration Loading")
    print("-" * 30)
    try:
        # Load particle picking configuration
        config_path = "configs/relion/particle_picking_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"✅ Configuration loaded successfully")
            print(f"   Workflow: {config['workflow_name']}")
            print(f"   Stages: {len(config['stages'])}")
            print(f"   Steps in stage1: {len(config['stages']['stage1']['steps'])}")
        else:
            print(f"❌ Configuration file not found: {config_path}")
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
    
    print("\n6. Testing Tool Validation")
    print("-" * 30)
    try:
        # Test input validation
        validation_result = relion_tools.validate_inputs("star_file", test_micrographs_star)
        print(f"✅ Input validation test: {validation_result}")
        
        validation_result = relion_tools.validate_inputs("movies", "*.mrc")
        print(f"✅ Movies validation test: {validation_result}")
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Particle picking tools testing completed!")
    print("Note: These are command validation tests. Actual execution requires valid input files.")


if __name__ == "__main__":
    test_particle_picking_tools()
