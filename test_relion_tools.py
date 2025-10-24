#!/usr/bin/env python3
"""Test script for RELION tools with real dataset."""

import os
import sys
import json
from pathlib import Path

# Add the project root to the Python path
#sys.path.insert(0, '/home/daoyi/Github/cryoagent')

from cryoagent.tools.relion_tools import RELIONTools
from cryoagent.config.config_loader import ConfigLoader

def test_relion_setup():
    """Test RELION installation and configuration."""
    print("🔧 Testing RELION setup...")
    
    # Load configuration
    config_loader = ConfigLoader(
        config_path="configs/relion/preprocessing_config.json",
        master_config_path="configs/master_config.json"
    )
    
    # Get RELION settings
    relion_settings = config_loader.get_relion_settings()
    print(f"RELION executable: {relion_settings.relion_exe}")
    print(f"RELION directory: {relion_settings.relion_dir}")
    print(f"Continue job: {relion_settings.continue_job}")
    
    # Initialize RELION tools
    try:
        relion_tools = RELIONTools(relion_settings, config_loader)
        print("✅ RELION tools initialized successfully!")
        return relion_tools
    except Exception as e:
        print(f"❌ Failed to initialize RELION tools: {e}")
        return None

def test_dataset_access():
    """Test access to the real dataset."""
    print("\n📁 Testing dataset access...")
    
    # Load microscope config
    with open("configs/microscope_config.json", 'r') as f:
        microscope_config = json.load(f)
    
    movies_path = microscope_config["microscope_parameters"]["movies_path"]
    gain_ref_path = microscope_config["microscope_parameters"]["gain_ref_path"]
    
    print(f"Movies path: {movies_path}")
    print(f"Gain reference path: {gain_ref_path}")
    
    # Check if movies exist
    import glob
    movie_files = glob.glob(movies_path)
    print(f"Found {len(movie_files)} movie files")
    
    if len(movie_files) == 0:
        print("❌ No movie files found!")
        return False
    
    # Check if gain reference exists
    if os.path.exists(gain_ref_path):
        print(f"✅ Gain reference found: {gain_ref_path}")
    else:
        print(f"❌ Gain reference not found: {gain_ref_path}")
        return False
    
    # Show first few movie files
    print("First few movie files:")
    for i, movie in enumerate(movie_files[:3]):
        print(f"  {i+1}. {movie}")
    
    return True

def test_import_movies(relion_tools):
    """Test movie import functionality."""
    print("\n🎬 Testing movie import...")
    
    # Load microscope config
    with open("configs/microscope_config.json", 'r') as f:
        microscope_config = json.load(f)
    
    params = microscope_config["microscope_parameters"]
    
    try:
        # Test import movies - use absolute path from config
        result = relion_tools.import_movies(
            movies_path=params["movies_path"],  # Use absolute path from config
            output_dir="test_import",
            optics_group_name="opticsGroup1",
            angpix=params["pixel_size"],
            voltage=params["voltage"],
            cs=params["cs_mm"],
            q0=0.1,
            beamtilt_x=0.0,
            beamtilt_y=0.0,
            output_file="movies.star",
            wait_for_completion=True,
            timeout=600
        )
        
        print("✅ Movie import completed successfully!")
        print(f"Output directory: {result['output_dir']}")
        print(f"Output file: {result['output_file']}")
        
        # Check if the file actually exists and update path if needed
        if not os.path.exists(result['output_file']):
            # Try the parent directory
            parent_dir = os.path.dirname(result['output_dir'])
            expected_file = os.path.join(parent_dir, f"test_importmovies.star")
            if os.path.exists(expected_file):
                result['output_file'] = expected_file
                print(f"Found movies.star at: {expected_file}")
        
        return result
        
    except Exception as e:
        print(f"❌ Movie import failed: {e}")
        return None

def test_motion_correction(relion_tools, import_result):
    """Test motion correction functionality."""
    print("\n🎯 Testing motion correction...")
    
    if not import_result:
        print("❌ No import result available for motion correction")
        return None
    
    # Load microscope config for gain reference
    with open("configs/microscope_config.json", 'r') as f:
        microscope_config = json.load(f)
    
    # Use absolute path for gain reference from config
    gain_ref_path = microscope_config["microscope_parameters"]["gain_ref_path"]
    
    try:
        # Test motion correction
        result = relion_tools.motion_correction(
            input_star=import_result["output_file"],
            output_dir="test_motioncorr",
            first_frame_sum=1,
            last_frame_sum=-1,
            use_own=True,
            num_threads=4,  # Use fewer threads for testing
            bin_factor=1,
            bfactor=150.0,
            dose_per_frame=1.39,
            preexposure=0.0,
            patch_x=1,
            patch_y=1,
            eer_grouping=32,
            gainref=gain_ref_path,
            gain_rot=0,
            gain_flip=0,
            dose_weighting=True,
            grouping_for_ps=3,
            wait_for_completion=True,
            timeout=1800  # 30 minutes timeout
        )
        
        print("✅ Motion correction completed successfully!")
        print(f"Output directory: {result['output_dir']}")
        return result
        
    except Exception as e:
        print(f"❌ Motion correction failed: {e}")
        return None

def test_ctf_estimation(relion_tools, motioncorr_result):
    """Test CTF estimation functionality."""
    print("\n🔬 Testing CTF estimation...")
    
    if not motioncorr_result:
        print("❌ No motion correction result available for CTF estimation")
        return None
    
    try:
        # Test CTF estimation
        result = relion_tools.ctf_estimation(
            input_star=os.path.join(motioncorr_result["output_dir"], "corrected_micrographs.star"),
            output_dir="test_ctffind",
            box_size=512,
            res_min=30.0,
            res_max=5.0,
            df_min=5000.0,
            df_max=50000.0,
            fstep=500.0,
            dast=100.0,
            ctffind_exe="/home/daoyi/tools/ctffind/ctffind_4_1_14/ctffind",
            ctf_win=-1,
            is_ctffind4=True,
            fast_search=True,
            only_do_unfinished=True,
            wait_for_completion=True,
            timeout=1800  # 30 minutes timeout
        )
        
        print("✅ CTF estimation completed successfully!")
        print(f"Output directory: {result['output_dir']}")
        return result
        
    except Exception as e:
        print(f"❌ CTF estimation failed: {e}")
        return None

def main():
    """Main test function."""
    print("🧪 Testing RELION Tools with Real Dataset")
    print("=" * 50)
    
    # Test 1: RELION setup
    relion_tools = test_relion_setup()
    if not relion_tools:
        print("❌ RELION setup failed. Exiting.")
        return
    
    # Test 2: Dataset access
    if not test_dataset_access():
        print("❌ Dataset access failed. Exiting.")
        return
    
    # Test 3: Import movies
    import_result = test_import_movies(relion_tools)
    if not import_result:
        print("❌ Movie import failed. Exiting.")
        return
    
    # Test 4: Motion correction
    motioncorr_result = test_motion_correction(relion_tools, import_result)
    if not motioncorr_result:
        print("❌ Motion correction failed. Exiting.")
        return
    
    # Test 5: CTF estimation
    ctf_result = test_ctf_estimation(relion_tools, motioncorr_result)
    if not ctf_result:
        print("❌ CTF estimation failed. Exiting.")
        return
    
    print("\n🎉 All tests completed successfully!")
    print("=" * 50)
    print("Summary:")
    print(f"✅ Movie import: {import_result['output_dir']}")
    print(f"✅ Motion correction: {motioncorr_result['output_dir']}")
    print(f"✅ CTF estimation: {ctf_result['output_dir']}")

if __name__ == "__main__":
    main()
