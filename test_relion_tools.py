#!/usr/bin/env python3
"""Test script for RELION tools with real dataset and conda environment isolation."""

import os
import sys
import json
import time
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

def test_conda_environment_setup(relion_tools):
    """Test conda environment configuration and setup."""
    print("\n🐍 Testing conda environment setup...")
    
    try:
        # Configure conda environment
        relion_tools.configure_conda_environment(
            env_name="relion-5.0",
            conda_executable="conda",
            use_conda=True
        )
        
        # Check conda environment status
        status = relion_tools.get_conda_environment_status()
        print(f"   Conda enabled: {status['conda_enabled']}")
        print(f"   Conda available: {status['conda_available']}")
        print(f"   Environment exists: {status['environment_exists']}")
        print(f"   Environment name: {status['environment_name']}")
        
        if status['conda_available']:
            print(f"   Conda version: {status.get('conda_version', 'Unknown')}")
        
        if status['error']:
            print(f"   ⚠️  Warning: {status['error']}")
            return False
        
        # Enable backend execution
        relion_tools.enable_backend_execution(True)
        print("✅ Conda environment setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Conda environment setup failed: {e}")
        return False

def test_conda_backend_execution(relion_tools):
    """Test conda backend execution with a simple command."""
    print("\n🚀 Testing conda backend execution...")
    
    try:
        # Test with a simple command
        test_command = ["python", "--version"]
        job_id = f"conda_test_{int(time.time())}"
        output_dir = "CondaTest"
        
        print(f"   Running test command: {' '.join(test_command)}")
        print(f"   Job ID: {job_id}")
        
        # Run the command in backend mode
        job_info = relion_tools.run_relion_backend(
            command=test_command,
            job_id=job_id,
            output_dir=output_dir,
            timeout=60,
            check_interval=5
        )
        
        print(f"   ✅ Backend job started successfully!")
        print(f"   Process ID: {job_info['process_id']}")
        print(f"   Conda environment: {job_info.get('conda_env', 'None')}")
        print(f"   Full command: {job_info['command']}")
        
        # Monitor the job
        print(f"   ⏳ Monitoring job completion...")
        max_wait = 30  # 30 seconds max wait
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status = relion_tools.get_backend_job_status(job_id)
            
            if status['status'] in ['completed', 'failed', 'timeout']:
                break
                
            time.sleep(2)
        
        # Get final status
        final_status = relion_tools.get_backend_job_status(job_id)
        print(f"   Final status: {final_status['status']}")
        
        if final_status['status'] == 'completed':
            print(f"   ✅ Conda backend execution test completed successfully!")
            if 'stdout' in final_status and final_status['stdout']:
                print(f"   Output: {final_status['stdout'].strip()}")
            return True
        else:
            print(f"   ❌ Conda backend execution test failed")
            if 'stderr' in final_status and final_status['stderr']:
                print(f"   Error: {final_status['stderr'][:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Conda backend execution test failed: {e}")
        return False

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

def test_import_movies(relion_tools, use_conda_backend=False):
    """Test movie import functionality."""
    print(f"\n🎬 Testing movie import{' (with conda backend)' if use_conda_backend else ''}...")
    
    # Load microscope config
    with open("configs/microscope_config.json", 'r') as f:
        microscope_config = json.load(f)
    
    params = microscope_config["microscope_parameters"]
    
    try:
        # Test import movies - use absolute path from config
        result = relion_tools.import_movies(
            movies_path=params["movies_path"],  # Use absolute path from config
            output_dir="test_import_conda" if use_conda_backend else "test_import",
            optics_group_name="opticsGroup1",
            angpix=params["pixel_size"],
            voltage=params["voltage"],
            cs=params["cs_mm"],
            q0=0.1,
            beamtilt_x=0.0,
            beamtilt_y=0.0,
            output_file="movies.star",
            wait_for_completion=True,
            timeout=600,
            use_backend=use_conda_backend  # Enable conda backend if requested
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

def test_motion_correction(relion_tools, import_result, use_conda_backend=False):
    """Test motion correction functionality."""
    print(f"\n🎯 Testing motion correction{' (with conda backend)' if use_conda_backend else ''}...")
    
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
            output_dir="test_motioncorr_conda" if use_conda_backend else "test_motioncorr",
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
            timeout=1800,  # 30 minutes timeout
            use_backend=use_conda_backend  # Enable conda backend if requested
        )
        
        print("✅ Motion correction completed successfully!")
        print(f"Output directory: {result['output_dir']}")
        return result
        
    except Exception as e:
        print(f"❌ Motion correction failed: {e}")
        return None

def test_ctf_estimation(relion_tools, motioncorr_result, use_conda_backend=False):
    """Test CTF estimation functionality."""
    print(f"\n🔬 Testing CTF estimation{' (with conda backend)' if use_conda_backend else ''}...")
    
    if not motioncorr_result:
        print("❌ No motion correction result available for CTF estimation")
        return None
    
    try:
        # Test CTF estimation
        result = relion_tools.ctf_estimation(
            input_star=os.path.join(motioncorr_result["output_dir"], "corrected_micrographs.star"),
            output_dir="test_ctffind_conda" if use_conda_backend else "test_ctffind",
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
            timeout=1800,  # 30 minutes timeout
            use_backend=use_conda_backend  # Enable conda backend if requested
        )
        
        print("✅ CTF estimation completed successfully!")
        print(f"Output directory: {result['output_dir']}")
        return result
        
    except Exception as e:
        print(f"❌ CTF estimation failed: {e}")
        return None

def run_conda_backend_tests(relion_tools):
    """Run tests with conda backend execution."""
    print("\n" + "=" * 60)
    print("🐍 TESTING WITH CONDA BACKEND EXECUTION")
    print("=" * 60)
    
    # Test conda environment setup
    if not test_conda_environment_setup(relion_tools):
        print("❌ Conda environment setup failed. Skipping conda backend tests.")
        return False
    
    # Test conda backend execution with simple command
    if not test_conda_backend_execution(relion_tools):
        print("❌ Conda backend execution test failed. Skipping RELION conda tests.")
        return False
    
    # Test RELION operations with conda backend
    print("\n🎬 Testing RELION operations with conda backend...")
    
    # Test import movies with conda backend
    import_result = test_import_movies(relion_tools, use_conda_backend=True)
    if not import_result:
        print("❌ Movie import with conda backend failed.")
        return False
    
    # Test motion correction with conda backend
    motioncorr_result = test_motion_correction(relion_tools, import_result, use_conda_backend=True)
    if not motioncorr_result:
        print("❌ Motion correction with conda backend failed.")
        return False
    
    # Test CTF estimation with conda backend
    ctf_result = test_ctf_estimation(relion_tools, motioncorr_result, use_conda_backend=True)
    if not ctf_result:
        print("❌ CTF estimation with conda backend failed.")
        return False
    
    print("\n✅ All conda backend tests completed successfully!")
    print("Summary of conda backend results:")
    print(f"✅ Movie import (conda): {import_result['output_dir']}")
    print(f"✅ Motion correction (conda): {motioncorr_result['output_dir']}")
    print(f"✅ CTF estimation (conda): {ctf_result['output_dir']}")
    
    return True

def run_regular_tests(relion_tools):
    """Run tests with regular execution."""
    print("\n" + "=" * 60)
    print("🔧 TESTING WITH REGULAR EXECUTION")
    print("=" * 60)
    
    # Test import movies
    import_result = test_import_movies(relion_tools, use_conda_backend=False)
    if not import_result:
        print("❌ Movie import failed.")
        return False
    
    # Test motion correction
    motioncorr_result = test_motion_correction(relion_tools, import_result, use_conda_backend=False)
    if not motioncorr_result:
        print("❌ Motion correction failed.")
        return False
    
    # Test CTF estimation
    ctf_result = test_ctf_estimation(relion_tools, motioncorr_result, use_conda_backend=False)
    if not ctf_result:
        print("❌ CTF estimation failed.")
        return False
    
    print("\n✅ All regular tests completed successfully!")
    print("Summary of regular results:")
    print(f"✅ Movie import (regular): {import_result['output_dir']}")
    print(f"✅ Motion correction (regular): {motioncorr_result['output_dir']}")
    print(f"✅ CTF estimation (regular): {ctf_result['output_dir']}")
    
    return True

def main():
    """Main test function."""
    print("🧪 Testing RELION Tools with Real Dataset and Conda Environment")
    print("=" * 70)
    
    # Test 1: RELION setup
    relion_tools = test_relion_setup()
    if not relion_tools:
        print("❌ RELION setup failed. Exiting.")
        return
    
    # Test 2: Dataset access
    if not test_dataset_access():
        print("❌ Dataset access failed. Exiting.")
        return
    
    # Test 3: Regular execution tests
    regular_success = run_regular_tests(relion_tools)
    
    # Test 4: Conda backend execution tests
    conda_success = run_conda_backend_tests(relion_tools)
    
    # Final summary
    print("\n" + "=" * 70)
    print("🎉 FINAL TEST SUMMARY")
    print("=" * 70)
    
    if regular_success:
        print("✅ Regular execution tests: PASSED")
    else:
        print("❌ Regular execution tests: FAILED")
    
    if conda_success:
        print("✅ Conda backend execution tests: PASSED")
    else:
        print("❌ Conda backend execution tests: FAILED")
    
    if regular_success and conda_success:
        print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("✅ RELION tools work correctly with both regular and conda backend execution")
        print("✅ Conda environment isolation is working properly")
    elif regular_success:
        print("\n⚠️  Regular execution works, but conda backend has issues")
    elif conda_success:
        print("\n⚠️  Conda backend works, but regular execution has issues")
    else:
        print("\n❌ Both regular and conda backend execution have issues")

if __name__ == "__main__":
    main()
