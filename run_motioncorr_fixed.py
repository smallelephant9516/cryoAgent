#!/usr/bin/env python3
"""
Fixed script to run RELION motion correction in the background using conda environment
and monitor job completion using the RELIONTools class.
"""

import os
import sys
import time
import subprocess
import threading
import signal
from pathlib import Path

# Add the cryoagent module to the path
sys.path.insert(0, '/home/daoyi/Github/cryoagent')

from cryoagent.tools.relion_tools import RELIONTools
from cryoagent.config.config_loader import RELIONSettings


def run_motioncorr_background():
    """
    Run RELION motion correction command in the background using conda environment.
    """
    # The actual RELION command
    relion_cmd = [
        "conda", "run", "-n", "relion-5.0",
        "bash", "-c",
        "cd /home/daoyi/relion/relion_test2 && "
        "/usr/local/bin/relion_run_motioncorr "
        "--i /home/daoyi/relion/relion_test2/Import/job001/movies.star "
        "--o test_motioncorr/ "
        "--first_frame_sum 1 "
        "--last_frame_sum -1 "
        "--j 4 "
        "--bin_factor 1 "
        "--bfactor 150.0 "
        "--dose_per_frame 1.39 "
        "--preexposure 0.0 "
        "--patch_x 1 "
        "--patch_y 1 "
        "--eer_grouping 32 "
        "--gain_rot 0 "
        "--gain_flip 0 "
        "--pipeline_control test_motioncorr/ "
        "--grouping_for_ps 3 "
        "--use_own "
        "--gainref norm-amibox05-0.mrc "
        "--dose_weighting"
    ]
    
    print("🚀 Starting RELION motion correction in background...")
    print("Command:", " ".join(relion_cmd))
    
    try:
        # Set environment variables to avoid display issues
        env = os.environ.copy()
        env['DISPLAY'] = ''
        env['QT_QPA_PLATFORM'] = 'offscreen'
        env['QT_AUTO_SCREEN_SCALE_FACTOR'] = '0'
        env['QT_SCALE_FACTOR'] = '1'
        
        # Start the process in the background
        process = subprocess.Popen(
            relion_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd="/home/daoyi/relion/relion_test2",
            preexec_fn=os.setsid if os.name != 'nt' else None
        )
        
        print(f"✅ RELION process started with PID: {process.pid}")
        return process
        
    except Exception as e:
        print(f"❌ Failed to start RELION process: {e}")
        return None


def monitor_job_completion(output_dir, check_interval=30, timeout=3600):
    """
    Monitor the job completion using RELIONTools._check_job_completion_files.
    
    Args:
        output_dir: Path to the job output directory
        check_interval: Time between checks in seconds
        timeout: Maximum time to wait in seconds
    """
    print(f"🛰️ Monitoring job completion in: {output_dir}")
    print(f"   Check interval: {check_interval}s, Timeout: {timeout}s")
    
    # Create a minimal RELIONTools instance for monitoring
    class JobMonitor:
        def __init__(self):
            pass
        
        def _check_job_completion_files(self, output_dir: str) -> str:
            """
            Check for RELION job completion files in the output directory.
            
            Args:
                output_dir: Path to the job output directory
                
            Returns:
                Status string: 'completed', 'failed', or 'running'
            """
            if not os.path.exists(output_dir):
                return "unknown"
            
            success_file = os.path.join(output_dir, "RELION_JOB_EXIT_SUCCESS")
            failure_file = os.path.join(output_dir, "RELION_JOB_EXIT_FAILURE")
            
            if os.path.exists(success_file):
                return "completed"
            elif os.path.exists(failure_file):
                return "failed"
            else:
                return "running"
    
    monitor = JobMonitor()
    start_time = time.time()
    last_status = None
    
    while time.time() - start_time < timeout:
        status = monitor._check_job_completion_files(output_dir)
        elapsed = int(time.time() - start_time)
        
        # Only print status if it changed
        if status != last_status:
            print(f"📊 Job status: {status} (elapsed: {elapsed}s)")
            last_status = status
        
        if status == "completed":
            print("✅ Job completed successfully!")
            return True
        elif status == "failed":
            print("❌ Job failed!")
            return False
        elif status == "unknown":
            if elapsed % 60 == 0:  # Print every minute when unknown
                print(f"⚠️  Job directory not found yet, waiting... (elapsed: {elapsed}s)")
        
        time.sleep(check_interval)
    
    print(f"⏰ Job monitoring timed out after {timeout} seconds")
    return False


def monitor_process(process, output_dir, check_interval=30, timeout=3600):
    """
    Monitor both the process and the job completion files.
    
    Args:
        process: The subprocess to monitor
        output_dir: Path to the job output directory
        check_interval: Time between checks in seconds
        timeout: Maximum time to wait in seconds
    """
    print(f"🛰️ Monitoring process (PID: {process.pid}) and job completion...")
    print(f"   Check interval: {check_interval}s, Timeout: {timeout}s")
    
    start_time = time.time()
    last_status = None
    
    # Create job monitor
    class JobMonitor:
        def _check_job_completion_files(self, output_dir: str) -> str:
            if not os.path.exists(output_dir):
                return "unknown"
            
            success_file = os.path.join(output_dir, "RELION_JOB_EXIT_SUCCESS")
            failure_file = os.path.join(output_dir, "RELION_JOB_EXIT_FAILURE")
            
            if os.path.exists(success_file):
                return "completed"
            elif os.path.exists(failure_file):
                return "failed"
            else:
                return "running"
    
    monitor = JobMonitor()
    
    while time.time() - start_time < timeout:
        # Check if process is still running

        if int(time.time() - start_time) % check_interval == 0:
            print(f"📊 Job completion status: {job_status}, elapsed: {int(time.time() - start_time)}s")

        if process.poll() is not None:
            # Process has finished
            stdout, stderr = process.communicate()
            print(f"📊 Process finished with return code: {process.returncode}")
            
            if process.returncode == 0:
                print("✅ Process completed successfully!")
            else:
                print(f"❌ Process failed with return code {process.returncode}")
                print(f"Error output: {stderr}")
            
            # Check job completion files as well
            job_status = monitor._check_job_completion_files(output_dir)
            print(f"📊 Job completion status: {job_status}")
            
            return process.returncode == 0
        
        # Check job completion files
        job_status = monitor._check_job_completion_files(output_dir)
        elapsed = int(time.time() - start_time)
        
        # Only print status if it changed
        if job_status != last_status:
            print(f"📊 Job status: {job_status} (elapsed: {elapsed}s)")
            last_status = job_status
        
        if job_status == "completed":
            print("✅ Job completed successfully!")
            # Terminate the process since job is done
            process.terminate()
            return True
        elif job_status == "failed":
            print("❌ Job failed!")
            process.terminate()
            return False
        
        time.sleep(check_interval)
    
    print(f"⏰ Monitoring timed out after {timeout} seconds")
    process.terminate()
    return False


def main():
    """
    Main function to run motion correction and monitor completion.
    """
    print("=" * 70)
    print("RELION Motion Correction Background Runner (Fixed)")
    print("=" * 70)
    
    # Define paths
    output_dir = "/home/daoyi/relion/relion_test2/test_motioncorr"
    check_interval = 2  # Check every 10 seconds
    timeout = 3600  # 1 hour timeout
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Start the motion correction in the background
    process = run_motioncorr_background()
    
    if process is None:
        print("❌ Failed to start motion correction process")
        return 1
    
    # Wait a moment for the job to start
    print("⏳ Waiting 5 seconds for job to initialize...")
    time.sleep(5)
    
    # Monitor both process and job completion
    success = monitor_process(process, output_dir, check_interval, timeout)
    
    if success:
        print("🎉 Motion correction completed successfully!")
        
        # List output files
        if os.path.exists(output_dir):
            print("\n📋 Output files:")
            files = os.listdir(output_dir)
            if files:
                for file in sorted(files):
                    file_path = os.path.join(output_dir, file)
                    if os.path.isfile(file_path):
                        size = os.path.getsize(file_path)
                        print(f"   {file} ({size:,} bytes)")
            else:
                print("   No files found in output directory")
    else:
        print("💥 Motion correction failed or timed out!")
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
