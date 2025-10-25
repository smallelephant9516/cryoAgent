#!/usr/bin/env python3
"""
Demonstration script showing how to use the new backend execution feature
with the existing CryoAgent workflow system.
"""

import sys
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cryoagent.config.config_loader import ConfigLoader
from cryoagent.tools.relion_tools import RELIONTools


def demo_backend_workflow():
    """Demonstrate backend execution with a realistic workflow scenario."""
    print("🚀 RELION Backend Workflow Demonstration")
    print("=" * 60)
    
    try:
        # Load configuration
        print("📋 Loading configuration...")
        config_loader = ConfigLoader(config_path="configs/master_config.json")
        config = config_loader.load_config()
        
        # Initialize RELION tools with backend execution
        print("🔧 Initializing RELION tools with backend execution...")
        relion_tools = RELIONTools(config.relion, config_loader)
        
        print(f"✅ Backend execution enabled: {relion_tools._backend_enabled}")
        print(f"✅ Max concurrent jobs: {relion_tools._max_concurrent_jobs}")
        print(f"✅ Default timeout: {relion_tools._backend_timeout}s")
        
        # Simulate a realistic workflow scenario
        print("\n🎬 Simulating CryoEM Preprocessing Workflow with Backend Execution")
        print("-" * 70)
        
        # Step 1: Validate input movies (this is what was happening in your terminal)
        print("\n📁 Step 1: Validating input movies")
        movies_path = "/home/daoyi/cryoEM_dataset/10025_subset/agent_test/*.tif"
        
        validation_result = relion_tools.validate_inputs("movie_files", movies_path)
        print(f"   Validation result: {validation_result}")
        
        # Step 2: Import movies with backend execution
        print("\n📥 Step 2: Import movies with backend execution")
        try:
            import_job = relion_tools.import_movies(
                movies_path=movies_path,
                output_dir="Import/job001",
                optics_group_name="opticsGroup1",
                angpix=0.6575,
                voltage=300.0,
                cs=2.7,
                use_backend=True,  # Enable backend execution
                timeout=1800  # 30 minutes timeout
            )
            
            print(f"   ✅ Import job started: {import_job['job_id']}")
            print(f"   📊 Process ID: {import_job.get('process_id', 'N/A')}")
            print(f"   📁 Output directory: {import_job['output_dir']}")
            
            # Monitor the import job
            print("\n   ⏱️  Monitoring import job...")
            for i in range(6):  # Monitor for 30 seconds
                time.sleep(5)
                status = relion_tools.get_backend_job_status(import_job['job_id'])
                print(f"   📊 Status check {i+1}: {status['status']}")
                
                if status['status'] in ['completed', 'failed', 'timeout']:
                    print(f"   ✅ Import job finished with status: {status['status']}")
                    if status['status'] == 'completed':
                        print("   🎉 Movies imported successfully!")
                    break
            
        except Exception as e:
            print(f"   ❌ Import job failed: {e}")
            # Continue with demonstration even if import fails
        
        # Step 3: Motion correction with backend execution
        print("\n🎞️  Step 3: Motion correction with backend execution")
        try:
            motion_job = relion_tools.motion_correction(
                input_star="Import/job001/movies.star",
                output_dir="MotionCorr/job002",
                use_backend=True,  # Enable backend execution
                timeout=3600  # 1 hour timeout
            )
            
            print(f"   ✅ Motion correction job started: {motion_job['job_id']}")
            print(f"   📊 Process ID: {motion_job.get('process_id', 'N/A')}")
            
            # Monitor the motion correction job
            print("\n   ⏱️  Monitoring motion correction job...")
            for i in range(4):  # Monitor for 20 seconds
                time.sleep(5)
                status = relion_tools.get_backend_job_status(motion_job['job_id'])
                print(f"   📊 Status check {i+1}: {status['status']}")
                
                if status['status'] in ['completed', 'failed', 'timeout']:
                    print(f"   ✅ Motion correction job finished with status: {status['status']}")
                    break
            
        except Exception as e:
            print(f"   ❌ Motion correction job failed: {e}")
        
        # Step 4: CTF estimation with backend execution
        print("\n🔬 Step 4: CTF estimation with backend execution")
        try:
            ctf_job = relion_tools.ctf_estimation(
                input_star="MotionCorr/job002/corrected_micrographs.star",
                output_dir="CtfFind/job003",
                use_backend=True,  # Enable backend execution
                timeout=1800  # 30 minutes timeout
            )
            
            print(f"   ✅ CTF estimation job started: {ctf_job['job_id']}")
            print(f"   📊 Process ID: {ctf_job.get('process_id', 'N/A')}")
            
            # Monitor the CTF estimation job
            print("\n   ⏱️  Monitoring CTF estimation job...")
            for i in range(4):  # Monitor for 20 seconds
                time.sleep(5)
                status = relion_tools.get_backend_job_status(ctf_job['job_id'])
                print(f"   📊 Status check {i+1}: {status['status']}")
                
                if status['status'] in ['completed', 'failed', 'timeout']:
                    print(f"   ✅ CTF estimation job finished with status: {status['status']}")
                    break
            
        except Exception as e:
            print(f"   ❌ CTF estimation job failed: {e}")
        
        # Step 5: Show all backend jobs
        print("\n📋 Step 5: Backend job summary")
        all_jobs = relion_tools.list_backend_jobs()
        print(f"   📊 Total backend jobs: {len(all_jobs)}")
        
        for job in all_jobs:
            print(f"   🔹 {job['job_id']}: {job['status']}")
            if job.get('process_id'):
                print(f"      Process ID: {job['process_id']}")
            if job.get('started_at'):
                print(f"      Started: {time.ctime(job['started_at'])}")
        
        # Step 6: Demonstrate job management
        print("\n🛠️  Step 6: Job management demonstration")
        
        # Show how to stop specific jobs
        if all_jobs:
            print("   🛑 Stopping all backend jobs...")
            stopped_count = relion_tools.stop_all_backend_jobs()
            print(f"   ✅ Stopped {stopped_count} backend jobs")
        
        print("\n🎉 Backend workflow demonstration completed!")
        print("\n💡 Key Benefits of Backend Execution:")
        print("   • Non-blocking execution of long-running RELION jobs")
        print("   • Concurrent processing of multiple jobs")
        print("   • Better resource management and monitoring")
        print("   • Seamless integration with existing workflow")
        print("   • Configurable timeouts and resource limits")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function."""
    demo_backend_workflow()


if __name__ == "__main__":
    main()
