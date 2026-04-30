#!/usr/bin/env python3
"""
Simple test script for creating a regroup job with 2 classes.
This script demonstrates how to:
1. Connect to CryoSPARC
2. Inspect the regroup job type to find the exact parameter name
3. Create a regroup job with the correct parameter
4. Queue the job
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cryosparc.tools import CryoSPARC
from cryoagent.config.config_loader import ConfigLoader

# 1. Load configuration from master_config.json and initialize CryoSPARC client
config_loader = ConfigLoader("configs/master_config.json")
config = config_loader.load_config()

# Build connection parameters (license is required by CryoSPARC client)
connection_params = {
    "host": config.cryosparc.host,
    "base_port": config.cryosparc.base_port,
    "email": config.cryosparc.username,
    "password": config.cryosparc.password
}

# Add license if available (required by CryoSPARC client)
license_id = config.cryosparc.license_id
if license_id and license_id.strip() and license_id != "your-cryosparc-license-id-here":
    connection_params["license"] = license_id
else:
    raise ValueError(
        "CryoSPARC license ID is required but not set. "
        "Please set the LICENSE_ID environment variable or update configs/master_config.json"
    )

cs = CryoSPARC(**connection_params)

# Configuration
project_id = config.workflow.project_uid
workspace_id = config.workflow.workspace_uid

# 2. Inspect the job type to find the exact parameter name
job_type = 'reference_motion_correction' 

job = cs.find_job("P10", "J77")

# 3. Inspect the 'params_base' dictionary
# This contains the default key-value pairs for the job parameters.
print("--- Available Parameters (Keys & Defaults) ---")
print(job.doc)

# 4. Validate input slot connections for validation jobs
print("\n--- Validating Input Slot Connections ---")
doc = job.doc

# Check if this is a validation job with volume input group
if doc.get('job_type') == 'validation':
    input_slot_groups = doc.get('input_slot_groups', [])
    
    for group in input_slot_groups:
        if group.get('name') == 'volume':
            connections = group.get('connections', [])
            slots = group.get('slots', [])
            
            # Check if we have map_half_A and map_half_B slots
            slot_names = [slot.get('name') for slot in slots]
            if 'map_half_A' in slot_names and 'map_half_B' in slot_names:
                print(f"Found volume input group with map_half_A and map_half_B slots")
                print(f"Number of connections: {len(connections)}")
                
                # Check for incorrect connections
                if len(connections) > 1:
                    print("❌ ERROR: Multiple connection entries found (should be 1)")
                    print("   This indicates incorrect wiring - both slots may be connected to the same result")
                
                # Check each connection
                for i, conn in enumerate(connections):
                    print(f"\nConnection {i+1}:")
                    conn_slots = conn.get('slots', [])
                    print(f"  Number of slot connections: {len(conn_slots)}")
                    
                    # Track which slots are connected to which results
                    slot_to_result = {}
                    for slot_conn in conn_slots:
                        slot_name = slot_conn.get('slot_name')
                        result_name = slot_conn.get('result_name')
                        job_uid = slot_conn.get('job_uid')
                        slot_to_result[slot_name] = {
                            'result_name': result_name,
                            'job_uid': job_uid
                        }
                        print(f"    {slot_name} → {job_uid}:{result_name}")
                    
                    # Validate: each slot should connect to a different result
                    if 'map_half_A' in slot_to_result and 'map_half_B' in slot_to_result:
                        result_A = slot_to_result['map_half_A']['result_name']
                        result_B = slot_to_result['map_half_B']['result_name']
                        job_A = slot_to_result['map_half_A']['job_uid']
                        job_B = slot_to_result['map_half_B']['job_uid']
                        
                        # Check if both slots connect to the same result
                        if result_A == result_B:
                            print(f"  ❌ ERROR: Both slots connect to the same result: {result_A}")
                            if job_A == job_B:
                                print(f"  ❌ ERROR: Both slots also connect to the same job: {job_A}")
                        else:
                            print(f"  ✓ OK: Slots connect to different results ({result_A} vs {result_B})")
                            if job_A != job_B:
                                print(f"  ✓ OK: Slots connect to different jobs ({job_A} vs {job_B})")
                            else:
                                print(f"  ⚠ WARNING: Both slots connect to the same job ({job_A}) but different results")


# 4. Queue the job
# job.queue()
