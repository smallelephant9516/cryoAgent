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
job_type = 'regroup_3D_new' 

job = cs.find_job("P2", "J267")

# 3. Inspect the 'params_base' dictionary
# This contains the default key-value pairs for the job parameters.
print("--- Available Parameters (Keys & Defaults) ---")
print(job.doc)


# 4. Queue the job
# job.queue()
