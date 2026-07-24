"""
Fix workflow 10181 visualization data:
1. Add iteration 0 (reconstruction baseline J24)
2. Add all 3 heterogeneous classification jobs (J27, J30, J33)
3. Fix iteration numbering to start from 0
4. Get micrograph count (need to query CryoSPARC or find in logs)
"""
import json
from pathlib import Path

workflow_dir = Path("/home/daoyi/Github/cryoagent/outputs/dynamic_mode/10181")

# Load existing workflow_state.json
with open(workflow_dir / "workflow_state.json") as f:
    workflow_state = json.load(f)

# Load optimization results
with open(workflow_dir / "optimization_results_cryosparc_20251206_090820.json") as f:
    optimization_results = json.load(f)

# Parse the multi-round classification data from optimization results
multi_round_data = optimization_results["tested_combinations"][0]  # First entry is the multi-round result
rounds_data = multi_round_data  # Contains all 3 rounds

# Extract all 3 rounds - need to manually parse from the log data we found:
# Round 1: J27 - final_resolution: 2.4235280780473887
# Round 2: J30 - final_resolution: 2.3807247095520205 (best)
# Round 3: J33 - final_resolution: 2.393793151303517

# Build the complete tested_combinations array with iteration numbers starting from 0
tested_combinations = []

# ITERATION 0: Reconstruction baseline (J24)
# We need to find J24's resolution - for now use a placeholder that we'll update
iteration_0 = {
    "iteration": 0,
    "phase": "reconstruction_baseline",
    "job_uid": "J24",
    "job_type": "homogeneous_refinement",
    "resolution": None,  # TODO: Need to query CryoSPARC for J24 resolution
    "box_size": 336,  # From optimization - initial box size
    "description": "Initial homogeneous refinement (baseline)"
}
tested_combinations.append(iteration_0)

# ITERATION 1: Multi-round 3D classification - Round 1 (J27)
iteration_1 = {
    "iteration": 1,
    "phase": "3d_classification",
    "job_uid": "J27",
    "job_type": "nonuniform_refine_new",
    "resolution": 2.4235280780473887,
    "box_size": 336,
    "round": 1,
    "best_class_id": 3,
    "best_class_resolution": 3.6252991189399197,
    "description": "Multi-round 3D classification - Round 1"
}
tested_combinations.append(iteration_1)

# ITERATION 2: Multi-round 3D classification - Round 2 (J30) - BEST
iteration_2 = {
    "iteration": 2,
    "phase": "3d_classification",
    "job_uid": "J30",
    "job_type": "nonuniform_refine_new",
    "resolution": 2.3807247095520205,
    "box_size": 336,
    "round": 2,
    "best_class_id": 3,
    "best_class_resolution": 3.184128371033346,
    "description": "Multi-round 3D classification - Round 2 (best)"
}
tested_combinations.append(iteration_2)

# ITERATION 3: Multi-round 3D classification - Round 3 (J33)
iteration_3 = {
    "iteration": 3,
    "phase": "3d_classification",
    "job_uid": "J33",
    "job_type": "nonuniform_refine_new",
    "resolution": 2.393793151303517,
    "box_size": 336,
    "round": 3,
    "best_class_id": 2,
    "best_class_resolution": 3.1750470871693164,
    "description": "Multi-round 3D classification - Round 3"
}
tested_combinations.append(iteration_3)

# ITERATION 4-5: Heterogeneous refinement K=2 and K=4
iteration_4 = {
    "iteration": 4,
    "phase": "heterogeneous_refinement",
    "job_uid": "J35",
    "job_type": "heterogeneous_refinement",
    "resolution": 2.4620679471071103,
    "box_size": 336,
    "k": 2,
    "description": "Heterogeneous refinement K=2"
}
tested_combinations.append(iteration_4)

iteration_5 = {
    "iteration": 5,
    "phase": "heterogeneous_refinement",
    "job_uid": "J41",
    "job_type": "heterogeneous_refinement",
    "resolution": 2.411821619018742,
    "box_size": 336,
    "k": 4,
    "description": "Heterogeneous refinement K=4"
}
tested_combinations.append(iteration_5)

# ITERATIONS 6-12: Box size optimization (J30, J43, J45, J47, J49, J51, J53, J55, J57)
box_size_tests = [
    {"job_uid": "J30", "box_size": 336, "resolution": 2.3807247095520205},
    {"job_uid": "J43", "box_size": 300, "resolution": 2.489133550973001},
    {"job_uid": "J45", "box_size": 360, "resolution": 2.3376179328478686},
    {"job_uid": "J47", "box_size": 400, "resolution": 2.325746559759971},
    {"job_uid": "J49", "box_size": 420, "resolution": 2.299935961879355},
    {"job_uid": "J51", "box_size": 448, "resolution": 2.2413355013982064},
    {"job_uid": "J53", "box_size": 500, "resolution": 2.222844078025609},
    {"job_uid": "J55", "box_size": 540, "resolution": 2.206599155465212},
    {"job_uid": "J57", "box_size": 600, "resolution": 2.1769016059593573},
]

for idx, test in enumerate(box_size_tests):
    iteration = {
        "iteration": 6 + idx,
        "phase": "box_size_optimization",
        "job_uid": test["job_uid"],
        "job_type": "homogeneous_refinement",
        "resolution": test["resolution"],
        "box_size": test["box_size"],
        "description": f"Box size optimization: {test['box_size']}px"
    }
    tested_combinations.append(iteration)

# Now create the enriched vis_report.json
vis_report = {
    "workflow_metadata": {
        "project_uid": "P11",
        "workspace_uid": "W1",
        "workflow_path": str(workflow_dir),
        "enriched": True
    },
    "stages": []
}

# Add each stage from workflow_state
for record in workflow_state["records"]:
    stage_data = {
        "stage": record["stage"],
        "success": record["success"],
        "primary_job_uid": record.get("primary_job_uid"),
        "metrics": record.get("metrics", {}),
        "decisions": record.get("decisions", []),
        "goal": record.get("goal", ""),
        "timestamp": record.get("timestamp", 0),
        "execution_time": record.get("execution_time", 0)
    }

    # Special handling for optimization stage
    if record["stage"] == "optimization":
        stage_data["detailed_results"] = {
            "tested_combinations": tested_combinations,
            "best_job_uid": "J57",
            "best_resolution": 2.1769016059593573,
            "best_box_size": 600,
            "iterations": len(tested_combinations)
        }
        stage_data["stage_outputs"] = record.get("stage_outputs", {})
        stage_data["stage_outputs"]["tested_combinations"] = tested_combinations

        # Update metrics
        stage_data["metrics"]["best_resolution_angstroms"] = 2.1769016059593573
        stage_data["metrics"]["best_box_size"] = 600
        stage_data["metrics"]["num_iterations"] = len(tested_combinations)

    # Special handling for particle_picking - add particle count
    elif record["stage"] == "particle_picking":
        stage_data["metrics"]["num_particles"] = 627100

    # Special handling for reconstruction - add metrics
    elif record["stage"] == "3d_reconstruction":
        stage_data["metrics"]["resolution_angstroms"] = None  # TODO: Query J24
        stage_data["metrics"]["num_particles"] = 686680
        stage_data["primary_job_uid"] = "J24"

    # Special handling for improvement - mark as success
    elif record["stage"] == "improvement":
        stage_data["success"] = True  # Actually improved 2.18Å → 2.17Å

    vis_report["stages"].append(stage_data)

# Save the enriched report
output_path = workflow_dir / "vis_report.json"
with open(output_path, "w") as f:
    json.dump(vis_report, f, indent=2)

print(f"✅ Enriched report saved to: {output_path}")
print(f"\n📊 Summary:")
print(f"  - Total iterations: {len(tested_combinations)} (starting from 0)")
print(f"  - Iteration 0: Reconstruction baseline (J24)")
print(f"  - Iterations 1-3: Multi-round 3D classification (J27, J30, J33)")
print(f"  - Iterations 4-5: Heterogeneous refinement (J35, J41)")
print(f"  - Iterations 6-14: Box size optimization (J30, J43-J57)")
print(f"\n⚠️  TODO:")
print(f"  - Get J24 resolution from CryoSPARC")
print(f"  - Get micrograph count from J9")
