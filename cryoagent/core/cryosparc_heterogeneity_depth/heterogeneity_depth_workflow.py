"""ReAct-based heterogeneity depth analysis workflow orchestrator."""

import json
import csv
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .heterogeneity_depth_agent import HeterogeneityDepthAgent
from ...config.config_loader import CryoAgentConfig


class HeterogeneityDepthStep(Enum):
    """Enumeration of heterogeneity depth analysis workflow steps."""
    ANALYZE_HETEROGENEITY_DEPTH = "analyze_heterogeneity_depth"


@dataclass
class HeterogeneityDepthResult:
    """Result of a heterogeneity depth analysis workflow execution."""
    step: HeterogeneityDepthStep
    success: bool
    message: str = ""
    error: Optional[str] = None
    reasoning: Optional[str] = None
    final_refinement_job_uids: Optional[List[str]] = None  # List of all final job UIDs from all branches
    branches: Optional[List[Dict[str, Any]]] = None  # Tree structure with all branches
    output_json_path: Optional[str] = None
    
    def __post_init__(self):
        """Initialize default values for lists."""
        if self.final_refinement_job_uids is None:
            self.final_refinement_job_uids = []
        if self.branches is None:
            self.branches = []


class HeterogeneityDepthWorkflow:
    """ReAct-based orchestrator for heterogeneity depth analysis workflows."""
    
    def __init__(self, agent: HeterogeneityDepthAgent, config: CryoAgentConfig, stage_config_path: Optional[str] = None):
        """
        Initialize the heterogeneity depth analysis workflow.
        
        Args:
            agent: Heterogeneity depth analysis agent instance
            config: Complete configuration object
            stage_config_path: Path to stage-specific configuration file
        """
        self.agent = agent
        self.config = config
        self.results: List[HeterogeneityDepthResult] = []
        self.workflow_state: Dict[str, Any] = {}
        
        # Load stage-specific configuration
        self.stage_config = self._load_stage_config(stage_config_path)
        self.workflow_params = self._parse_workflow_params()
        if hasattr(self.agent, "update_workflow_defaults"):
            try:
                self.agent.update_workflow_defaults(self.workflow_params)
            except Exception:
                pass
    
    def _load_stage_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load stage-specific configuration from JSON file."""
        import json
        from pathlib import Path
        
        if not config_path:
            return {}
        
        stage_config_path = Path(config_path)
        if not stage_config_path.exists():
            return {}
        
        with open(stage_config_path, 'r') as f:
            return json.load(f)
    
    def _parse_workflow_params(self) -> Dict[str, Any]:
        """Parse workflow parameters from stage config."""
        workflow_config = self.stage_config.get("workflow", {})
        
        # Heterogeneity depth analysis parameters
        depth_config = workflow_config.get("heterogeneity_depth_analysis", {})
        k = depth_config.get("k", 4)
        resolution_threshold = depth_config.get("resolution_threshold", 12.0)
        
        return {
            "k": k,
            "resolution_threshold": resolution_threshold
        }
    
    def execute_heterogeneity_depth_analysis(
        self,
        conversation_id: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> HeterogeneityDepthResult:
        """
        Execute heterogeneity depth analysis workflow.
        
        Args:
            conversation_id: Optional conversation ID for logging
            output_dir: Optional output directory for results
            
        Returns:
            HeterogeneityDepthResult with analysis results
        """
        k = self.workflow_params.get("k", 4)
        resolution_threshold = self.workflow_params.get("resolution_threshold", 10.0)
        
        prompt = f"""Perform heterogeneity depth analysis on EACH starting cluster from the heterogeneity analysis results.

**MANDATORY FIRST STEP:** call `read_input_json` before any refinement job. Do not proceed until clusters are loaded from `heterogeneity_analysis_results_*.json`.

## Goal
Among structures that refine below {resolution_threshold} Å, determine whether further heterogeneity exists. Split good branches recursively; discard bad density clusters; terminate every branch with non-uniform refinement + FSC resolution.

## Resolution threshold: {resolution_threshold} Å
- GOOD class: resolution < {resolution_threshold} Å (lower is better)
- BAD class: resolution ≥ {resolution_threshold} Å
- PER CLUSTER / PER BRANCH filtering — a passing sibling does NOT rescue a failed cluster

## After each `run_heterogeneous_refinement` (auto-includes density comparison)
Use the returned fields directly — do NOT manually re-run compare unless re-analyzing an old job.

### Case A — ZERO good classes (`good_classes` empty or `fallback_non_uniform` present)
- Do NOT discard the branch
- `run_non_uniform_refinement` with:
  * `particles_group_names=["particles_all_classes"]` from the hetero job
  * `volume_group_name` = best-volume class among ALL classes (lowest resolution)
- `wait_for_job` → `get_fsc_info` → branch COMPLETE

### Case B — one or more good classes
1. Read KEPT vs FILTERED OUT clusters from `density_comparison`
2. **Throw away every BAD / FILTERED OUT density cluster** — no further hetero or refinement on them
3. Among KEPT clusters only:

**B1 — ONE KEPT cluster** → branch converged:
- `run_non_uniform_refinement` on that cluster's good `particles_class_X` + best `volume_class_X`
- `wait_for_job` → `get_fsc_info` → branch COMPLETE

**B2 — MULTIPLE KEPT clusters** → split and recurse:
- For EACH KEPT cluster only (skip discarded ones):
  * `run_heterogeneous_refinement` with K={k} on that cluster's good classes
  * Repeat Case A / B on the new hetero result

## Execution steps

1. **Read input JSON** → get all starting clusters from `final_refinement_jobs`
   - Each cluster: refinement_job_uid, particles_job_uid, volume_job_uid, particles_group_names, volume_group_name
   - Upstream refinement jobs use `particles_group_names=["particles"]`, `volume_group_name="volume"`
   - Process each starting cluster as an independent tree

2. **For each starting cluster:**
   a. `run_heterogeneous_refinement` with K={k} on cluster particles + volume (always hetero first — never skip to non-uniform)
   b. Apply Case A / B / B1 / B2 to the result
   c. Recurse on KEPT sub-branches until all branches terminate

3. **Record outputs for every terminated branch:**
   - Final refinement job UID
   - `final_resolution_angstroms` from `get_fsc_info`
   - Bad clusters discarded at every round (never refined further)

## Rules you must follow
- BAD density clusters after comparison → discard completely, even if another cluster in the same job passed
- Zero good classes → fallback non-uniform on all particles (branch is NOT abandoned)
- One KEPT good cluster → converged → non-uniform refinement (NOT homogeneous)
- Multiple KEPT good clusters → hetero on each good cluster separately
- Before each job, state which case (A, B1, B2) applies and which clusters you keep vs discard"""
        
        try:
            # Execute analysis using the agent
            result = self.agent.run_react_workflow(prompt, conversation_id=conversation_id)
            
            # Parse the result from tool execution log
            tool_execution_log = self.agent.get_tool_execution_log()
            
            # Collect results - track all branches and final job UIDs
            final_refinement_job_uids = []  # List of ALL final job UIDs from all terminal branches
            final_refinement_resolutions = {}  # job_uid -> resolution_angstroms after non-uniform refinement
            hetero_jobs = []  # All heterogeneous refinement jobs
            branches = []  # Tree structure with branch information
            comparison_results = []
            
            # Track the relationship between hetero jobs and their resulting clusters
            hetero_to_clusters = {}  # hetero_job_uid -> list of clusters found
            hetero_to_final = {}  # hetero_job_uid -> final refinement job UID (if branch terminated)
            
            for tool_exec in tool_execution_log:
                tool_name = tool_exec.get("tool")
                tool_result = tool_exec.get("result")
                tool_params = tool_exec.get("params", {})
                
                if tool_name == "run_heterogeneous_refinement" and tool_result:
                    try:
                        if isinstance(tool_result, str):
                            result_data = json.loads(tool_result)
                        else:
                            result_data = tool_result
                        
                        if result_data.get("success"):
                            hetero_job_uid = result_data.get("hetero_job_uid")
                            if hetero_job_uid:
                                hetero_jobs.append(hetero_job_uid)
                                # Track parent job if available
                                parent_job = tool_params.get("particles_job_uid") or tool_params.get("volume_job_uid")
                                branches.append({
                                    "type": "heterogeneous_refinement",
                                    "job_uid": hetero_job_uid,
                                    "parent_job_uid": parent_job,
                                    "k": tool_params.get("k", 4)
                                })
                                density_comparison = result_data.get("density_comparison")
                                if density_comparison:
                                    comparison_results.append(density_comparison)
                                    cluster_match = re.search(
                                        r"Number of clusters \(groups\): (\d+)",
                                        density_comparison,
                                    )
                                    if cluster_match:
                                        hetero_to_clusters[hetero_job_uid] = int(cluster_match.group(1))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
                
                elif tool_name in ("run_non_uniform_refinement", "run_homogeneous_refinement") and tool_result:
                    try:
                        if isinstance(tool_result, str):
                            result_data = json.loads(tool_result)
                        else:
                            result_data = tool_result
                        
                        if result_data.get("success") and result_data.get("job_uid"):
                            final_job_uid = result_data.get("job_uid")
                            if final_job_uid not in final_refinement_job_uids:
                                final_refinement_job_uids.append(final_job_uid)
                            parent_hetero = tool_params.get("hetero_job_uid") or tool_params.get("particles_job_uid") or tool_params.get("volume_job_uid")
                            branches.append({
                                "type": tool_name,
                                "job_uid": final_job_uid,
                                "parent_job_uid": parent_hetero,
                                "is_terminal": True,
                                "final_refinement_job_uid": final_job_uid,
                                "final_resolution_angstroms": final_refinement_resolutions.get(final_job_uid),
                            })
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue

                elif tool_name == "get_fsc_info" and tool_result:
                    try:
                        if isinstance(tool_result, str):
                            result_data = json.loads(tool_result)
                        else:
                            result_data = tool_result
                        if result_data.get("success"):
                            job_uid = result_data.get("refinement_job_uid")
                            resolution = result_data.get("resolution_angstroms")
                            if job_uid and resolution is not None:
                                final_refinement_resolutions[job_uid] = resolution
                                for branch in branches:
                                    if branch.get("job_uid") == job_uid:
                                        branch["final_resolution_angstroms"] = resolution
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
                
                elif tool_name == "compare_all_densities" and tool_result:
                    try:
                        if isinstance(tool_result, str):
                            comparison_results.append(tool_result)
                            cluster_match = re.search(r"Number of clusters \(groups\): (\d+)", tool_result)
                            if cluster_match:
                                num_clusters = int(cluster_match.group(1))
                                # Find the most recent hetero job to associate with this comparison
                                if hetero_jobs:
                                    hetero_to_clusters[hetero_jobs[-1]] = num_clusters
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
            
            # Build tree structure
            tree_structure = self._build_tree_structure(branches, hetero_to_clusters)
            
            # Create output JSON with comprehensive information
            converged_branches = [
                {
                    "final_refinement_job_uid": uid,
                    "final_resolution_angstroms": final_refinement_resolutions.get(uid),
                }
                for uid in final_refinement_job_uids
            ]
            output_data = {
                "status": "completed",
                "final_refinement_job_uids": final_refinement_job_uids,
                "final_refinement_resolutions": final_refinement_resolutions,
                "converged_branches": converged_branches,
                "total_final_refinements": len(final_refinement_job_uids),
                "hetero_jobs": hetero_jobs,
                "branches": branches,
                "tree_structure": tree_structure,
                "summary": {
                    "total_hetero_jobs": len(hetero_jobs),
                    "total_final_refinements": len(final_refinement_job_uids),
                    "final_refinement_job_uids": final_refinement_job_uids,
                    "converged_branches": converged_branches,
                }
            }
            
            # Save output JSON
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
            else:
                output_path = Path("outputs")
                output_path.mkdir(parents=True, exist_ok=True)
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_json_path = output_path / f"heterogeneity_depth_analysis_results_{timestamp}.json"
            
            with open(output_json_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            heterogeneity_depth_result = HeterogeneityDepthResult(
                step=HeterogeneityDepthStep.ANALYZE_HETEROGENEITY_DEPTH,
                success=True,
                message="Heterogeneity depth analysis completed",
                final_refinement_job_uids=final_refinement_job_uids,
                branches=branches,
                output_json_path=str(output_json_path)
            )
            
            self.results.append(heterogeneity_depth_result)
            return heterogeneity_depth_result
            
        except Exception as e:
            error_result = HeterogeneityDepthResult(
                step=HeterogeneityDepthStep.ANALYZE_HETEROGENEITY_DEPTH,
                success=False,
                error=str(e),
                message=f"Heterogeneity depth analysis failed: {str(e)}"
            )
            self.results.append(error_result)
            return error_result
    
    def _build_tree_structure(self, branches: List[Dict[str, Any]], hetero_to_clusters: Dict[str, int]) -> Dict[str, Any]:
        """Build a tree structure representation of the workflow branches."""
        tree = {
            "starting_clusters": [],
            "branches": []
        }
        
        # Group branches by type
        hetero_branches = [b for b in branches if b.get("type") == "heterogeneous_refinement"]
        final_branches = [b for b in branches if b.get("type") == "homogeneous_refinement"]
        
        # Build tree starting from final branches and working backwards
        for final_branch in final_branches:
            branch_path = [final_branch]
            current_job = final_branch.get("parent_job_uid")
            
            # Trace back to find the path
            while current_job:
                parent_hetero = next((b for b in hetero_branches if b.get("job_uid") == current_job), None)
                if parent_hetero:
                    branch_path.insert(0, parent_hetero)
                    current_job = parent_hetero.get("parent_job_uid")
                else:
                    # This is a starting cluster
                    tree["starting_clusters"].append(current_job)
                    break
            
            tree["branches"].append({
                "path": branch_path,
                "final_refinement_job_uid": final_branch.get("final_refinement_job_uid"),
                "depth": len(branch_path) - 1  # Depth of the tree (number of hetero refinements)
            })
        
        return tree

