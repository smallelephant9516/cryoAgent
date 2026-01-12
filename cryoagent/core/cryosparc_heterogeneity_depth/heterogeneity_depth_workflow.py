"""ReAct-based heterogeneity depth analysis workflow orchestrator."""

import json
import csv
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
        
        prompt = f"""Perform heterogeneity depth analysis to iteratively refine until only one cluster remains OR no cluster passes resolution threshold for EACH cluster from the heterogeneity analysis results.

**Resolution threshold: {resolution_threshold} Å**
- Classes with resolution BETTER (lower) than {resolution_threshold} Å PASS the threshold
- Classes with resolution WORSE (higher) than {resolution_threshold} Å FAIL the threshold
- If NO cluster passes the threshold, terminate the branch with homogeneous refinement

Workflow:
1. Read input JSON from heterogeneity job (heterogeneity_analysis_results_*.json) - PREFERRED
   - The tool will return all clusters from final_refinement_jobs
   - Each cluster contains: refinement_job_uid, particles_job_uid, volume_job_uid, particles_group_name, volume_group_name
   - You need to process EACH cluster separately

2. For EACH cluster from the heterogeneity analysis results:
   a. **Initial Heterogeneous Refinement** (ALWAYS do this first - do NOT skip to homogeneous refinement):
      - Use particles_job_uid and volume_job_uid from the cluster (from hetero job J10)
      - Use particles_group_names and volume_group_name from the cluster
      - Run heterogeneous refinement with K={k} using:
        * Particles: Use particles_group_names from the cluster (e.g., ["particles_class_0"] or ["particles_class_0", "particles_class_2"])
        * Volume: Use volume_group_name from the cluster (e.g., "volume_class_0")
      - Wait for completion

   b. **Extract and Compare Densities**:
      - Extract density maps from heterogeneous refinement job
      - Compare all density maps using compare_all_densities tool
      - Check number of clusters from comparison results
      - **CRITICAL**: Get class resolutions using `get_hetero_class_resolutions` tool
      - Check if any class passes the resolution threshold (resolution_threshold from config, default: 10.0 Å)
      - Resolution threshold means: classes with resolution BETTER (lower) than threshold pass, classes with resolution WORSE (higher) than threshold fail

   c. **Tree Structure Expansion (Recursive)**:
      - **CRITICAL**: After the initial heterogeneous refinement, ALWAYS check comparison results AND class resolutions
      - **If NO cluster passes resolution threshold** (all classes have resolution worse than threshold):
        * This branch should terminate with homogeneous refinement
        * **CRITICAL**: Use `particles_all_classes` from the heterogeneous refinement job (NOT particles_class_X)
        * Get best volume from all classes (volume_class_X with best resolution, even if it doesn't pass threshold)
        * Run homogeneous refinement with:
          - particles_job_uid: the heterogeneous refinement job UID
          - particles_group_name: "particles_all_classes" (or leave empty to default to particles_all_classes)
          - volume_job_uid: the heterogeneous refinement job UID
          - volume_group_name: volume_class_X with best resolution
        * Record this final refinement job UID - this branch is COMPLETE
        * Move to next starting cluster
      
      - **If comparison shows multiple clusters (e.g., 2, 3, or more) AND at least one cluster passes resolution threshold:**
        * This creates a TREE STRUCTURE - the hetero job splits into multiple branches
        * For EACH new cluster found in the comparison:
          i. Extract which classes belong to this cluster from comparison results
          ii. **CRITICAL**: Determine how many classes are in this cluster:
             - If cluster has only 1 class (e.g., class 3): Use `particles_group_names=["particles_class_3"]` (list with single element)
             - If cluster has multiple classes (e.g., classes 0 and 2): Use `particles_group_names=["particles_class_0", "particles_class_2"]` (list with all classes)
          iii. Get volume from best class in that cluster (volume_class_X with best resolution)
          iv. Run heterogeneous refinement with K={k} using:
             * particles_group_names: List of particles_class_X for each class in the cluster
             * volume_group_name: volume_class_X with best resolution
          v. Wait for completion → get new hetero job
          vi. **CRITICAL - DO NOT STOP HERE**: After the heterogeneous refinement job completes, you MUST continue:
             a. Extract density maps from the new hetero job using `extract_density_maps` tool
             b. Compare all density maps using `compare_all_densities` tool
             c. Get class resolutions using `get_hetero_class_resolutions` tool
             d. Check if any class passes the resolution threshold
          vii. **Recursively apply the same logic based on comparison and resolution results:**
             - If NO cluster passes resolution threshold → use particles_all_classes from that hetero job + best volume → run homogeneous refinement → record final job UID → branch COMPLETE
             - If 1 cluster → use particles_all_classes from that hetero job + best volume → run homogeneous refinement → record final job UID → branch COMPLETE
             - If multiple clusters AND at least one passes threshold → split into more branches → repeat steps i-vii recursively for EACH new cluster
        * **CRITICAL**: Do NOT stop after a heterogeneous refinement job completes. ALWAYS extract density maps, compare, check resolutions, and continue recursively until the branch terminates.
        * Continue this recursive expansion until EVERY branch reaches only 1 cluster OR no cluster passes resolution threshold
        * Each branch that reaches 1 cluster OR no cluster passes threshold gets a final homogeneous refinement job UID
      
      - **If comparison shows only 1 cluster:**
        * This branch is ready for final homogeneous refinement
        * **CRITICAL**: Use `particles_all_classes` from the heterogeneous refinement job (NOT particles_class_X)
        * This uses ALL particles from the heterogeneous refinement job, regardless of which classes are in the cluster
        * Get best volume from the classes in this cluster (volume_class_X with best resolution)
        * Run homogeneous refinement with:
          - particles_job_uid: the heterogeneous refinement job UID
          - particles_group_name: "particles_all_classes" (or leave empty to default to particles_all_classes)
          - volume_job_uid: the heterogeneous refinement job UID
          - volume_group_name: volume_class_X with best resolution
        * Record this final refinement job UID - this branch is COMPLETE
        * Move to next starting cluster

   d. **Final Homogeneous Refinement** (for each terminal branch):
      - A branch terminates when:
        * Comparison shows only 1 cluster in a heterogeneous refinement job, OR
        * NO cluster passes the resolution threshold (all classes have resolution worse than threshold)
      - **CRITICAL**: Use `particles_all_classes` from that heterogeneous refinement job (NOT particles_class_X)
      - This uses ALL particles from the heterogeneous refinement job, not just specific classes
      - Get best volume from the classes (volume_class_X with best resolution)
      - Run homogeneous refinement with:
        - particles_job_uid: the heterogeneous refinement job UID
        - particles_group_name: "particles_all_classes" (or leave empty to default to particles_all_classes)
        - volume_job_uid: the heterogeneous refinement job UID
        - volume_group_name: volume_class_X with best resolution
      - Record the final refinement job UID for this branch
      - This completes the depth analysis for this branch

3. After processing all clusters:
   - All starting clusters have been analyzed independently
   - Each starting cluster has been expanded into a tree structure
   - Every branch in every tree has been refined until only one cluster remains OR no cluster passes resolution threshold
   - Final homogeneous refinement has been performed for each terminal branch
   - ALL final refinement job UIDs from ALL terminal branches have been recorded

**CRITICAL**: 
- Track and record ALL final refinement job UIDs from ALL terminal branches. The output JSON must include all branches and their final job UIDs.
- Always check class resolutions after each heterogeneous refinement using `get_hetero_class_resolutions` tool
- If NO cluster passes the resolution threshold, terminate the branch with homogeneous refinement (even if multiple clusters exist)
- Resolution threshold is {resolution_threshold} Å (from config) - classes with resolution BETTER (lower) than this pass, classes with resolution WORSE (higher) than this fail"""
        
        try:
            # Execute analysis using the agent
            result = self.agent.run_react_workflow(prompt, conversation_id=conversation_id)
            
            # Parse the result from tool execution log
            tool_execution_log = self.agent.get_tool_execution_log()
            
            # Collect results - track all branches and final job UIDs
            final_refinement_job_uids = []  # List of ALL final job UIDs from all terminal branches
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
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
                
                elif tool_name == "run_homogeneous_refinement" and tool_result:
                    try:
                        if isinstance(tool_result, str):
                            result_data = json.loads(tool_result)
                        else:
                            result_data = tool_result
                        
                        if result_data.get("success") and result_data.get("job_uid"):
                            final_job_uid = result_data.get("job_uid")
                            final_refinement_job_uids.append(final_job_uid)
                            # Track which hetero job this final refinement came from
                            parent_hetero = tool_params.get("particles_job_uid") or tool_params.get("volume_job_uid")
                            branches.append({
                                "type": "homogeneous_refinement",
                                "job_uid": final_job_uid,
                                "parent_job_uid": parent_hetero,
                                "is_terminal": True,
                                "final_refinement_job_uid": final_job_uid
                            })
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
                
                elif tool_name == "compare_all_densities" and tool_result:
                    try:
                        if isinstance(tool_result, str):
                            comparison_results.append(tool_result)
                            # Try to extract number of clusters from the result
                            import re
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
            output_data = {
                "status": "completed",
                "final_refinement_job_uids": final_refinement_job_uids,  # All final job UIDs from all branches
                "total_final_refinements": len(final_refinement_job_uids),
                "hetero_jobs": hetero_jobs,
                "branches": branches,
                "tree_structure": tree_structure,
                "summary": {
                    "total_hetero_jobs": len(hetero_jobs),
                    "total_final_refinements": len(final_refinement_job_uids),
                    "final_refinement_job_uids": final_refinement_job_uids
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

