"""ReAct-based heterogeneity analysis workflow orchestrator."""

import json
import csv
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .heterogeneity_agent import HeterogeneityAgent
from ...config.config_loader import CryoAgentConfig
from ...prompts.prompt_loader import load_prompt


class HeterogeneityStep(Enum):
    """Enumeration of heterogeneity analysis workflow steps."""
    ANALYZE_HETEROGENEITY = "analyze_heterogeneity"


@dataclass
class HeterogeneityResult:
    """Result of a heterogeneity analysis workflow execution."""
    step: HeterogeneityStep
    success: bool
    message: str = ""
    error: Optional[str] = None
    reasoning: Optional[str] = None
    converged_k: Optional[int] = None
    true_num_classes: Optional[int] = None
    filtered_groups: Optional[List[Dict[str, Any]]] = None
    final_refinement_jobs: Optional[List[Dict[str, Any]]] = None
    output_json_path: Optional[str] = None


class HeterogeneityWorkflow:
    """ReAct-based orchestrator for heterogeneity analysis workflows."""
    
    def __init__(self, agent: HeterogeneityAgent, config: CryoAgentConfig, stage_config_path: Optional[str] = None):
        """
        Initialize the heterogeneity analysis workflow.
        
        Args:
            agent: Heterogeneity analysis agent instance
            config: Complete configuration object
            stage_config_path: Path to stage-specific configuration file
        """
        self.agent = agent
        self.config = config
        self.results: List[HeterogeneityResult] = []
        self.workflow_state: Dict[str, Any] = {}
        
        # Load stage-specific configuration
        self.stage_config = self._load_stage_config(stage_config_path)
        self.workflow_params = self._parse_workflow_params()
        if hasattr(self.agent, "update_workflow_defaults"):
            try:
                self.agent.update_workflow_defaults(self.workflow_params)
            except Exception:
                # Non-fatal; agent may decline to store defaults
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
        
        # Heterogeneity analysis parameters
        heterogeneity_config = workflow_config.get("heterogeneity_analysis", {})
        initial_k_values = heterogeneity_config.get("initial_k_values", [3, 5])
        max_k = heterogeneity_config.get("max_k", 10)
        convergence_threshold = heterogeneity_config.get("convergence_threshold", 0)
        resolution_threshold = heterogeneity_config.get("resolution_threshold", 12.0)
        
        return {
            "initial_k_values": initial_k_values,
            "max_k": max_k,
            "convergence_threshold": convergence_threshold,
            "resolution_threshold": resolution_threshold
        }
    
    def execute_heterogeneity_analysis(
        self,
        refinement_job_uid: str,
        particles_job_uid: str,
        micrographs_job_uid: str,
        volume_job_uid: str,
        conversation_id: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> HeterogeneityResult:
        """
        Execute heterogeneity analysis workflow.
        
        Args:
            refinement_job_uid: UID of the first refinement job
            particles_job_uid: UID of the picking job (for particles)
            micrographs_job_uid: UID of the micrographs job
            volume_job_uid: UID of the initial volume
            conversation_id: Optional conversation ID for logging
            output_dir: Optional output directory for results
            
        Returns:
            HeterogeneityResult with analysis results
        """
        # Update workflow defaults with input parameters
        workflow_defaults = {
            "refinement_job_uid": refinement_job_uid,
            "particles_job_uid": particles_job_uid,
            "micrographs_job_uid": micrographs_job_uid,
            "volume_job_uid": volume_job_uid
        }
        if hasattr(self.agent, "update_workflow_defaults"):
            self.agent.update_workflow_defaults(workflow_defaults)
        
        initial_k_values = self.workflow_params.get("initial_k_values", [3, 5])
        prompt = load_prompt(
            "cryosparc/heterogeneity/task.md",
            {
                "refinement_job_uid": refinement_job_uid,
                "volume_job_uid": volume_job_uid,
                "particles_job_uid": particles_job_uid,
                "micrographs_job_uid": micrographs_job_uid,
                "initial_k_first": initial_k_values[0],
                "initial_k_second": initial_k_values[1] if len(initial_k_values) > 1 else initial_k_values[0],
                "resolution_threshold": self.workflow_params.get("resolution_threshold", 12.0),
                "max_k": self.workflow_params.get("max_k", 10),
            },
        )
        
        try:
            # Execute analysis using the agent
            result = self.agent.run_react_workflow(prompt, conversation_id=conversation_id)
            
            # Parse the result from tool execution log
            tool_execution_log = self.agent.get_tool_execution_log()
            
            # Collect results
            k_results = {}  # k -> {hetero_job_uid, density_folder, num_classes, clusters}
            all_density_folders = []
            group_refinement_jobs = {}  # group_id -> refinement_job_uid
            
            for tool_exec in tool_execution_log:
                tool_name = tool_exec.get("tool")
                tool_result = tool_exec.get("result")
                
                if tool_name == "run_ab_initio_hetero_combo" and tool_result:
                    try:
                        if isinstance(tool_result, str):
                            result_data = json.loads(tool_result)
                        else:
                            result_data = tool_result
                        
                        if result_data.get("success"):
                            k = result_data.get("k")
                            hetero_job_uid = result_data.get("hetero_job_uid")
                            if k and hetero_job_uid:
                                if k not in k_results:
                                    k_results[k] = {}
                                k_results[k]["hetero_job_uid"] = hetero_job_uid
                                k_results[k]["ab_initio_job_uid"] = result_data.get("ab_initio_job_uid")
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
                
                elif tool_name == "extract_density_maps" and tool_result:
                    try:
                        if isinstance(tool_result, str):
                            result_data = json.loads(tool_result)
                        else:
                            result_data = tool_result
                        
                        if result_data.get("success"):
                            hetero_job_uid = result_data.get("hetero_job_uid")
                            density_folder = result_data.get("output_folder")
                            all_density_folders.append(density_folder)
                            
                            # Find which K this belongs to
                            for k, k_data in k_results.items():
                                if k_data.get("hetero_job_uid") == hetero_job_uid:
                                    k_data["density_folder"] = density_folder
                                    k_data["num_maps"] = result_data.get("num_maps_extracted", 0)
                                    break
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
                
                elif tool_name == "compare_all_densities" and tool_result:
                    try:
                        # Parse clustering results from the tool output
                        # The result is a string that contains cluster information
                        # We can extract cluster information from the summary section
                        if isinstance(tool_result, str):
                            # Try to extract output directory from the result string
                            import re
                            output_dir_match = re.search(r'Output directory: (.+)', tool_result)
                            if output_dir_match:
                                output_dir_path = Path(output_dir_match.group(1))
                                cluster_csv = output_dir_path / "clusters.csv"
                                
                                # Read cluster information from CSV
                                if cluster_csv.exists():
                                    cluster_groups_info = {}
                                    with open(cluster_csv, 'r') as f:
                                        reader = csv.DictReader(f)
                                        for row in reader:
                                            map_name = row.get('map_name', '')
                                            cluster_id = int(row.get('cluster_id', '-1'))
                                            
                                            # Extract class ID from map name (e.g., "J838_class_00_00062_volume.mrc" -> 0)
                                            class_match = re.search(r'class_(\d+)', map_name)
                                            if class_match:
                                                class_id = int(class_match.group(1))
                                                
                                                if cluster_id not in cluster_groups_info:
                                                    cluster_groups_info[cluster_id] = []
                                                cluster_groups_info[cluster_id].append(class_id)
                                    
                                    # Store cluster information for later use
                                    tool_params = tool_exec.get("params", {})
                                    hetero_job_uid = tool_params.get("hetero_job_uid") or tool_params.get("folder", "")
                                    
                                    # Try to find which K this belongs to
                                    for k, k_data in k_results.items():
                                        if k_data.get("density_folder") in str(output_dir_path):
                                            k_data["cluster_groups"] = cluster_groups_info
                                            k_data["hetero_job_uid"] = hetero_job_uid
                                            break
                    except Exception as e:
                        # Non-fatal - continue processing
                        continue
                
                elif tool_name in ["run_non_uniform_refinement", "nonuniform_refine_new", "homogeneous_refinement"] and tool_result:
                    try:
                        # Track refinement jobs - these are the final refinement jobs for each group
                        if isinstance(tool_result, str):
                            result_data = json.loads(tool_result)
                        else:
                            result_data = tool_result
                        
                        if result_data.get("success") and result_data.get("job_uid"):
                            refinement_job_uid = result_data.get("job_uid")
                            # Extract group information from tool parameters and result
                            tool_params = tool_exec.get("params", {})
                            
                            # Determine refinement type
                            if tool_name == "run_non_uniform_refinement" or tool_name == "nonuniform_refine_new":
                                refinement_type = "nonuniform"
                            else:
                                refinement_type = "homogeneous"
                            
                            # Extract particles_group_names and volume_group_name to identify the group
                            particles_group_names = result_data.get("particles_group_names") or tool_params.get("particles_group_names", [])
                            volume_group_name = result_data.get("volume_group_name") or tool_params.get("volume_group_name")
                            
                            # Create a unique group identifier based on particles and volume groups
                            # This helps avoid duplicate entries for the same group
                            group_key = f"{volume_group_name}_{','.join(sorted(particles_group_names) if isinstance(particles_group_names, list) else [particles_group_names])}"
                            
                            # Only add if we haven't seen this group before
                            if group_key not in group_refinement_jobs:
                                group_refinement_jobs[group_key] = {
                                    "refinement_job_uid": refinement_job_uid,
                                    "refinement_type": refinement_type,
                                    "particles_group_names": particles_group_names,
                                    "volume_group_name": volume_group_name,
                                    "hetero_job_uid": result_data.get("hetero_job_uid") or tool_params.get("hetero_job_uid")
                                }
                    except (json.JSONDecodeError, TypeError, ValueError) as e:
                        continue
            
            # Determine convergence and final results
            # Extract information from k_results and group_refinement_jobs
            converged_k = None
            true_num_classes = None
            filtered_groups = []
            final_refinement_jobs = []
            
            # Try to determine converged_k and true_num_classes from k_results
            if k_results:
                # Find the K with the most groups (after filtering)
                # For now, use the highest K that has cluster information
                max_k = max(k_results.keys()) if k_results else None
                if max_k:
                    converged_k = max_k
                    k_data = k_results.get(max_k, {})
                    cluster_groups = k_data.get("cluster_groups", {})
                    if cluster_groups:
                        # Count number of clusters (groups)
                        true_num_classes = len(cluster_groups)
            
            # Build filtered_groups with refinement job UIDs
            # Each group should have: group_id, class_ids, best_resolution, refinement_job_uid
            group_index = 0
            for group_key, refinement_info in group_refinement_jobs.items():
                refinement_job_uid = refinement_info.get("refinement_job_uid")
                refinement_type = refinement_info.get("refinement_type", "nonuniform")
                particles_group_names = refinement_info.get("particles_group_names", [])
                volume_group_name = refinement_info.get("volume_group_name")
                hetero_job_uid = refinement_info.get("hetero_job_uid")
                
                # Extract class IDs from group names
                class_ids = []
                if isinstance(particles_group_names, list):
                    for group_name in particles_group_names:
                        # Extract class ID from "particles_class_X"
                        import re
                        match = re.search(r'class_(\d+)', group_name)
                        if match:
                            class_ids.append(int(match.group(1)))
                elif isinstance(particles_group_names, str):
                    import re
                    match = re.search(r'class_(\d+)', particles_group_names)
                    if match:
                        class_ids.append(int(match.group(1)))
                
                # Extract class ID from volume group name
                volume_class_id = None
                if volume_group_name:
                    import re
                    match = re.search(r'class_(\d+)', volume_group_name)
                    if match:
                        volume_class_id = int(match.group(1))
                
                group_data = {
                    "group_id": group_index,
                    "class_ids": sorted(set(class_ids)) if class_ids else [],
                    "volume_class_id": volume_class_id,
                    "particles_group_names": particles_group_names if isinstance(particles_group_names, list) else [particles_group_names],
                    "volume_group_name": volume_group_name,
                    "hetero_job_uid": hetero_job_uid,
                    "refinement_job_uid": refinement_job_uid,  # The final refinement job UID for this group
                    "homogeneous_refinement_job_uid": refinement_job_uid if refinement_type == "homogeneous" else None,
                    "nonuniform_refinement_job_uid": refinement_job_uid if refinement_type == "nonuniform" else None,
                    "refinement_type": refinement_type
                }
                filtered_groups.append(group_data)
                final_refinement_jobs.append({
                    "group_id": group_index,
                    "class_ids": sorted(set(class_ids)) if class_ids else [],
                    "volume_class_id": volume_class_id,
                    "particles_group_names": particles_group_names if isinstance(particles_group_names, list) else [particles_group_names],
                    "volume_group_name": volume_group_name,
                    "hetero_job_uid": hetero_job_uid,
                    "refinement_job_uid": refinement_job_uid,
                    "homogeneous_refinement_job_uid": refinement_job_uid if refinement_type == "homogeneous" else None,
                    "nonuniform_refinement_job_uid": refinement_job_uid if refinement_type == "nonuniform" else None,
                    "refinement_type": refinement_type
                })
                group_index += 1
            
            # Create output JSON with comprehensive information
            output_data = {
                "status": "completed",
                "converged_k": converged_k,
                "true_num_classes": true_num_classes,
                "k_results": k_results,
                "filtered_groups": filtered_groups,
                "final_refinement_jobs": final_refinement_jobs,
                "summary": {
                    "total_groups_refined": len(filtered_groups),
                    "refinement_jobs": [group.get("nonuniform_refinement_job_uid") or group.get("homogeneous_refinement_job_uid") 
                                       for group in filtered_groups if group.get("nonuniform_refinement_job_uid") or group.get("homogeneous_refinement_job_uid")]
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
            output_json_path = output_path / f"heterogeneity_analysis_results_{timestamp}.json"
            
            with open(output_json_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            heterogeneity_result = HeterogeneityResult(
                step=HeterogeneityStep.ANALYZE_HETEROGENEITY,
                success=True,
                message="Heterogeneity analysis completed",
                converged_k=converged_k,
                true_num_classes=true_num_classes,
                filtered_groups=filtered_groups,
                final_refinement_jobs=final_refinement_jobs,
                output_json_path=str(output_json_path)
            )
            
            self.results.append(heterogeneity_result)
            return heterogeneity_result
            
        except Exception as e:
            error_result = HeterogeneityResult(
                step=HeterogeneityStep.ANALYZE_HETEROGENEITY,
                success=False,
                error=str(e),
                message=f"Heterogeneity analysis failed: {str(e)}"
            )
            self.results.append(error_result)
            return error_result

