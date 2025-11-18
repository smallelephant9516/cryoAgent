"""ReAct-based box size optimization workflow orchestrator."""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from .optimizer_agent import OptimizerAgent
from ...config.config_loader import CryoAgentConfig


class OptimizationStep(Enum):
    """Enumeration of optimization workflow steps."""
    OPTIMIZE_DIAMETER = "optimize_diameter"


@dataclass
class OptimizationResult:
    """Result of an optimization workflow execution."""
    step: OptimizationStep
    success: bool
    job_uid: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    reasoning: Optional[str] = None
    best_box_size: Optional[int] = None
    best_resolution: Optional[float] = None
    tested_combinations: Optional[List[Dict[str, Any]]] = None


class OptimizerWorkflow:
    """ReAct-based orchestrator for box size optimization workflows."""
    
    def __init__(self, agent: OptimizerAgent, config: CryoAgentConfig, stage_config_path: Optional[str] = None):
        """
        Initialize the optimization workflow.
        
        Args:
            agent: Optimization agent instance
            config: Complete configuration object
            stage_config_path: Path to stage-specific configuration file
        """
        self.agent = agent
        self.config = config
        self.results: List[OptimizationResult] = []
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
        
        # Optimization parameters
        optimization_config = workflow_config.get("optimization", {})
        max_iterations = optimization_config.get("max_iterations", 5)
        box_size_variation = optimization_config.get("box_size_variation", 0.1)  # 10%
        
        return {
            "max_iterations": max_iterations,
            "box_size_variation": box_size_variation
        }
    
    def execute_optimization(
        self,
        refinement_job_uid: str,
        particles_job_uid: str,
        micrographs_job_uid: str,
        volume_job_uid: str,
        conversation_id: Optional[str] = None
    ) -> OptimizationResult:
        """
        Execute optimization workflow (box size and/or heterogeneous refinement).
        
        Args:
            refinement_job_uid: UID of the first refinement job
            particles_job_uid: UID of the picking job (for re-extraction)
            micrographs_job_uid: UID of the micrographs job
            volume_job_uid: UID of the initial volume
            conversation_id: Optional conversation ID for logging
            
        Returns:
            OptimizationResult with optimization results
        """
        # Check configuration flags
        workflow_config = self.stage_config.get("workflow", {})
        optimization_config = workflow_config.get("optimization", {})
        enable_box_size = optimization_config.get("enable_box_size_optimization", True)
        enable_hetero = optimization_config.get("enable_heterogeneous_refinement", False)
        
        # Update workflow defaults with input parameters
        workflow_defaults = {
            "refinement_job_uid": refinement_job_uid,
            "particles_job_uid": particles_job_uid,
            "micrographs_job_uid": micrographs_job_uid,
            "volume_job_uid": volume_job_uid
        }
        if hasattr(self.agent, "update_workflow_defaults"):
            self.agent.update_workflow_defaults(workflow_defaults)
        
        # Prepare prompt based on enabled optimizations
        if enable_box_size and enable_hetero:
            prompt = f"""Optimize both box size and heterogeneous refinement for 3D reconstruction.

I have completed the first round of homogeneous refinement with job UID: {refinement_job_uid}
The initial volume is from job: {volume_job_uid}
Particles can be re-extracted from picking job: {particles_job_uid}
Micrographs are available from job: {micrographs_job_uid}

Please FIRST optimize the box size, then use the optimized refinement job for heterogeneous refinement optimization."""
        elif enable_box_size:
            prompt = f"""Optimize the box size for 3D reconstruction.

I have completed the first round of homogeneous refinement with job UID: {refinement_job_uid}
The initial volume is from job: {volume_job_uid}
Particles can be re-extracted from picking job: {particles_job_uid}
Micrographs are available from job: {micrographs_job_uid}

Please optimize the box size using the test_box_size tool."""
        elif enable_hetero:
            prompt = f"""Optimize heterogeneous refinement for 3D reconstruction.

I have completed homogeneous refinement with job UID: {refinement_job_uid}

IMPORTANT: Box size optimization is DISABLED. DO NOT use test_box_size tool.
Proceed directly to heterogeneous refinement optimization using the refinement_job_uid provided ({refinement_job_uid}).
Use the test_heterogeneous_refinement tool to test different K values."""
        else:
            prompt = f"""I have a refinement job with UID: {refinement_job_uid}

Note: Both box size optimization and heterogeneous refinement are disabled.
Please report the current resolution from this refinement job."""
        
        try:
            # Execute optimization using the agent
            result = self.agent.run_react_workflow(prompt, conversation_id=conversation_id)
            
            # Parse the result - look for test_box_size and get_fsc_info tool executions
            tool_execution_log = self.agent.get_tool_execution_log()
            
            # Collect all test results from test_box_size tool executions
            tested_combinations = []
            best_result = None
            best_resolution = float('inf')  # Lower is better
            
            for tool_exec in tool_execution_log:
                tool_name = tool_exec.get("tool")
                tool_result = tool_exec.get("result")
                
                if tool_name == "test_box_size" and tool_result:
                    try:
                        # Parse the result
                        if isinstance(tool_result, str):
                            result_data = json.loads(tool_result)
                        else:
                            result_data = tool_result
                        
                        if result_data.get("success"):
                            box_size = result_data.get("box_size_pix") or result_data.get("box_size")
                            resolution = result_data.get("resolution_angstroms")
                            job_uid = result_data.get("refinement_job_uid")
                            
                            if box_size is not None and resolution is not None:
                                tested_combinations.append({
                                    "box_size": box_size,
                                    "resolution": resolution,
                                    "job_uid": job_uid
                                })
                                
                                # Track best result (lowest resolution)
                                if resolution < best_resolution:
                                    best_resolution = resolution
                                    best_result = {
                                        "box_size": box_size,
                                        "resolution": resolution,
                                        "job_uid": job_uid
                                    }
                    except (json.JSONDecodeError, TypeError, ValueError) as e:
                        # Skip invalid results
                        continue
                
                elif tool_name == "get_fsc_info" and tool_result:
                    # Also check get_fsc_info for baseline (original refinement)
                    try:
                        if isinstance(tool_result, str):
                            result_data = json.loads(tool_result)
                        else:
                            result_data = tool_result
                        
                        if result_data.get("success"):
                            box_size = result_data.get("box_size")
                            resolution = result_data.get("resolution_angstroms")
                            job_uid = result_data.get("refinement_job_uid")
                            
                            if box_size is not None and resolution is not None:
                                # Check if this is already in tested_combinations
                                found = False
                                for combo in tested_combinations:
                                    if combo.get("box_size") == box_size:
                                        found = True
                                        break
                                
                                if not found:
                                    tested_combinations.append({
                                        "box_size": box_size,
                                        "resolution": resolution,
                                        "job_uid": job_uid
                                    })
                                    
                                    # Track best result (lowest resolution)
                                    if resolution < best_resolution:
                                        best_resolution = resolution
                                        best_result = {
                                            "box_size": box_size,
                                            "resolution": resolution,
                                            "job_uid": job_uid
                                        }
                    except (json.JSONDecodeError, TypeError, ValueError) as e:
                        # Skip invalid results
                        continue
            
            # Return result based on what we found
            if best_result and tested_combinations:
                return OptimizationResult(
                    step=OptimizationStep.OPTIMIZE_DIAMETER,
                    success=True,
                    job_uid=best_result.get("job_uid"),
                    best_box_size=best_result.get("box_size"),
                    best_resolution=best_result.get("resolution"),
                    tested_combinations=tested_combinations,
                    message=f"Optimization completed successfully. Best box size: {best_result.get('box_size')}px with resolution: {best_result.get('resolution'):.3f} Å"
                )
            elif tested_combinations:
                # Have results but couldn't determine best
                return OptimizationResult(
                    step=OptimizationStep.OPTIMIZE_DIAMETER,
                    success=False,
                    error="Could not determine best box size from results",
                    tested_combinations=tested_combinations,
                    message="Optimization workflow executed but best result could not be determined"
                )
            else:
                # No result data found - check if we have any tool executions
                if tool_execution_log:
                    # At least some tools were executed
                    return OptimizationResult(
                        step=OptimizationStep.OPTIMIZE_DIAMETER,
                        success=False,
                        error="Could not parse optimization results from tool execution",
                        message="Optimization workflow executed but results could not be extracted"
                    )
                else:
                    # No tools were executed
                    return OptimizationResult(
                        step=OptimizationStep.OPTIMIZE_DIAMETER,
                        success=False,
                        error="No optimization tools were executed",
                        message="The agent did not execute any optimization tools"
                    )
                
        except Exception as e:
            return OptimizationResult(
                step=OptimizationStep.OPTIMIZE_DIAMETER,
                success=False,
                error=str(e),
                message=f"Failed to execute optimization: {str(e)}"
            )

