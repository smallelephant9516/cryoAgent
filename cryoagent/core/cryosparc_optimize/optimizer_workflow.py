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
        Execute box size optimization workflow.
        
        Args:
            refinement_job_uid: UID of the first refinement job
            particles_job_uid: UID of the picking job (for re-extraction)
            micrographs_job_uid: UID of the micrographs job
            volume_job_uid: UID of the initial volume
            conversation_id: Optional conversation ID for logging
            
        Returns:
            OptimizationResult with optimization results
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
        
        # Prepare prompt for optimization
        prompt = f"""Optimize the box size for 3D reconstruction.

I have completed the first round of homogeneous refinement with job UID: {refinement_job_uid}
The initial volume is from job: {volume_job_uid}
Particles can be re-extracted from picking job: {particles_job_uid}
Micrographs are available from job: {micrographs_job_uid}

Please run the optimize_diameter tool to find the optimal box size."""
        
        try:
            # Execute optimization using the agent
            result = self.agent.run_react_workflow(prompt, conversation_id=conversation_id)
            
            # Parse the result - look for optimize_diameter tool execution in the log
            tool_execution_log = self.agent.get_tool_execution_log()
            optimize_result_data = None
            
            # Find the optimize_diameter tool execution (check from end to get the last one)
            for tool_exec in reversed(tool_execution_log):
                if tool_exec.get("tool") == "optimize_diameter":
                    tool_result = tool_exec.get("result")
                    if tool_result:
                        try:
                            if isinstance(tool_result, str):
                                optimize_result_data = json.loads(tool_result)
                            else:
                                optimize_result_data = tool_result
                            break
                        except (json.JSONDecodeError, TypeError):
                            # Try to extract from result dict directly
                            if isinstance(tool_result, dict):
                                optimize_result_data = tool_result
                                break
                
            # If not found in log, try to parse from result string directly
            if not optimize_result_data:
                # run_react_workflow returns a string, try to parse it as JSON
                try:
                    if isinstance(result, str):
                        # Try to extract JSON from the result string
                        result_dict = json.loads(result)
                        if result_dict.get("success"):
                            tool_results = result_dict.get("tool_results", {})
                            optimize_result = tool_results.get("optimize_diameter")
                            if optimize_result:
                                optimize_result_data = json.loads(optimize_result) if isinstance(optimize_result, str) else optimize_result
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass
            
            # Return result based on what we found
            if optimize_result_data and optimize_result_data.get("success"):
                return OptimizationResult(
                    step=OptimizationStep.OPTIMIZE_DIAMETER,
                    success=True,
                    job_uid=optimize_result_data.get("best_job_uid"),  # Map best_job_uid from result to job_uid field
                    best_box_size=optimize_result_data.get("best_box_size"),
                    best_resolution=optimize_result_data.get("best_resolution_angstroms"),
                    tested_combinations=optimize_result_data.get("tested_combinations", []),
                    message="Optimization completed successfully"
                )
            elif optimize_result_data:
                # Found data but not successful
                return OptimizationResult(
                    step=OptimizationStep.OPTIMIZE_DIAMETER,
                    success=False,
                    error=optimize_result_data.get("error", "Optimization failed"),
                    message="Optimization did not complete successfully"
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
                        message="The agent did not execute the optimize_diameter tool"
                    )
                
        except Exception as e:
            return OptimizationResult(
                step=OptimizationStep.OPTIMIZE_DIAMETER,
                success=False,
                error=str(e),
                message=f"Failed to execute optimization: {str(e)}"
            )

