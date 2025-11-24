"""ReAct-based 2D classification optimization workflow orchestrator."""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from .optimizer_2d_agent import Optimizer2DAgent
from ...config.config_loader import CryoAgentConfig


class Optimization2DStep(Enum):
    """Enumeration of 2D optimization workflow steps."""
    OPTIMIZE_2D = "optimize_2d"


@dataclass
class Optimization2DResult:
    """Result of a 2D optimization workflow execution."""
    step: Optimization2DStep
    success: bool
    final_particles_job_uid: Optional[str] = None
    final_good_particles_count: Optional[int] = None
    final_good_particles_percentage: Optional[float] = None
    total_rounds: Optional[int] = None
    message: str = ""
    error: Optional[str] = None
    workflow_summary: Optional[Dict[str, Any]] = None


class Optimizer2DWorkflow:
    """ReAct-based orchestrator for 2D classification optimization workflows."""
    
    def __init__(self, agent: Optimizer2DAgent, config: CryoAgentConfig, stage_config_path: Optional[str] = None):
        """
        Initialize the 2D optimization workflow.
        
        Args:
            agent: 2D optimization agent instance
            config: Complete configuration object
            stage_config_path: Path to stage-specific configuration file
        """
        self.agent = agent
        self.config = config
        self.results: List[Optimization2DResult] = []
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
        opt_config = workflow_config.get("2d_optimization", {})
        
        return {
            "enable_function1_iterative": opt_config.get("enable_function1_iterative", True),
            "enable_function2_rescue": opt_config.get("enable_function2_rescue", True),
            "max_iterative_rounds": opt_config.get("max_iterative_rounds", 5),
            "good_particles_threshold": opt_config.get("good_particles_threshold", 0.9)
        }
    
    def execute_2d_optimization(
        self,
        particles_job_uid: str,
        conversation_id: Optional[str] = None
    ) -> Optimization2DResult:
        """
        Execute 2D optimization workflow (Function 1: iterative, Function 2: rescue).
        
        Args:
            particles_job_uid: UID of the particles job from picking
            conversation_id: Optional conversation ID for logging
            
        Returns:
            Optimization2DResult with optimization results
        """
        # Check configuration flags
        workflow_config = self.stage_config.get("workflow", {})
        opt_config = workflow_config.get("2d_optimization", {})
        enable_f1 = opt_config.get("enable_function1_iterative", True)
        enable_f2 = opt_config.get("enable_function2_rescue", True)
        max_rounds = opt_config.get("max_iterative_rounds", 5)
        threshold = opt_config.get("good_particles_threshold", 0.9)
        threshold_pct = int(threshold * 100)
        
        # Update workflow defaults with input parameters
        workflow_defaults = {
            "particles_job_uid": particles_job_uid
        }
        if hasattr(self.agent, "update_workflow_defaults"):
            self.agent.update_workflow_defaults(workflow_defaults)
        
        # Prepare prompt based on enabled functions
        if enable_f1 and enable_f2:
            prompt = f"""Optimize particle selection through iterative 2D classification and rescue excluded particles.

I have particles from picking with job UID: {particles_job_uid}

Workflow:
1. Step A: Run initial 2D classification and select good classes using CryoSift
2. Step B (Function 2 - Rescue): Get excluded particles, run 2D classification on them, select good classes
3. Step C (Function 1 - Iterative): Call class_2d ONCE to start iterative refinement until ≥{threshold_pct}% good particles (max {max_rounds} rounds)

CRITICAL INSTRUCTIONS FOR STEP C:
- After Step B, you will have two select_2d jobs (J157 from Step A and J159 from Step B)
- For Step C, call class_2d ONCE with any one of the job UIDs (e.g., J157)
- The class_2d tool will AUTOMATICALLY detect and connect BOTH jobs in this SINGLE call
- DO NOT call class_2d multiple times - ONE call handles both jobs automatically
- DO NOT merge the jobs - the tool connects them directly without merging

Please execute the complete workflow and return the final particles_job_uid."""
        elif enable_f1:
            prompt = f"""Optimize particle selection through iterative 2D classification.

I have particles from picking with job UID: {particles_job_uid}

Workflow:
1. Step A: Run initial 2D classification and select good classes using CryoSift
2. Step C (Function 1 - Iterative): Iteratively refine until ≥{threshold_pct}% good particles (max {max_rounds} rounds)

Note: Function 2 (Rescue) is DISABLED. Skip Step B and go directly to iterative refinement.

Please execute the workflow and return the final particles_job_uid."""
        elif enable_f2:
            prompt = f"""Rescue good particles from excluded set.

I have particles from picking with job UID: {particles_job_uid}

Workflow:
1. Step A: Run initial 2D classification and select good classes using CryoSift
2. Step B (Function 2 - Rescue): Get excluded particles, run 2D classification on them, select good classes, and merge with good particles from Step A

Note: Function 1 (Iterative) is DISABLED. After Step B, return the merged particles.

Please execute the workflow and return the final particles_job_uid."""
        else:
            prompt = f"""I have particles from picking with job UID: {particles_job_uid}

Note: Both Function 1 (Iterative) and Function 2 (Rescue) are disabled.
Please report the current particle count from this job."""
        
        try:
            # Execute optimization using the agent
            result = self.agent.run_react_workflow(prompt, conversation_id=conversation_id)
            
            # Parse the result - look for tool executions to extract final state
            tool_execution_log = self.agent.get_tool_execution_log()
            
            # Track workflow execution
            workflow_summary = {
                "function1_enabled": enable_f1,
                "function2_enabled": enable_f2,
                "max_rounds": max_rounds,
                "threshold_percentage": threshold_pct,
                "steps_executed": []
            }
            
            # Find final particles job UID and calculate statistics
            final_particles_job_uid = None
            final_good_particles_count = None
            final_good_particles_percentage = None
            total_rounds = 0
            
            # Track rounds by counting class_2d + select_2d_classes pairs
            class_2d_count = 0
            select_2d_count = 0
            
            # Find the last select_2d_classes or merge_particles job
            for tool_exec in reversed(tool_execution_log):
                tool_name = tool_exec.get("tool")
                tool_result = tool_exec.get("result")
                
                if tool_name == "class_2d":
                    class_2d_count += 1
                elif tool_name == "select_2d_classes":
                    select_2d_count += 1
                    # Always update to the last (most recent) select_2d_classes result
                    if tool_result:
                        try:
                            if isinstance(tool_result, str):
                                result_data = json.loads(tool_result)
                            else:
                                result_data = tool_result
                            
                            if result_data.get("success") and result_data.get("job_uid"):
                                final_particles_job_uid = result_data.get("job_uid")
                                # Get particle count for selected particles
                                count_result = self._get_particle_count_from_log(
                                    tool_execution_log,
                                    final_particles_job_uid,
                                    "particles_selected"
                                )
                                if count_result:
                                    final_good_particles_count = count_result.get("num_particles")
                        except (json.JSONDecodeError, TypeError, ValueError):
                            pass
                
                elif tool_name == "merge_particles":
                    if tool_result:
                        try:
                            if isinstance(tool_result, str):
                                result_data = json.loads(tool_result)
                            else:
                                result_data = tool_result
                            
                            if result_data.get("success") and result_data.get("job_uid"):
                                final_particles_job_uid = result_data.get("job_uid")
                                # Get particle count for merged particles
                                count_result = self._get_particle_count_from_log(
                                    tool_execution_log,
                                    final_particles_job_uid,
                                    "particles"
                                )
                                if count_result:
                                    final_good_particles_count = count_result.get("num_particles")
                        except (json.JSONDecodeError, TypeError, ValueError):
                            pass
            
            # Calculate total rounds (each round = class_2d + select_2d_classes)
            # The minimum of class_2d_count and select_2d_count gives us the number of complete rounds
            total_rounds = min(class_2d_count, select_2d_count)
            
            # If we have a final particles job, try to get the count and calculate percentage
            if final_particles_job_uid:
                # Try to get particle count from the final job
                count_info = self.agent.cryosparc_tools.get_particle_count(
                    self.config.workflow.project_uid,
                    final_particles_job_uid,
                    "particles_selected" if "select" in final_particles_job_uid.lower() else "particles"
                )
                if count_info.get("success"):
                    final_good_particles_count = count_info.get("num_particles")
                    
                    # Try to get input particle count to calculate percentage
                    # The input count would be from the particles_job_uid or from previous round
                    # For now, we'll calculate percentage based on the current input
                    # This is a simplified calculation - in practice, the agent should track this
                    if final_good_particles_count:
                        # Get input count from the original particles_job_uid
                        input_count_info = self.agent.cryosparc_tools.get_particle_count(
                            self.config.workflow.project_uid,
                            particles_job_uid,
                            "particles"
                        )
                        if input_count_info.get("success"):
                            input_count = input_count_info.get("num_particles", 1)
                            final_good_particles_percentage = (final_good_particles_count / input_count) * 100 if input_count > 0 else 0
            
            # Build summary message
            if final_particles_job_uid:
                summary_msg = f"Final Good Particles: {final_good_particles_count or 'N/A'}"
                if final_good_particles_percentage is not None:
                    summary_msg += f" ({final_good_particles_percentage:.1f}% of current input)"
                summary_msg += f". Total Rounds: {total_rounds}."
                
                # Log the summary
                self.agent.logger.info(summary_msg)
                
                workflow_summary["final_particles_job_uid"] = final_particles_job_uid
                workflow_summary["final_good_particles_count"] = final_good_particles_count
                workflow_summary["final_good_particles_percentage"] = final_good_particles_percentage
                workflow_summary["total_rounds"] = total_rounds
                
                return Optimization2DResult(
                    step=Optimization2DStep.OPTIMIZE_2D,
                    success=True,
                    final_particles_job_uid=final_particles_job_uid,
                    final_good_particles_count=final_good_particles_count,
                    final_good_particles_percentage=final_good_particles_percentage,
                    total_rounds=total_rounds,
                    message=summary_msg,
                    workflow_summary=workflow_summary
                )
            else:
                # No final particles job found
                return Optimization2DResult(
                    step=Optimization2DStep.OPTIMIZE_2D,
                    success=False,
                    error="Could not determine final particles job from workflow execution",
                    message="2D optimization workflow executed but final particles job could not be determined",
                    workflow_summary=workflow_summary
                )
                
        except Exception as e:
            return Optimization2DResult(
                step=Optimization2DStep.OPTIMIZE_2D,
                success=False,
                error=str(e),
                message=f"Failed to execute 2D optimization: {str(e)}"
            )
    
    def _get_particle_count_from_log(
        self,
        tool_execution_log: List[Dict[str, Any]],
        job_uid: str,
        group_name: str = "particles"
    ) -> Optional[Dict[str, Any]]:
        """Helper to find particle count from tool execution log."""
        for tool_exec in tool_execution_log:
            if tool_exec.get("tool") == "get_particle_count":
                tool_result = tool_exec.get("result")
                if tool_result:
                    try:
                        if isinstance(tool_result, str):
                            result_data = json.loads(tool_result)
                        else:
                            result_data = tool_result
                        
                        if (result_data.get("success") and 
                            result_data.get("job_uid") == job_uid and
                            result_data.get("particles_group_name") == group_name):
                            return result_data
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass
        return None

