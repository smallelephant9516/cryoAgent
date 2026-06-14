"""ReAct-based 2D classification optimization workflow orchestrator."""

import json
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from .optimizer_2d_agent import Optimizer2DAgent
from ...config.config_loader import CryoAgentConfig
from ...prompts.prompt_loader import load_prompt


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
        
        # Select minimal task prompt (system manual lives in system.md)
        if enable_f1 and enable_f2:
            task_path = "cryosparc/optimization_2d/task_both.md"
        elif enable_f1:
            task_path = "cryosparc/optimization_2d/task_f1_only.md"
        elif enable_f2:
            task_path = "cryosparc/optimization_2d/task_f2_only.md"
        else:
            task_path = "cryosparc/optimization_2d/task_none.md"

        prompt = load_prompt(
            task_path,
            {
                "particles_job_uid": particles_job_uid,
                "threshold_pct": threshold_pct,
                "max_rounds": max_rounds,
            },
        )
        
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
            
            # First, try to extract final particles job UID from LLM's response text
            # The LLM often explicitly states the final job UID (e.g., "Final particles_job_uid: J521")
            if result:
                extracted_uid = self._extract_final_job_uid_from_text(result)
                if extracted_uid:
                    final_particles_job_uid = extracted_uid
                    self.agent.logger.info(f"Extracted final particles job UID from LLM response: {final_particles_job_uid}")
                    # Try to get particle count for the extracted job UID
                    count_result = self._get_particle_count_from_log(
                        tool_execution_log,
                        final_particles_job_uid,
                        "particles_selected"
                    )
                    if count_result:
                        final_good_particles_count = count_result.get("num_particles")
            
            # Track rounds by counting class_2d + select_2d_classes pairs
            class_2d_count = 0
            select_2d_count = 0
            
            # First, count all jobs by iterating forward
            for tool_exec in tool_execution_log:
                tool_name = tool_exec.get("tool")
                if tool_name == "class_2d":
                    class_2d_count += 1
                elif tool_name == "select_2d_classes":
                    select_2d_count += 1
            
            # Then, find the last (most recent) select_2d_classes or merge_particles job
            # Iterate in reverse to find the most recent one first
            for tool_exec in reversed(tool_execution_log):
                tool_name = tool_exec.get("tool")
                tool_result = tool_exec.get("result")
                
                if tool_name == "select_2d_classes":
                    # Always update to the last (most recent) select_2d_classes result
                    # Since we iterate in reverse, the first valid one we find is the most recent
                    if tool_result and final_particles_job_uid is None:
                        try:
                            if isinstance(tool_result, str):
                                result_data = json.loads(tool_result)
                            else:
                                result_data = tool_result
                            
                            if result_data.get("success") and result_data.get("job_uid"):
                                # Only set if we haven't found a final job yet
                                # Since we iterate in reverse, first valid result is most recent
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
                    if tool_result and final_particles_job_uid is None:
                        try:
                            if isinstance(tool_result, str):
                                result_data = json.loads(tool_result)
                            else:
                                result_data = tool_result
                            
                            if result_data.get("success") and result_data.get("job_uid"):
                                # Only set if we haven't found a final job yet
                                # Since we iterate in reverse, first valid result is most recent
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
            
            # Check if we should run select_2d to select all particles after the last round
            enable_select_all = opt_config.get("enable_select_all_after_last_round", False)
            if enable_select_all and class_2d_count > 0:
                # Find the last class_2d job from the execution log
                last_class_2d_job_uid = None
                for tool_exec in reversed(tool_execution_log):
                    if tool_exec.get("tool") == "class_2d":
                        tool_result = tool_exec.get("result")
                        if tool_result:
                            try:
                                if isinstance(tool_result, str):
                                    result_data = json.loads(tool_result)
                                else:
                                    result_data = tool_result
                                
                                if result_data.get("success") and result_data.get("job_uid"):
                                    last_class_2d_job_uid = result_data.get("job_uid")
                                    break
                            except (json.JSONDecodeError, TypeError, ValueError):
                                pass
                
                if last_class_2d_job_uid:
                    try:
                        self.agent.logger.info(
                            f"Option 'enable_select_all_after_last_round' is enabled. "
                            f"Running select_2d job to select all particles from the last 2D classification job {last_class_2d_job_uid}"
                        )
                        
                        # Run select_2d with "all" selection mode to select all particles
                        select_result = self.agent.cryosparc_tools.select_2d_classes(
                            project_uid=self.config.workflow.project_uid,
                            workspace_uid=self.config.workflow.workspace_uid,
                            class_2d_job_uid=last_class_2d_job_uid,
                            selection_mode="all",
                            wait_for_completion=True,
                            timeout=self.config.job_management.default_timeout,
                            check_interval=self.config.job_management.status_check_interval
                        )
                        
                        if select_result.get("job_uid"):
                            # Update final particles job to be this new select_2d job
                            final_particles_job_uid = select_result.get("job_uid")
                            self.agent.logger.info(
                                f"Successfully created select_2d job {final_particles_job_uid} "
                                f"to select all particles from the last round"
                            )
                            
                            # Update workflow summary
                            workflow_summary["select_all_after_last_round"] = True
                            workflow_summary["select_all_job_uid"] = final_particles_job_uid
                            workflow_summary["select_all_source_job_uid"] = last_class_2d_job_uid
                        else:
                            error_msg = select_result.get("error", "Unknown error")
                            self.agent.logger.warning(
                                f"Failed to create select_2d job to select all particles: {error_msg}"
                            )
                    except Exception as e:
                        self.agent.logger.error(
                            f"Error running select_2d to select all particles after last round: {e}"
                        )
            
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
    
    def _extract_final_job_uid_from_text(self, text: str) -> Optional[str]:
        """
        Extract the final particles job UID from LLM's response text.
        
        The LLM often explicitly states the final job UID in patterns like:
        
        Args:
            text: The LLM's response text
            
        Returns:
            Job UID if found, None otherwise
        """
        if not text:
            return None
        
        # Patterns to match job UIDs in various formats
        patterns = [
            r'(?:Final|final)\s+particles[_\s]?job[_\s]?uid[:\s]+(J\d+)',
            r'(?:Final|final)\s+particles[:\s]+(J\d+)',
            r'(?:Final|final)\s+job[:\s]+(J\d+)',
            r'(?:Final|final)\s+result[:\s]+(J\d+)',
            r'particles[_\s]?job[_\s]?uid[:\s]+(J\d+)',  # More general pattern
            r'final[_\s]?particles[_\s]?job[_\s]?uid[:\s]+(J\d+)',
        ]
        
        # Try each pattern
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                job_uid = match.group(1)
                # Validate it's a proper job UID format (J followed by digits)
                if re.match(r'^J\d+$', job_uid):
                    return job_uid
        
        # Also look for job UIDs mentioned near "final" keywords
        # Find all job UIDs in the text
        all_job_uids = re.findall(r'\b(J\d+)\b', text)
        if all_job_uids:
            # Look for the last job UID mentioned near "final" keywords
            # Split text into sentences and find ones with "final"
            sentences = re.split(r'[.!?\n]', text)
            for sentence in reversed(sentences):  # Check from end
                if re.search(r'\bfinal\b', sentence, re.IGNORECASE):
                    # Extract job UID from this sentence
                    job_match = re.search(r'\b(J\d+)\b', sentence)
                    if job_match:
                        return job_match.group(1)
            
            # If no "final" keyword found, return the last job UID mentioned
            # (assuming the LLM mentions jobs in chronological order)
            return all_job_uids[-1]
        
        return None
    
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

