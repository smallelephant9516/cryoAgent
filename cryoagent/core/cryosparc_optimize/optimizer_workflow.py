"""ReAct-based box size optimization workflow orchestrator."""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from .optimizer_agent import OptimizerAgent
from ...config.config_loader import CryoAgentConfig
from ...prompts.prompt_loader import load_prompt


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
    # Multi-round 3D classification results
    best_multi_round_refinement_job_uid: Optional[str] = None
    best_multi_round_resolution: Optional[float] = None
    multi_round_rounds_completed: Optional[int] = None
    # Final homogeneous refinement
    final_refinement_job_uid: Optional[str] = None
    final_refinement_resolution: Optional[float] = None


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
        ab_initio_initial_resolution = optimization_config.get("ab_initio_initial_resolution", 9.0)
        ab_initio_final_resolution = optimization_config.get("ab_initio_final_resolution", 7.0)
        
        return {
            "max_iterations": max_iterations,
            "box_size_variation": box_size_variation,
            "ab_initio_initial_resolution": ab_initio_initial_resolution,
            "ab_initio_final_resolution": ab_initio_final_resolution
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
        enable_multi_round = optimization_config.get("enable_multi_round_3d_classification", False)
        multi_round_num_classes = optimization_config.get("multi_round_3d_classification_num_classes", 4)
        multi_round_max_rounds = optimization_config.get("multi_round_3d_classification_max_rounds", 5)
        multi_round_improvement_threshold = optimization_config.get("multi_round_3d_classification_improvement_threshold", 0.1)
        
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
        optimization_steps = []
        if enable_multi_round:
            optimization_steps.append("1. FIRST: Run multi-round 3D classification")
        if enable_hetero:
            optimization_steps.append("2. SECOND: Optimize heterogeneous refinement (K values)")
        if enable_box_size:
            optimization_steps.append("3. THIRD: Optimize box size/diameter")
        optimization_order = "\n".join(optimization_steps) if optimization_steps else ""

        task_context = {
            "refinement_job_uid": refinement_job_uid,
            "volume_job_uid": volume_job_uid,
            "particles_job_uid": particles_job_uid,
            "micrographs_job_uid": micrographs_job_uid,
            "optimization_order": optimization_order,
            "multi_round_num_classes": multi_round_num_classes,
            "multi_round_max_rounds": multi_round_max_rounds,
            "multi_round_improvement_threshold": multi_round_improvement_threshold,
        }

        if enable_box_size and enable_hetero and enable_multi_round:
            task_path = "cryosparc/optimization/task_all_three.md"
        elif enable_box_size and enable_hetero:
            task_path = "cryosparc/optimization/task_box_hetero.md"
        elif enable_box_size and enable_multi_round:
            task_path = "cryosparc/optimization/task_box_multi_round.md"
        elif enable_hetero and enable_multi_round:
            task_path = "cryosparc/optimization/task_hetero_multi_round.md"
        elif enable_box_size:
            task_path = "cryosparc/optimization/task_box_only.md"
        elif enable_hetero:
            task_path = "cryosparc/optimization/task_hetero_only.md"
        elif enable_multi_round:
            task_path = "cryosparc/optimization/task_multi_round_only.md"
        else:
            task_path = "cryosparc/optimization/task_none.md"

        prompt = load_prompt(task_path, task_context)
        
        try:
            # Execute optimization using the agent
            result = self.agent.run_react_workflow(prompt, conversation_id=conversation_id)
            
            # Parse the result. With the decomposed (atomic) optimization tools, the
            # LLM drives the box-size / hetero-K / multi-round recipes itself and
            # measures every candidate with get_fsc_info. So the source of truth for
            # "which refinement was best" is the set of get_fsc_info calls: each ties a
            # refinement_job_uid to its resolution_angstroms. The best = lowest resolution.
            tool_execution_log = self.agent.get_tool_execution_log()

            tested_combinations = []
            best_result = None
            best_resolution = float('inf')  # Lower is better
            multi_round_result = None  # retained for backward-compatible result shape

            for tool_exec in tool_execution_log:
                tool_name = tool_exec.get("tool")
                tool_result = tool_exec.get("result")

                if tool_name == "get_fsc_info" and tool_result:
                    try:
                        if isinstance(tool_result, str):
                            result_data = json.loads(tool_result)
                        else:
                            result_data = tool_result

                        if result_data.get("success"):
                            box_size = result_data.get("box_size")
                            resolution = result_data.get("resolution_angstroms")
                            job_uid = (
                                result_data.get("refinement_job_uid")
                                or result_data.get("job_uid")
                            )

                            if resolution is not None and job_uid is not None:
                                # De-duplicate by the measured refinement job UID.
                                if not any(c.get("job_uid") == job_uid for c in tested_combinations):
                                    tested_combinations.append({
                                        "box_size": box_size,
                                        "resolution": resolution,
                                        "job_uid": job_uid,
                                    })
                                    if resolution < best_resolution:
                                        best_resolution = resolution
                                        best_result = {
                                            "box_size": box_size,
                                            "resolution": resolution,
                                            "job_uid": job_uid,
                                        }
                    except (json.JSONDecodeError, TypeError, ValueError):
                        # Skip invalid results
                        continue
            
            # Return result based on what we found
            if best_result and tested_combinations:
                # Format message based on optimization type
                opt_type = best_result.get("type", "box_size")
                if opt_type == "heterogeneous_refinement":
                    message = f"Optimization completed successfully. Best K value: {best_result.get('k')} with resolution: {best_result.get('resolution'):.3f} Å"
                elif opt_type == "multi_round_3d_classification":
                    rounds = best_result.get("rounds_completed", 0)
                    improvement = best_result.get("improvement", 0)
                    initial_res = best_result.get("initial_resolution", 0)
                    message = f"Multi-round 3D classification completed successfully. Completed {rounds} rounds. Best resolution: {best_result.get('resolution'):.3f} Å (improved from {initial_res:.3f} Å by {improvement:.3f} Å)"
                else:
                    message = f"Optimization completed successfully. Best box size: {best_result.get('box_size')}px with resolution: {best_result.get('resolution'):.3f} Å"
                
                # Extract multi-round 3D classification results if available
                best_multi_round_job_uid = None
                best_multi_round_resolution = None
                multi_round_rounds = None
                if multi_round_result:
                    best_multi_round_job_uid = multi_round_result.get("best_refinement_job_uid")
                    best_multi_round_resolution = multi_round_result.get("best_resolution_angstroms")
                    multi_round_rounds = multi_round_result.get("rounds_completed")
                
                # Run final homogeneous refinement
                best_homogeneous_refinement_job_uid = best_result.get("job_uid")
                final_refinement_job_uid = None
                final_refinement_resolution = None
                
                if best_homogeneous_refinement_job_uid:
                    try:
                        # Get symmetry from agent
                        symmetry = self.agent._get_refinement_symmetry()
                        
                        # Get project and workspace UIDs from config
                        project_uid = self.config.workflow.project_uid
                        workspace_uid = self.config.workflow.workspace_uid
                        
                        # Log the final refinement step
                        self.agent.logger.info(f"🔧 Running final homogeneous refinement...")
                        self.agent.logger.info(f"   Source job: {best_homogeneous_refinement_job_uid}")
                        self.agent.logger.info(f"   Symmetry: {symmetry}")
                        
                        # Get initial lowpass resolution from optimization config if available
                        refine_res_init = None
                        if hasattr(self.agent, '_get_refinement_res_init'):
                            refine_res_init = self.agent._get_refinement_res_init()
                        
                        # Run homogeneous refinement
                        refine_result = self.agent.cryosparc_tools.homogeneous_refinement(
                            project_uid=project_uid,
                            workspace_uid=workspace_uid,
                            particles_job_uid=best_homogeneous_refinement_job_uid,
                            volume_job_uid=best_homogeneous_refinement_job_uid,
                            symmetry=symmetry,
                            refine_res_init=refine_res_init,  # Initial lowpass resolution if configured
                            wait_for_completion=True,
                            timeout=self.config.job_management.default_timeout,
                            check_interval=self.config.job_management.status_check_interval
                        )
                        
                        if refine_result.get("success", False):
                            final_refinement_job_uid = refine_result.get("job_uid")
                            refine_status = refine_result.get("status", "unknown")
                            
                            if refine_status == "completed" and final_refinement_job_uid:
                                # Get FSC resolution for the final refinement
                                fsc_info = self.agent.cryosparc_tools.get_refinement_fsc_info(
                                    project_uid, final_refinement_job_uid
                                )
                                if fsc_info.get("success"):
                                    final_refinement_resolution = fsc_info.get("resolution_angstroms")
                                    self.agent.logger.info(f"✅ Final homogeneous refinement completed successfully")
                                    self.agent.logger.info(f"   Final refinement job: {final_refinement_job_uid}")
                                    self.agent.logger.info(f"   Final resolution: {final_refinement_resolution:.3f} Å")
                                else:
                                    self.agent.logger.warning(f"⚠️  Final refinement completed but could not get FSC info: {fsc_info.get('error')}")
                            else:
                                self.agent.logger.warning(f"⚠️  Final refinement did not complete successfully: status={refine_status}")
                        else:
                            error_msg = refine_result.get("error") or "Unknown error"
                            self.agent.logger.error(f"❌ Final homogeneous refinement failed: {error_msg}")
                    except Exception as e:
                        self.agent.logger.error(f"❌ Error running final homogeneous refinement: {e}", exc_info=True)
                        # Continue with optimization result even if final refinement fails
                
                return OptimizationResult(
                    step=OptimizationStep.OPTIMIZE_DIAMETER,
                    success=True,
                    job_uid=best_result.get("job_uid"),
                    best_box_size=best_result.get("box_size"),  # May be None for hetero refinement or multi-round
                    best_resolution=best_result.get("resolution"),
                    tested_combinations=tested_combinations,
                    message=message,
                    best_multi_round_refinement_job_uid=best_multi_round_job_uid,
                    best_multi_round_resolution=best_multi_round_resolution,
                    multi_round_rounds_completed=multi_round_rounds,
                    final_refinement_job_uid=final_refinement_job_uid,
                    final_refinement_resolution=final_refinement_resolution
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

