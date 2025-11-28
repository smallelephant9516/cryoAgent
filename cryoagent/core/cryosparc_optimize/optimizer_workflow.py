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
    # Multi-round 3D classification results
    best_multi_round_refinement_job_uid: Optional[str] = None
    best_multi_round_resolution: Optional[float] = None
    multi_round_rounds_completed: Optional[int] = None


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
        # Build optimization order description
        # Order: 1. Multi-round 3D classification, 2. Heterogeneous refinement (K), 3. Box size
        optimization_steps = []
        if enable_multi_round:
            optimization_steps.append("1. FIRST: Run multi-round 3D classification")
        if enable_hetero:
            optimization_steps.append("2. SECOND: Optimize heterogeneous refinement (K values)")
        if enable_box_size:
            optimization_steps.append("3. THIRD: Optimize box size/diameter")
        
        optimization_order = "\n".join(optimization_steps) if optimization_steps else ""
        
        if enable_box_size and enable_hetero and enable_multi_round:
            prompt = f"""Optimize multi-round 3D classification, heterogeneous refinement (K values), and box size for 3D reconstruction.

I have completed the first round of homogeneous refinement with job UID: {refinement_job_uid}
The initial volume is from job: {volume_job_uid}
Particles can be re-extracted from picking job: {particles_job_uid}
Micrographs are available from job: {micrographs_job_uid}

Optimization order:
{optimization_order}

Use the best refinement job from each step as input for the next step."""
        elif enable_box_size and enable_hetero:
            prompt = f"""Optimize both heterogeneous refinement (K values) and box size for 3D reconstruction.

I have completed the first round of homogeneous refinement with job UID: {refinement_job_uid}
The initial volume is from job: {volume_job_uid}
Particles can be re-extracted from picking job: {particles_job_uid}
Micrographs are available from job: {micrographs_job_uid}

Please FIRST optimize the K value for heterogeneous refinement, then use the optimized refinement job for box size optimization."""
        elif enable_box_size and enable_multi_round:
            prompt = f"""Optimize multi-round 3D classification and box size for 3D reconstruction.

I have completed the first round of homogeneous refinement with job UID: {refinement_job_uid}
The initial volume is from job: {volume_job_uid}
Particles can be re-extracted from picking job: {particles_job_uid}
Micrographs are available from job: {micrographs_job_uid}

Optimization order:
{optimization_order}

Use the best refinement job from multi-round 3D classification as input for box size optimization."""
        elif enable_hetero and enable_multi_round:
            prompt = f"""Optimize multi-round 3D classification and heterogeneous refinement for 3D reconstruction.

I have completed homogeneous refinement with job UID: {refinement_job_uid}

IMPORTANT: Box size optimization is DISABLED. DO NOT use test_box_size tool.

Optimization order:
{optimization_order}

Use the best refinement job from multi-round 3D classification as input for heterogeneous refinement optimization."""
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
        elif enable_multi_round:
            prompt = f"""Run multi-round 3D classification for 3D reconstruction.

I have completed homogeneous refinement with job UID: {refinement_job_uid}

IMPORTANT: Box size optimization and heterogeneous refinement are DISABLED. DO NOT use test_box_size or test_heterogeneous_refinement tools.
Proceed directly to multi-round 3D classification using the refinement_job_uid provided ({refinement_job_uid}).
Use the test_multi_round_3d_classification tool with:
- num_classes: {multi_round_num_classes}
- max_rounds: {multi_round_max_rounds}
- improvement_threshold: {multi_round_improvement_threshold}

After completing multi-round 3D classification, you will proceed to heterogeneous refinement (K optimization) and then box size optimization."""
        else:
            prompt = f"""I have a refinement job with UID: {refinement_job_uid}

Note: Box size optimization, heterogeneous refinement, and multi-round 3D classification are all disabled.
Please report the current resolution from this refinement job."""
        
        try:
            # Execute optimization using the agent
            result = self.agent.run_react_workflow(prompt, conversation_id=conversation_id)
            
            # Parse the result - look for test_box_size, test_heterogeneous_refinement, test_multi_round_3d_classification, and get_fsc_info tool executions
            tool_execution_log = self.agent.get_tool_execution_log()
            
            # Collect all test results from test_box_size, test_heterogeneous_refinement, and test_multi_round_3d_classification tool executions
            tested_combinations = []
            best_result = None
            best_resolution = float('inf')  # Lower is better
            multi_round_result = None
            
            for tool_exec in tool_execution_log:
                tool_name = tool_exec.get("tool")
                tool_result = tool_exec.get("result")
                
                if tool_name == "test_multi_round_3d_classification" and tool_result:
                    # Handle multi-round 3D classification results
                    try:
                        if isinstance(tool_result, str):
                            result_data = json.loads(tool_result)
                        else:
                            result_data = tool_result
                        
                        if result_data.get("success"):
                            multi_round_result = {
                                "best_refinement_job_uid": result_data.get("best_refinement_job_uid"),
                                "best_resolution_angstroms": result_data.get("best_resolution_angstroms"),
                                "initial_resolution_angstroms": result_data.get("initial_resolution_angstroms"),
                                "total_improvement": result_data.get("total_improvement"),
                                "rounds_completed": result_data.get("rounds_completed"),
                                "all_rounds_data": result_data.get("all_rounds_data", []),
                                "type": "multi_round_3d_classification"
                            }
                            
                            resolution = multi_round_result.get("best_resolution_angstroms")
                            if resolution is not None:
                                tested_combinations.append({
                                    "type": "multi_round_3d_classification",
                                    "resolution": resolution,
                                    "job_uid": multi_round_result.get("best_refinement_job_uid"),
                                    "rounds_completed": multi_round_result.get("rounds_completed"),
                                    "initial_resolution": multi_round_result.get("initial_resolution_angstroms"),
                                    "improvement": multi_round_result.get("total_improvement")
                                })
                                
                                # Track best result (lowest resolution)
                                if resolution < best_resolution:
                                    best_resolution = resolution
                                    best_result = {
                                        "resolution": resolution,
                                        "job_uid": multi_round_result.get("best_refinement_job_uid"),
                                        "type": "multi_round_3d_classification",
                                        "rounds_completed": multi_round_result.get("rounds_completed"),
                                        "initial_resolution": multi_round_result.get("initial_resolution_angstroms"),
                                        "improvement": multi_round_result.get("total_improvement")
                                    }
                    except (json.JSONDecodeError, TypeError, ValueError) as e:
                        # Skip invalid results
                        continue
                
                elif tool_name == "test_heterogeneous_refinement" and tool_result:
                    # Handle heterogeneous refinement optimization results
                    try:
                        if isinstance(tool_result, str):
                            result_data = json.loads(tool_result)
                        else:
                            result_data = tool_result
                        
                        if result_data.get("success"):
                            k = result_data.get("k")
                            resolution = result_data.get("final_resolution_angstroms")
                            refine_job_uid = result_data.get("refine_job_uid")
                            
                            if k is not None and resolution is not None:
                                tested_combinations.append({
                                    "k": k,
                                    "resolution": resolution,
                                    "job_uid": refine_job_uid,
                                    "type": "heterogeneous_refinement"
                                })
                                
                                # Track best result (lowest resolution)
                                if resolution < best_resolution:
                                    best_resolution = resolution
                                    best_result = {
                                        "k": k,
                                        "resolution": resolution,
                                        "job_uid": refine_job_uid,
                                        "type": "heterogeneous_refinement"
                                    }
                    except (json.JSONDecodeError, TypeError, ValueError) as e:
                        # Skip invalid results
                        continue
                
                elif tool_name == "test_box_size" and tool_result:
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
                    multi_round_rounds_completed=multi_round_rounds
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

