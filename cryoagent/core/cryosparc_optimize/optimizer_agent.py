"""ReAct-based box size optimization agent for CryoEM 3D reconstruction."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain.tools import Tool
from langchain_core.language_models import BaseLanguageModel

from ..base_react_agent import BaseReActAgent
from .optimizer_tools import OptimizerTools
from ...tools.cryosparc_tools import CryoSPARCTools
from ...config.config_loader import CryoAgentConfig


class OptimizerAgent(BaseReActAgent):
    """ReAct-based agent for optimizing box size/diameter in CryoEM 3D reconstruction."""
    
    def __init__(
        self,
        cryosparc_tools: CryoSPARCTools,
        config: CryoAgentConfig,
        llm: Optional[BaseLanguageModel] = None
    ):
        """
        Initialize the box size optimization agent.
        
        Args:
            cryosparc_tools: CryoSPARC tools instance
            config: Complete configuration object
            llm: Language model for the agent
        """
        # Initialize stage_config BEFORE calling super().__init__() because
        # BaseReActAgent.__init__() calls _create_tools() which may access stage_config
        self.workflow_defaults: Dict[str, Any] = {}
        self.stage_config = self._load_stage_config()
        self.stage_workflow = self.stage_config.get("workflow", {})
        
        # Now call super().__init__() which will call _create_tools()
        super().__init__(cryosparc_tools, config, llm)
        
        # Initialize logger for this agent
        self.logger = logging.getLogger("OptimizerAgent")
    
    def _create_tools(self) -> List[Tool]:
        """Create optimization-specific tools."""
        return [
            OptimizerTools.create_optimize_diameter_tool(self),
            OptimizerTools.create_get_job_status_tool(self),
            OptimizerTools.create_wait_for_job_tool(self),
            OptimizerTools.create_get_job_log_tool(self),
            OptimizerTools.create_reason_about_workflow_tool(self)
        ]
    
    def _load_stage_config(self) -> Dict[str, Any]:
        """Load optimization stage configuration."""
        config_path = Path("configs/cryosparc/optimization_config.json")
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        try:
            with open(config_path, "r", encoding="utf-8") as fp:
                return json.load(fp) or {}
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}
    
    def _load_reconstruction_config(self) -> Dict[str, Any]:
        """Load reconstruction stage configuration to get symmetry settings."""
        config_path = Path("configs/cryosparc/reconstruction_config.json")
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        try:
            with open(config_path, "r", encoding="utf-8") as fp:
                return json.load(fp) or {}
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}
    
    def _get_refinement_symmetry(self) -> str:
        """Get symmetry from reconstruction_config.json for homogeneous refinement."""
        recon_config = self._load_reconstruction_config()
        
        # First try to get from workflow.refinement.symmetry (preferred)
        refinement_symmetry = recon_config.get("workflow", {}).get("refinement", {}).get("symmetry")
        if refinement_symmetry:
            return refinement_symmetry
        
        # Fall back to microscope_parameters.symmetry
        microscope_symmetry = recon_config.get("microscope_parameters", {}).get("symmetry")
        if microscope_symmetry:
            return microscope_symmetry
        
        # Default to C1 if not found
        return "C1"
    
    def _get_stage_param(self, section: str, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """Fetch a parameter from the stage workflow configuration."""
        return self.stage_workflow.get(section, {}).get(key, default)
    
    def _get_react_system_prompt(self) -> str:
        """Get the optimization-specific ReAct system prompt."""
        return f"""You are a CryoEM box size optimization assistant using the ReAct (Reasoning + Acting) framework. 
You specialize in optimizing box size/diameter for 3D reconstruction by testing different box sizes and comparing FSC resolutions.

## ReAct Framework Rules:
1. **REASONING**: Always think through the problem step by step before taking action
2. **ACTING**: Execute specific tools based on your reasoning
3. **OBSERVING**: Analyze the results and update your understanding

## Box Size Optimization Workflow:

**Purpose**: After the first round of 3D homogeneous refinement, optimize the box size to achieve the best resolution.

**Process**:
1. Read the original box size and FSC resolution from the first refinement job
2. Test with 10% less and 10% more box sizes
3. Extract particles with new box sizes
4. Run homogeneous refinement with each new box size, and map from the output volume in the reconstruction job
5. Compare FSC resolutions from all tests
6. Iteratively refine the box size until optimal resolution is found

**Optimization Algorithm**:
- Start with original box size from first refinement
- Test 10% less and 10% more box sizes
- If middle is best: choose value in between
- If side is better: test another 10% more/less on that side
- Stop if >5 diameters tested or new box size equals original

## Tool Usage:

- **optimize_diameter**: Main optimization tool that automatically:
  * Reads FSC info from refinement job
  * Tests different box sizes
  * Extracts particles and runs refinement
  * Compares results and finds optimal box size
  * Required: refinement_job_uid (first refinement), particles_job_uid (picking job), 
    micrographs_job_uid, volume_job_uid
  * Returns best box size, resolution, and tested combinations

- **get_job_status**: Check status of a specific job (use job UID only, e.g., "J113")
- **wait_for_job**: Wait for job completion (use job UID only, e.g., "J113")
- **get_job_log**: Read and analyze job logs
- **reason_about_workflow**: Analyze current optimization state

## Job UID Format:
- Job UIDs are strings like "J113", "J114", etc.
- When calling get_job_status or wait_for_job, pass ONLY the job UID

## Current Configuration:
- Project UID: {self.config.workflow.project_uid}
- Workspace UID: {self.config.workflow.workspace_uid}

## Example Workflow:

1. Get refinement_job_uid from previous reconstruction stage
2. Get particles_job_uid (picking job for re-extraction)
3. Get micrographs_job_uid (for re-extraction)
4. Get volume_job_uid (initial volume from reconstruction)
5. Run optimize_diameter tool with all required parameters
6. Wait for optimization to complete
7. Review results and best box size/resolution

Remember: Always follow the Thought → Action → Observation pattern!
Box size optimization can take significant time as it runs multiple refinement jobs."""
    
    def update_workflow_defaults(self, defaults: Dict[str, Any]) -> None:
        """Store workflow-level default parameters for later tool invocations."""
        if defaults:
            if not hasattr(self, "workflow_defaults") or self.workflow_defaults is None:
                self.workflow_defaults = {}
            self.workflow_defaults.update(defaults)
    
    # =================================================================
    # Tool Implementation Methods
    # =================================================================
    
    def _optimize_diameter_tool(self, tool_input: str) -> str:
        """
        Optimize box size/diameter by testing different box sizes and comparing FSC resolutions.
        
        This tool implements an iterative optimization algorithm:
        1. After first homogeneous refinement, test with 10% less and 10% more box sizes
        2. Extract particles with new box sizes and run homogeneous refinement
        3. Compare three results (original, -10%, +10%)
        4. If middle is best, choose value in between
        5. If side is better, test another 10% more/less on that side
        6. Stop if >5 diameters tested or new box size equals original
        """
        try:
            params = self._parse_tool_input(tool_input)
            
            # Extract required parameters
            refinement_job_uid = params.get("refinement_job_uid")
            particles_job_uid = params.get("particles_job_uid")
            micrographs_job_uid = params.get("micrographs_job_uid")
            volume_job_uid = params.get("volume_job_uid")
            
            # Try to get from workflow defaults if not provided
            defaults = getattr(self, "workflow_defaults", {}) or {}
            if not particles_job_uid:
                # Try to get picking job (for re-extraction) from various possible keys
                particles_job_uid = (
                    defaults.get("picking_job_uid")
                    or defaults.get("blob_picker_job_uid")
                    or defaults.get("particle_picking_job_uid")
                    or defaults.get("particles_job_uid")  # Fallback to extracted particles
                    or defaults.get("selected_particles_job_uid")
                )
            if not micrographs_job_uid:
                micrographs_job_uid = (
                    defaults.get("micrographs_job_uid")
                    or defaults.get("micrograph_selection_job_uid")
                    or defaults.get("final_micrographs_job_uid")
                )
            if not volume_job_uid:
                volume_job_uid = (
                    defaults.get("volume_job_uid")
                    or defaults.get("last_volume_job_uid")
                    or defaults.get("ab_init_job_uid")
                    or defaults.get("homogeneous_reconstruction_job_uid")
                )
            
            if not all([refinement_job_uid, particles_job_uid, micrographs_job_uid, volume_job_uid]):
                missing = []
                if not refinement_job_uid:
                    missing.append("refinement_job_uid")
                if not particles_job_uid:
                    missing.append("particles_job_uid")
                if not micrographs_job_uid:
                    missing.append("micrographs_job_uid")
                if not volume_job_uid:
                    missing.append("volume_job_uid")
                return json.dumps({
                    "success": False,
                    "error": f"Missing required parameters: {', '.join(missing)}"
                })
            
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            
            # Get original FSC info
            fsc_info = self.cryosparc_tools.get_refinement_fsc_info(project_uid, refinement_job_uid)
            if not fsc_info.get("success"):
                return json.dumps({
                    "success": False,
                    "error": f"Failed to get FSC info from refinement job: {fsc_info.get('error')}"
                })
            
            original_box_size = fsc_info["box_size"]
            original_resolution = fsc_info["resolution_angstroms"]
            
            # Track tested box sizes and their resolutions
            tested_data = [
                {"box_size": original_box_size, "resolution": original_resolution, "job_uid": refinement_job_uid}
            ]
            
            max_iterations = 5
            iteration = 0
            initial_three_tested = False
            
            while iteration < max_iterations:
                iteration += 1
                
                # First iteration: test all three box sizes (original, -10%, +10%)
                # Note: Original is already tested from initial refinement, so we only test -10% and +10%
                if not initial_three_tested:
                    box_size_less = int(original_box_size * 0.9)
                    box_size_more = int(original_box_size * 1.1)
                    
                    # Normalize to allowed box sizes
                    box_size_less = self._normalize_box_size(box_size_less)
                    box_size_more = self._normalize_box_size(box_size_more)
                    
                    # Skip if same as original (shouldn't happen, but check anyway)
                    if box_size_less == original_box_size:
                        box_size_less = None
                    if box_size_more == original_box_size:
                        box_size_more = None
                    
                    # Skip if already tested (shouldn't happen in first iteration, but check)
                    if box_size_less and box_size_less in [d["box_size"] for d in tested_data]:
                        box_size_less = None
                    if box_size_more and box_size_more in [d["box_size"] for d in tested_data]:
                        box_size_more = None
                    
                    initial_three_tested = True
                    self.logger.info(f"First iteration: Testing box sizes -10%={box_size_less}, +10%={box_size_more} (original={original_box_size} already tested)")
                else:
                    # Subsequent iterations: determine next box size based on best result
                    # Sort by box size to find positions
                    tested_data_sorted = sorted(tested_data, key=lambda x: x["box_size"])
                    
                    # Find best resolution (lower is better)
                    best_idx = min(range(len(tested_data_sorted)), key=lambda i: tested_data_sorted[i]["resolution"])
                    best_box_size = tested_data_sorted[best_idx]["box_size"]
                    best_resolution = tested_data_sorted[best_idx]["resolution"]
                    
                    # Determine next box size based on which is best
                    next_box_size = None
                    
                    # Find position of best in sorted list
                    if len(tested_data_sorted) >= 3:
                        # Check if best is in the middle
                        middle_idx = len(tested_data_sorted) // 2
                        if best_idx == middle_idx:
                            # Middle is best, choose value between middle and better side
                            # Compare left and right sides
                            if best_idx > 0 and best_idx < len(tested_data_sorted) - 1:
                                left_res = tested_data_sorted[best_idx - 1]["resolution"]
                                right_res = tested_data_sorted[best_idx + 1]["resolution"]
                                
                                if left_res < right_res:
                                    # Left is better, choose between left and middle
                                    next_box_size = int((tested_data_sorted[best_idx - 1]["box_size"] + best_box_size) / 2)
                                else:
                                    # Right is better, choose between middle and right
                                    next_box_size = int((best_box_size + tested_data_sorted[best_idx + 1]["box_size"]) / 2)
                        elif best_idx == 0:
                            # Leftmost (smallest) is best, test 10% less
                            next_box_size = int(best_box_size * 0.9)
                        elif best_idx == len(tested_data_sorted) - 1:
                            # Rightmost (largest) is best, test 10% more
                            next_box_size = int(best_box_size * 1.1)
                        else:
                            # Best is somewhere in between, move towards it
                            if best_idx < middle_idx:
                                # Best is on left side, test 10% less
                                next_box_size = int(best_box_size * 0.9)
                            else:
                                # Best is on right side, test 10% more
                                next_box_size = int(best_box_size * 1.1)
                    else:
                        # Less than 3 data points, continue in direction of best
                        if best_idx == 0:
                            next_box_size = int(best_box_size * 0.9)
                        else:
                            next_box_size = int(best_box_size * 1.1)
                    
                    if next_box_size:
                        next_box_size = self._normalize_box_size(next_box_size)
                        
                        # Check if already tested
                        if next_box_size in [d["box_size"] for d in tested_data]:
                            # Already tested, stop
                            break
                        
                        box_size_less = None
                        box_size_more = None
                        
                        # Set the appropriate variable based on direction
                        if next_box_size < best_box_size:
                            box_size_less = next_box_size
                        else:
                            box_size_more = next_box_size
                    else:
                        # No next box size determined, stop
                        break
                
                # If no box sizes to test, stop
                if not box_size_less and not box_size_more:
                    break
                
                # Test box_size_less
                if box_size_less:
                    # Check if already tested
                    if box_size_less in [d["box_size"] for d in tested_data]:
                        self.logger.info(f"Skipping box_size_less {box_size_less} - already tested")
                    else:
                        try:
                            self.logger.info(f"🔬 Testing box_size_less: {box_size_less}")
                            
                            # Step 1: Extract particles with new box size
                            self.logger.info(f"📦 Step 1/3: Extracting particles with box_size {box_size_less}...")
                            extract_params = {
                                "project_uid": project_uid,
                                "workspace_uid": workspace_uid,
                                "particles_job_uid": particles_job_uid,
                                "micrographs_job_uid": micrographs_job_uid,
                                "box_size_pix": box_size_less
                            }
                            self._record_tool_execution("extract_particles", extract_params)
                            extract_result = self.cryosparc_tools.extract_particles(
                                **extract_params,
                                wait_for_completion=True,
                                timeout=self.config.job_management.default_timeout,
                                check_interval=self.config.job_management.status_check_interval
                            )
                            self._record_tool_execution("extract_particles", extract_params, result=extract_result)
                            
                            # Verify extraction completed successfully
                            extract_status = extract_result.get("status", "unknown")
                            if extract_status != "completed":
                                error_msg = extract_result.get("error") or f"Status: {extract_status}"
                                self.logger.error(f"❌ Particle extraction failed for box_size {box_size_less}: {error_msg}")
                                # Don't proceed to refinement if extraction failed
                                continue
                            
                            extract_job_uid = extract_result["job_uid"]
                            self.logger.info(f"✅ Step 1/3: Extraction completed for box_size {box_size_less}, job: {extract_job_uid}")
                            
                            # Step 2: Run homogeneous refinement (wait for completion)
                            self.logger.info(f"🔧 Step 2/3: Starting refinement for box_size {box_size_less}...")
                            # Get symmetry from reconstruction_config.json
                            symmetry = self._get_refinement_symmetry()
                            refine_params = {
                                "project_uid": project_uid,
                                "workspace_uid": workspace_uid,
                                "particles_job_uid": extract_job_uid,
                                "volume_job_uid": volume_job_uid,
                                "symmetry": symmetry
                            }
                            self._record_tool_execution("homogeneous_refinement", refine_params)
                            refine_result = self.cryosparc_tools.homogeneous_refinement(
                                **refine_params,
                                wait_for_completion=True,
                                timeout=self.config.job_management.default_timeout,
                                check_interval=self.config.job_management.status_check_interval
                            )
                            self._record_tool_execution("homogeneous_refinement", refine_params, result=refine_result)
                            
                            # Verify refinement completed successfully
                            # homogeneous_refinement returns "success" and updates with status_result
                            if not refine_result.get("success", False):
                                error_msg = refine_result.get("error") or "Unknown error"
                                self.logger.error(f"❌ Refinement failed for box_size {box_size_less}: {error_msg}")
                                # Don't proceed to FSC extraction if refinement failed
                                continue
                            
                            refine_status = refine_result.get("status", "unknown")
                            if refine_status != "completed":
                                error_msg = refine_result.get("error") or f"Status: {refine_status}"
                                self.logger.error(f"❌ Refinement did not complete for box_size {box_size_less}: {error_msg}")
                                # Don't proceed to FSC extraction if refinement didn't complete
                                continue
                            
                            refine_job_uid = refine_result["job_uid"]
                            self.logger.info(f"✅ Step 2/3: Refinement completed for box_size {box_size_less}, job: {refine_job_uid}")
                            
                            # Step 3: Get FSC info and resolution
                            self.logger.info(f"📊 Step 3/3: Getting FSC resolution for box_size {box_size_less}...")
                            fsc_info_less = self.cryosparc_tools.get_refinement_fsc_info(project_uid, refine_job_uid)
                            if fsc_info_less.get("success"):
                                resolution = fsc_info_less["resolution_angstroms"]
                                self.logger.info(f"✅ Step 3/3: Box size {box_size_less}: Resolution = {resolution} Å")
                                tested_data.append({
                                    "box_size": box_size_less,
                                    "resolution": resolution,
                                    "job_uid": refine_job_uid
                                })
                                self.logger.info(f"✅ Completed testing box_size {box_size_less}: Resolution = {resolution} Å")
                            else:
                                self.logger.error(f"❌ Failed to get FSC info for box_size {box_size_less}: {fsc_info_less.get('error')}")
                        except Exception as e:
                            # Continue with other tests even if one fails
                            self.logger.warning(f"⚠️ Failed to test box_size_less {box_size_less}: {e}", exc_info=True)
                            pass
                
                # Test box_size_more
                if box_size_more:
                    # Check if already tested
                    if box_size_more in [d["box_size"] for d in tested_data]:
                        self.logger.info(f"Skipping box_size_more {box_size_more} - already tested")
                    else:
                        try:
                            self.logger.info(f"🔬 Testing box_size_more: {box_size_more}")
                            
                            # Step 1: Extract particles with new box size
                            self.logger.info(f"📦 Step 1/3: Extracting particles with box_size {box_size_more}...")
                            extract_params = {
                                "project_uid": project_uid,
                                "workspace_uid": workspace_uid,
                                "particles_job_uid": particles_job_uid,
                                "micrographs_job_uid": micrographs_job_uid,
                                "box_size_pix": box_size_more
                            }
                            self._record_tool_execution("extract_particles", extract_params)
                            extract_result = self.cryosparc_tools.extract_particles(
                                **extract_params,
                                wait_for_completion=True,
                                timeout=self.config.job_management.default_timeout,
                                check_interval=self.config.job_management.status_check_interval
                            )
                            self._record_tool_execution("extract_particles", extract_params, result=extract_result)
                            
                            # Verify extraction completed successfully
                            extract_status = extract_result.get("status", "unknown")
                            if extract_status != "completed":
                                error_msg = extract_result.get("error") or f"Status: {extract_status}"
                                self.logger.error(f"❌ Particle extraction failed for box_size {box_size_more}: {error_msg}")
                                # Don't proceed to refinement if extraction failed
                                continue
                            
                            extract_job_uid = extract_result["job_uid"]
                            self.logger.info(f"✅ Step 1/3: Extraction completed for box_size {box_size_more}, job: {extract_job_uid}")
                            
                            # Step 2: Run homogeneous refinement (wait for completion)
                            self.logger.info(f"🔧 Step 2/3: Starting refinement for box_size {box_size_more}...")
                            # Get symmetry from reconstruction_config.json
                            symmetry = self._get_refinement_symmetry()
                            refine_params = {
                                "project_uid": project_uid,
                                "workspace_uid": workspace_uid,
                                "particles_job_uid": extract_job_uid,
                                "volume_job_uid": volume_job_uid,
                                "symmetry": symmetry
                            }
                            self._record_tool_execution("homogeneous_refinement", refine_params)
                            refine_result = self.cryosparc_tools.homogeneous_refinement(
                                **refine_params,
                                wait_for_completion=True,
                                timeout=self.config.job_management.default_timeout,
                                check_interval=self.config.job_management.status_check_interval
                            )
                            self._record_tool_execution("homogeneous_refinement", refine_params, result=refine_result)
                            
                            # Verify refinement completed successfully
                            # homogeneous_refinement returns "success" and updates with status_result
                            if not refine_result.get("success", False):
                                error_msg = refine_result.get("error") or "Unknown error"
                                self.logger.error(f"❌ Refinement failed for box_size {box_size_more}: {error_msg}")
                                # Don't proceed to FSC extraction if refinement failed
                                continue
                            
                            refine_status = refine_result.get("status", "unknown")
                            if refine_status != "completed":
                                error_msg = refine_result.get("error") or f"Status: {refine_status}"
                                self.logger.error(f"❌ Refinement did not complete for box_size {box_size_more}: {error_msg}")
                                # Don't proceed to FSC extraction if refinement didn't complete
                                continue
                            
                            refine_job_uid = refine_result["job_uid"]
                            self.logger.info(f"✅ Step 2/3: Refinement completed for box_size {box_size_more}, job: {refine_job_uid}")
                            
                            # Step 3: Get FSC info and resolution
                            self.logger.info(f"📊 Step 3/3: Getting FSC resolution for box_size {box_size_more}...")
                            fsc_info_more = self.cryosparc_tools.get_refinement_fsc_info(project_uid, refine_job_uid)
                            if fsc_info_more.get("success"):
                                resolution = fsc_info_more["resolution_angstroms"]
                                self.logger.info(f"✅ Step 3/3: Box size {box_size_more}: Resolution = {resolution} Å")
                                tested_data.append({
                                    "box_size": box_size_more,
                                    "resolution": resolution,
                                    "job_uid": refine_job_uid
                                })
                                self.logger.info(f"✅ Completed testing box_size {box_size_more}: Resolution = {resolution} Å")
                            else:
                                self.logger.error(f"❌ Failed to get FSC info for box_size {box_size_more}: {fsc_info_more.get('error')}")
                        except Exception as e:
                            # Continue with other tests even if one fails
                            self.logger.warning(f"⚠️ Failed to test box_size_more {box_size_more}: {e}", exc_info=True)
                            pass
                
                # After testing, sort by box size for next iteration's analysis
                tested_data.sort(key=lambda x: x["box_size"])
            
            # Find final best result
            best_result = min(tested_data, key=lambda x: x["resolution"])
            
            result = {
                "success": True,
                "optimization_complete": True,
                "best_box_size": best_result["box_size"],
                "best_resolution_angstroms": best_result["resolution"],
                "best_job_uid": best_result["job_uid"],
                "tested_combinations": tested_data,
                "iterations": iteration
            }
            
            self._record_tool_execution("optimize_diameter", params, result=result)
            return json.dumps(result)
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("optimize_diameter", params if 'params' in locals() else {}, error=str(e))
            return json.dumps(error_result)
    
    def _reason_about_workflow_tool(self, input_str: str) -> str:
        """Tool for reasoning about optimization workflow state."""
        try:
            reasoning = f"""
🤔 **Box Size Optimization Workflow Analysis**:

**Current State**: {input_str}

**Optimization Process**:
1. Read original box size and FSC resolution from first refinement
2. Test with 10% less and 10% more box sizes
3. Extract particles with new box sizes
4. Run homogeneous refinement for each
5. Compare FSC resolutions
6. Iteratively refine until optimal box size found

**Parameters**:
- **refinement_job_uid**: First homogeneous refinement job (required)
- **particles_job_uid**: Picking job for re-extraction (required)
- **micrographs_job_uid**: Micrographs for re-extraction (required)
- **volume_job_uid**: Initial volume from reconstruction (required)

**Optimization Strategy**:
- Tests up to 5 different box sizes
- Stops if optimal found or max iterations reached
- Returns best box size and resolution

**Next Steps**:
- If optimization not started: Run optimize_diameter tool
- If optimization running: Wait for completion
- If optimization complete: Review results and best box size
"""
            self._record_tool_execution("reason_about_workflow", {"input": input_str}, result={"analysis": reasoning})
            return reasoning
            
        except Exception as e:
            self._record_tool_execution("reason_about_workflow", {"input": input_str}, error=str(e))
            return f"❌ Error in workflow reasoning: {str(e)}"


