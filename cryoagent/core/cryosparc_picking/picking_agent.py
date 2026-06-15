"""ReAct-based particle picking agent for CryoEM data processing."""

import json
import logging
import math
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain.tools import Tool
from langchain_core.language_models import BaseLanguageModel

from ..cryosparc_common_tools import CryoSPARCCommonTools
from ..base_react_agent import BaseReActAgent
from .picking_tools import PickingTools
from ...tools.cryosparc_tools import CryoSPARCTools
from ...tools.cryosparc_parser_tools import CryoSPARCPickingParser, WorkflowContext
from ...config.config_loader import CryoAgentConfig
from ...prompts.prompt_loader import load_prompt


class PickingAgent(BaseReActAgent):
    """ReAct-based agent for CryoEM particle picking operations."""
    
    # Class-level lock to prevent concurrent classification job creation
    _class_2d_lock = None
    
    def __init__(
        self,
        cryosparc_tools: CryoSPARCTools,
        config: CryoAgentConfig,
        llm: Optional[BaseLanguageModel] = None
    ):
        """
        Initialize the particle picking agent.
        
        Args:
            cryosparc_tools: CryoSPARC tools instance
            config: Complete configuration object
            llm: Language model for the agent
        """
        # Initialize class-level lock if not already done
        import threading
        if PickingAgent._class_2d_lock is None:
            PickingAgent._class_2d_lock = threading.Lock()
        
        super().__init__(cryosparc_tools, config, llm)
        # Initialize logger for this agent
        self.logger = logging.getLogger("PickingAgent")
        # Load stage configuration for picking (agent-specific parameters)
        self.stage_config = self._load_stage_config()
        self.stage_workflow = self.stage_config.get("workflow", {})
        # Cache microscope configuration for derived defaults, respecting microscope_config overrides
        stage_defaults = self.stage_config.get("microscope_parameters", {})
        self.microscope_config = self._resolve_microscope_defaults(stage_defaults, update_cache=True)
    
    def _create_tools(self) -> List[Tool]:
        """Create particle picking-specific tools."""
        return [
            PickingTools.create_blob_picker_tool(self),
            PickingTools.create_extract_particles_tool(self),
            PickingTools.create_class_2d_tool(self),
            PickingTools.create_select_2d_classes_tool(self),
            PickingTools.create_template_picker_tool(self),
            PickingTools.create_get_job_status_tool(self),
            PickingTools.create_wait_for_job_tool(self),
            PickingTools.create_get_job_log_tool(self),
            CryoSPARCCommonTools.create_search_cryosparc_forum_tool(self),
            CryoSPARCCommonTools.create_describe_job_params_tool(self),
            PickingTools.create_reason_about_workflow_tool(self)
        ]
    
    def _load_stage_config(self) -> Dict[str, Any]:
        """Load particle picking stage configuration."""
        config_path = Path("configs/cryosparc/particle_picking_config.json")
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        try:
            with open(config_path, "r", encoding="utf-8") as fp:
                return json.load(fp) or {}
        except FileNotFoundError:
            self.logger.warning("Particle picking configuration not found at %s; using defaults.", config_path)
        except json.JSONDecodeError as exc:
            self.logger.warning("Invalid JSON in particle picking configuration %s: %s", config_path, exc)
        return {}
    
    def _get_stage_param(self, section: str, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """Convenience helper to read a parameter from the stage workflow config."""
        return self.stage_workflow.get(section, {}).get(key, default)
    
    def _get_system_prompt_context(self) -> Dict[str, Any]:
        """Build template variables for cryosparc/particle_picking/system.md."""
        return {
            "project_uid": self.config.workflow.project_uid,
            "workspace_uid": self.config.workflow.workspace_uid,
        }

    def _get_react_system_prompt(self) -> str:
        """Get the particle picking-specific ReAct system prompt."""
        return self._compose_stage_system_prompt(
            "cryosparc/particle_picking/system.md",
            self._get_system_prompt_context(),
        )
    
    # Tool implementation methods
    def _blob_picker_tool(self, input_str: str) -> str:
        """Tool wrapper for blob picker particle detection."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            
            # Resolve particle diameter from params or stage configuration
            particle_diameter_param = params.get("particle_diameter")
            base_diameter = self._get_base_particle_diameter()
            if base_diameter is None:
                stage_default = self._get_stage_param("blob_picker", "particle_diameter")
                try:
                    base_diameter = float(stage_default) if stage_default is not None else None
                except (TypeError, ValueError):
                    base_diameter = None
            auto_min_diameter = base_diameter * 0.7 if base_diameter else None
            auto_max_diameter = base_diameter * 1.3 if base_diameter else None

            if particle_diameter_param is not None:
                particle_diameter_value = float(particle_diameter_param)
            elif base_diameter is not None:
                particle_diameter_value = float(base_diameter)
            else:
                particle_diameter_value = self._get_stage_param("blob_picker", "particle_diameter")
                if particle_diameter_value is None:
                    particle_diameter_value = self.microscope_config.get("particle_diameter")
                if particle_diameter_value is None:
                    return "❌ Error: particle_diameter parameter is required for blob picker"
                particle_diameter_value = float(particle_diameter_value)
            
            used_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "micrographs_job_uid": params.get("micrographs_job_uid"),
                "particle_diameter": float(particle_diameter_value),
                "wait_for_completion": params.get("wait_for_completion", "false").lower() == "true",
                "timeout": int(params.get("timeout", self.config.job_management.default_timeout)),
                "check_interval": int(params.get("check_interval", self.config.job_management.status_check_interval))
            }
            
            # Determine diameter_max preference: user override > derived default > CryoSPARC fallback
            diameter_max_param = params.get("diameter_max")
            if diameter_max_param is not None:
                used_params["diameter_max"] = float(diameter_max_param)
            elif auto_max_diameter is not None:
                used_params["diameter_max"] = float(auto_max_diameter)
            else:
                stage_diameter_max = self._get_stage_param("blob_picker", "diameter_max")
                if stage_diameter_max is not None:
                    used_params["diameter_max"] = float(stage_diameter_max)

            passthrough = self._extract_passthrough_params(
                params,
                consumed_keys=["micrographs_job_uid", "particle_diameter", "diameter_max"],
            )
            if passthrough:
                used_params["params"] = passthrough

            result = self.cryosparc_tools.blob_picker(**used_params)
            self._record_tool_execution("blob_picker", used_params, result=result)
            diameter_range = f"{used_params['particle_diameter']}-{used_params.get('diameter_max', used_params['particle_diameter'] * 2.0)}"
            return f"✅ Successfully queued blob picker GPU job: {result['job_uid']} (diameter range: {diameter_range} Å)"
            
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("blob_picker", context, error=str(e))
            return f"❌ Error starting blob picker: {str(e)}"
    
    def _extract_particles_tool(self, input_str: str) -> str:
        """Tool wrapper for particle extraction."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            
            # Get box size from params or derived defaults
            box_size_pix = params.get("box_size_pix")
            if box_size_pix is None:
                box_size_pix = self._get_stage_param("particle_extraction", "box_size_pix")

            if box_size_pix is None:
                base_diameter = self._get_base_particle_diameter()
                pixel_size = self._get_microscope_parameter("pixel_size")
                try:
                    if base_diameter and pixel_size:
                        pixel_size_val = float(pixel_size)
                        if pixel_size_val > 0:
                            computed_box = (base_diameter / pixel_size_val) + 125
                            normalized_box = self._normalize_box_size(computed_box)
                            if normalized_box is not None:
                                box_size_pix = normalized_box
                            else:
                                box_size_pix = int(round(computed_box))
                except (TypeError, ValueError, ZeroDivisionError):
                    box_size_pix = None

            if box_size_pix is None:
                return "❌ Error: box_size_pix parameter is required for particle extraction"
            
            # Get micrographs_job_uid
            micrographs_job_uid = params.get("micrographs_job_uid")
            if not micrographs_job_uid:
                return "❌ Error: micrographs_job_uid parameter is required for particle extraction"
            
            used_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "particles_job_uid": params.get("particles_job_uid"),
                "micrographs_job_uid": micrographs_job_uid,
                "box_size_pix": int(box_size_pix),
                "wait_for_completion": params.get("wait_for_completion", "false").lower() == "true",
                "timeout": int(params.get("timeout", self.config.job_management.default_timeout)),
                "check_interval": int(params.get("check_interval", self.config.job_management.status_check_interval))
            }

            passthrough = self._extract_passthrough_params(
                params,
                consumed_keys=["particles_job_uid", "micrographs_job_uid", "box_size_pix"],
            )
            if passthrough:
                used_params["params"] = passthrough

            result = self.cryosparc_tools.extract_particles(**used_params)
            self._record_tool_execution("extract_particles", used_params, result=result)
            
            return f"✅ Successfully queued particle extraction job: {result['job_uid']} (box size: {box_size_pix} pixels)"
            
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("extract_particles", context, error=str(e))
            return f"❌ Error starting particle extraction: {str(e)}"
    
    def _class_2d_tool(self, input_str: str) -> str:
        """Tool wrapper for 2D classification."""
        # Use lock to prevent concurrent job creation when LLM makes parallel tool calls
        with PickingAgent._class_2d_lock:
            params: Dict[str, Any] = {}
            used_params: Dict[str, Any] = {}
            try:
                # Parse input once at the beginning
                params = self._parse_tool_input(input_str)
                project_uid = params.get("project_uid", self.config.workflow.project_uid)
                workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
                
                # Check if there's already a running classification job to prevent concurrent executions
                # IMPORTANT: Allow a second round of 2D classification if the first round is completed
                # We check the ACTUAL current status from CryoSPARC, not the cached status in the log
                recent_class_2d_jobs = [
                    entry for entry in self.tool_execution_log[-10:]  # Check last 10 entries
                    if entry.get("tool") == "class_2d"
                ]
                if recent_class_2d_jobs:
                    last_job = recent_class_2d_jobs[-1]
                    last_result = last_job.get("result")
                    if isinstance(last_result, dict):
                        last_job_uid = last_result.get("job_uid")
                        if last_job_uid:
                            # Get the ACTUAL current status from CryoSPARC (not cached log status)
                            # This fixes the issue where log shows "queued" but job is actually "completed"
                            try:
                                # Query CryoSPARC for the actual current status
                                actual_status_info = self.cryosparc_tools.get_job_status(
                                    last_job_uid,
                                    project_uid=project_uid,
                                    workspace_uid=workspace_uid
                                )
                                actual_status = actual_status_info.get("status", "")
                                
                                # Update the execution log entry with the fresh status
                                # This ensures the log has accurate status for future checks
                                if isinstance(last_result, dict):
                                    last_result["status"] = actual_status
                                    # Also update other status-related fields if available
                                    if "progress" in actual_status_info:
                                        last_result["progress"] = actual_status_info.get("progress")
                                    if "message" in actual_status_info:
                                        last_result["message"] = actual_status_info.get("message")
                                
                                # Only block if the previous job is actually still running (not completed)
                                # This allows Round 2 to proceed after Round 1 completes
                                if actual_status in ("queued", "launched", "running", "waiting"):
                                    return f"❌ Error: A 2D classification job is already running (status: {actual_status}, job: {last_job_uid}). Wait for it to complete before starting a new one."
                                # If the job is completed, failed, or cancelled, allow the new job to proceed
                                # This enables Round 2 to start after Round 1 completes
                            except Exception as status_check_error:
                                # If we can't check the status, fall back to the cached status
                                # But log a warning that we're using potentially stale data
                                last_status = last_result.get("status", "")
                                if last_status in ("queued", "launched", "running", "waiting"):
                                    self.logger.warning(
                                        f"Could not verify job {last_job_uid} status from CryoSPARC, "
                                        f"using cached status '{last_status}'. Error: {status_check_error}"
                                    )
                                    return f"❌ Error: A 2D classification job may be running (cached status: {last_status}, job: {last_job_uid}). Wait for it to complete before starting a new one. If this is Round 2 and Round 1 is completed, check the job status to confirm."
                
                # Support both particles_job_uid and job_uid (when LLM passes just "J88")
                particles_job_uid = params.get("particles_job_uid") or params.get("job_uid")
                if not particles_job_uid:
                    return f"❌ Error: Missing required parameter: particles_job_uid (or job_uid)"
                
                # Get num_classes from params or config (default 200, matches optimization_2d stage)
                num_classes = params.get("num_classes")
                if not num_classes:
                    num_classes = self._get_stage_param("2d_classification", "num_classes", 200)

                batchsize_per_class = (
                    params.get("batchsize_per_class")
                    or params.get("batch_size_per_class")
                )
                if batchsize_per_class is None:
                    batchsize_per_class = self._get_stage_param("2d_classification", "batch_size_per_class")
                if batchsize_per_class is None:
                    batchsize_per_class = self._get_stage_param("2d_classification", "batchsize_per_class")

                used_params = {
                    "project_uid": project_uid,
                    "workspace_uid": workspace_uid,
                    "particles_job_uid": particles_job_uid,
                    "num_classes": int(num_classes),
                    "wait_for_completion": params.get("wait_for_completion", "false").lower() == "true",
                    "timeout": int(params.get("timeout", self.config.job_management.default_timeout * 2)),  # 2D classification takes longer
                    "check_interval": int(params.get("check_interval", self.config.job_management.status_check_interval))
                }
                if batchsize_per_class is not None:
                    used_params["batchsize_per_class"] = int(batchsize_per_class)

                passthrough = self._extract_passthrough_params(
                    params,
                    consumed_keys=["particles_job_uid", "job_uid", "num_classes", "batchsize_per_class", "batch_size_per_class"],
                )
                if passthrough:
                    used_params["params"] = passthrough

                result = self.cryosparc_tools.class_2d(**used_params)
                self._record_tool_execution("class_2d", used_params, result=result)
                
                return f"✅ Successfully queued 2D classification job: {result['job_uid']} ({num_classes} classes)"
                
            except Exception as e:
                context = used_params or params or {"raw_input": input_str}
                self._record_tool_execution("class_2d", context, error=str(e))
                return f"❌ Error starting 2D classification: {str(e)}"
    
    def _select_2d_classes_tool(self, input_str: str) -> str:
        """Tool wrapper for 2D class selection."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            
            # Determine selection strategy and parameters
            selection_mode = (params.get("selection_mode") or params.get("selection_strategy") or "top_n").lower()

            top_n_classes = params.get("top_n_classes")
            if top_n_classes is None and selection_mode != "cryosift":
                top_n_classes = self._get_stage_param("select_2d_classes", "top_n_classes", 5)

            cryosift_threshold = params.get("cryosift_threshold")
            cryosift_env = params.get("cryosift_env")
            cryosift_weights_path = params.get("cryosift_weights_path")
            cryosift_evaluator_script_path = params.get("cryosift_evaluator_script_path")
            cryosift_output_dir = params.get("cryosift_output_dir")
            cryosift_output_subdir = params.get("cryosift_output_subdir")
            cryosift_python = params.get("cryosift_python_executable")
            cryosift_fallback = params.get("cryosift_fallback_strategy")
            
            # Read from config if not provided in params
            if selection_mode == "cryosift":
                if cryosift_threshold is None:
                    cryosift_threshold = self._get_stage_param("select_2d_classes", "cryosift_threshold")
                if not cryosift_env:
                    cryosift_env = self._get_stage_param("select_2d_classes", "cryosift_env")
                if not cryosift_weights_path:
                    cryosift_weights_path = self._get_stage_param("select_2d_classes", "cryosift_weights_path")
                if not cryosift_evaluator_script_path:
                    cryosift_evaluator_script_path = self._get_stage_param("select_2d_classes", "cryosift_evaluator_script_path")
                if not cryosift_output_subdir:
                    cryosift_output_subdir = self._get_stage_param("select_2d_classes", "cryosift_output_subdir")
                if not cryosift_fallback:
                    cryosift_fallback = self._get_stage_param("select_2d_classes", "cryosift_fallback_strategy")
            
            # Support both class_2d_job_uid and job_uid (when LLM passes just "J88")
            class_2d_job_uid = params.get("class_2d_job_uid") or params.get("job_uid")
            if not class_2d_job_uid:
                return "❌ Error: class_2d_job_uid parameter is required for 2D class selection"
            
            used_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "class_2d_job_uid": class_2d_job_uid,
                "selection_mode": selection_mode,
                "wait_for_completion": params.get("wait_for_completion", "false").lower() == "true",
                "timeout": int(params.get("timeout", 300)),
                "check_interval": int(params.get("check_interval", 10))
            }
            if top_n_classes is not None:
                try:
                    used_params["top_n_classes"] = int(top_n_classes)
                except (TypeError, ValueError):
                    used_params["top_n_classes"] = None

            cryosift_options: Dict[str, Any] = {}
            if cryosift_threshold is not None:
                try:
                    cryosift_options["threshold"] = float(cryosift_threshold)
                except (TypeError, ValueError):
                    pass
            if cryosift_env:
                cryosift_options["conda_env"] = cryosift_env
            if cryosift_weights_path:
                cryosift_options["weights_path"] = cryosift_weights_path
            if cryosift_evaluator_script_path:
                cryosift_options["evaluator_script_path"] = cryosift_evaluator_script_path
            if cryosift_output_dir:
                cryosift_options["output_dir"] = cryosift_output_dir
            if cryosift_output_subdir:
                cryosift_options["output_subdir"] = cryosift_output_subdir
            if cryosift_python:
                cryosift_options["python_executable"] = cryosift_python
            if cryosift_fallback:
                cryosift_options["fallback_strategy"] = cryosift_fallback

            if cryosift_options:
                used_params["cryosift_options"] = cryosift_options

            result = self.cryosparc_tools.select_2d_classes(**used_params)
            self._record_tool_execution("select_2d_classes", used_params, result=result)
            
            strategy = result.get("selection_metadata", {}).get("selection_mode", selection_mode)
            detail = ""
            if strategy == "top_n" and used_params.get("top_n_classes"):
                detail = f"selecting top {used_params['top_n_classes']} classes"
            elif strategy == "cryosift":
                selected = result.get("selected_template_indices") or []
                detail = f"CryoSift-selected classes {selected}" if selected else "CryoSift evaluation completed"

            detail_text = f" ({detail})" if detail else ""
            return f"✅ Successfully queued 2D class selection job: {result['job_uid']}{detail_text}"
            
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("select_2d_classes", context, error=str(e))
            return f"❌ Error starting 2D class selection: {str(e)}"
    
    def _template_picker_tool(self, input_str: str) -> str:
        """Tool wrapper for template-based picking."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            
            # Get micrographs_job_uid
            micrographs_job_uid = params.get("micrographs_job_uid")
            if not micrographs_job_uid:
                return "❌ Error: micrographs_job_uid parameter is required for template picker"
            
            # Resolve template picker parameters (fall back to recorded blob picker values if needed)
            lowpass_resolution = params.get("lowpass_resolution") or self._get_stage_param("template_picker", "lowpass_resolution", 20.0)

            particle_diameter = params.get("particle_diameter")
            if particle_diameter is None:
                particle_diameter = self._get_stage_param("blob_picker", "particle_diameter")
            if particle_diameter is None:
                particle_diameter = self.microscope_config.get("particle_diameter")
            angular_spacing_deg = params.get("angular_spacing_deg") or params.get("angle_search_range")
            blob_picker_job_uid = params.get("blob_picker_job_uid")

            if not particle_diameter or not blob_picker_job_uid:
                for entry in reversed(self.tool_execution_log):
                    if entry.get("tool") == "blob_picker" and isinstance(entry.get("result"), dict):
                        blob_picker_job_uid = blob_picker_job_uid or entry["result"].get("job_uid")
                        blob_params = entry["result"].get("params", {})
                        if not particle_diameter:
                            particle_diameter = blob_params.get("diameter") or blob_params.get("particle_diameter")
                        if blob_picker_job_uid and particle_diameter:
                            break

            used_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "micrographs_job_uid": micrographs_job_uid,
                "template_job_uid": params.get("template_job_uid"),
                "lowpass_resolution": float(lowpass_resolution),
                "particle_diameter": float(particle_diameter) if particle_diameter else None,
                "lowpass_micrograph": float(params.get("lowpass_micrograph")) if params.get("lowpass_micrograph") else None,
                "angular_spacing_deg": float(angular_spacing_deg) if angular_spacing_deg else None,
                "min_distance": float(params.get("min_distance")) if params.get("min_distance") else None,
                "use_ctf": params.get("use_ctf"),
                "blob_picker_job_uid": blob_picker_job_uid,
                "wait_for_completion": params.get("wait_for_completion", "false").lower() == "true",
                "timeout": int(params.get("timeout", self.config.job_management.default_timeout)),
                "check_interval": int(params.get("check_interval", self.config.job_management.status_check_interval))
            }

            result = self.cryosparc_tools.template_picker(**used_params)
            self._record_tool_execution("template_picker", used_params, result=result)
            
            return f"✅ Successfully queued template picker job: {result['job_uid']} (lowpass: {lowpass_resolution} Å)"
            
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("template_picker", context, error=str(e))
            return f"❌ Error starting template picker: {str(e)}"
    
    def _reason_about_workflow_tool(self, input_str: str) -> str:
        """Tool for reasoning about particle picking workflow state."""
        try:
            reasoning = f"""
🤔 **Particle Picking Workflow Analysis**:

**Current State**: {input_str}

**Complete Workflow Overview**:
1. **Blob Picker GPU**: Detect particles using GPU-accelerated Gaussian blob detection
2. **Particle Extraction**: Extract particle images based on picking coordinates
3. **2D Classification**: Group particles into classes for quality assessment

**Parameter Considerations**:
- **Particle Diameter (Min)**: Minimum expected size of the protein complex
  * Too small: May pick noise or fragment particles
  * Too large: May miss smaller particles
  * Should be determined from prior knowledge or initial screening
  * This is the lower bound of the search range for blob picker

- **Diameter Max**: Maximum expected particle size for blob picker
  * Default (2.0 × particle_diameter) works for most cases
  * Increase if particles have significant size variation
  * Defines the upper bound of the search range

- **Box Size (Angstroms)**: Size of extracted particle images
  * Typically 1.5-2x the particle diameter
  * Must be large enough to include the entire particle plus context
  * Affects downstream processing (classification, refinement)
  * Larger boxes provide more context but increase computational cost

- **Number of Classes**: Number of 2D classes for classification
  * Default: from stage config (same scale as 2D optimization, typically 200 classes)
  * More classes: Better separation of views and junk, but slower
  * Fewer classes: Faster but less detailed classification
  * Typical range: 10-100 depending on dataset size and heterogeneity

**Workflow Dependencies**:
1. Blob picker requires completed micrograph job (CTF or selection)
2. Extraction requires completed blob picker job
3. 2D classification requires completed extraction job
4. Each step must complete before proceeding to the next

**Next Steps Analysis**:
- If no jobs running: Start with blob_picker
- If blob picker running/done: Proceed to extraction
- If extraction running/done: Proceed to 2D classification
- If 2D classification done: Workflow complete, analyze results

**Recommended Actions**:
- Always verify each job UID is valid and completed before using it
- Wait for each job to complete before starting the next
- Monitor job status for errors or warnings
- Use appropriate parameters based on particle characteristics
"""
            self._record_tool_execution("reason_about_workflow", {"input": input_str}, result={"analysis": reasoning})
            return reasoning
            
        except Exception as e:
            self._record_tool_execution("reason_about_workflow", {"input": input_str}, error=str(e))
            return f"❌ Error in workflow reasoning: {str(e)}"
    
    def process_workflow_results(self, results: List, context: WorkflowContext) -> Dict[str, Any]:
        """
        Process workflow results and extract stage outputs.
        
        Args:
            results: List of picking workflow results
            context: Workflow context with project/workspace info
            
        Returns:
            Dictionary of stage outputs
        """
        parser = CryoSPARCPickingParser(self.cryosparc_tools, self.logger)
        return parser.process_workflow_results(results, context)
    
    def validate_results(self, stage_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that the picking workflow completed successfully.
        
        Args:
            stage_outputs: Dictionary of stage outputs to validate
            
        Returns:
            Dictionary with 'success' boolean and 'error' message if failed
        """
        parser = CryoSPARCPickingParser(self.cryosparc_tools, self.logger)
        return parser.validate_results(stage_outputs)
    
    def save_results(self, stage_outputs: Dict[str, Any], context: WorkflowContext, success: bool = True) -> str:
        """
        Save picking results to a JSON file.
        
        Args:
            stage_outputs: Dictionary of stage outputs
            context: Workflow context
            success: Whether picking was successful
            
        Returns:
            Path to the saved JSON file
        """
        parser = CryoSPARCPickingParser(self.cryosparc_tools, self.logger)
        return parser.save_results(stage_outputs, context, success)
