"""ReAct-based particle picking agent for RELION CryoEM data processing."""

import json
import subprocess
import os
import time
import logging
from typing import Dict, Any, List
from langchain.tools import Tool
from langchain_core.language_models import BaseLanguageModel
from typing import Optional
from pathlib import Path

from ..base_react_agent import BaseReActAgent
from .picking_tools import PickingTools
from ...config.config_loader import CryoAgentConfig, ConfigLoader
from ...tools.relion_tools import RELIONTools
from ...tools.relion_parser_tools import RelionPreprocessingParser, WorkflowContext


class PickingAgent(BaseReActAgent):
    """ReAct-based agent for RELION particle picking operations."""
    
    def __init__(
        self,
        config: CryoAgentConfig,
        llm: Optional[BaseLanguageModel] = None
    ):
        """
        Initialize the particle picking agent.
        
        Args:
            config: Complete configuration object
            llm: Language model for the agent
        """
        # Initialize RELION tools
        self.config_loader = ConfigLoader(
            config_path="configs/relion/particle_picking_config.json",
            master_config_path="configs/master_config.json"
        )
        self.relion_tools = RELIONTools(
            self.config_loader.get_relion_settings(),
            self.config_loader
        )
        
        # Enable backend execution for RELION tools
        self.relion_tools.enable_backend_execution(True)
        
        super().__init__(None, config, llm)  # No CryoSPARC tools needed for RELION
        # Initialize logger for this agent
        self.logger = logging.getLogger("RelionPickingAgent")
        
        # Initialize workflow state tracking for both rounds
        self.workflow_state = {
            # Round 1
            "blob_picker": {"completed": False, "job_dir": None, "output_file": None},
            "particle_extraction": {"completed": False, "job_dir": None, "output_file": None},
            "classification_2d": {"completed": False, "job_dir": None, "output_file": None, "optimiser_star": None},
            "auto_2d_selection": {"completed": False, "job_dir": None, "output_file": None, "class_averages_star": None},
            # Round 2
            "template_picker": {"completed": False, "job_dir": None, "output_file": None},
            "particle_extraction_2": {"completed": False, "job_dir": None, "output_file": None},
            "classification_2d_2": {"completed": False, "job_dir": None, "output_file": None, "optimiser_star": None},
            "auto_2d_selection_2": {"completed": False, "job_dir": None, "output_file": None}
        }
    
    def _parse_boolean_param(self, value: Any) -> bool:
        """Parse boolean parameter that might be string or boolean."""
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            return value.lower() in ['true', '1', 'yes', 'on']
        else:
            return bool(value)
    
    def _get_workflow_config(self) -> Dict[str, Any]:
        """Get workflow configuration from JSON file."""
        import json
        from pathlib import Path
        
        config_path = Path(self.config_loader.config_path)
        if not config_path.exists():
            return {}
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        return config_data.get("workflow", {})
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for particle picking operations."""
        return [
            PickingTools.create_blob_picker_tool(self),
            PickingTools.create_particle_extraction_tool(self),
            PickingTools.create_classification_2d_tool(self),
            PickingTools.create_auto_2d_selection_tool(self),
            PickingTools.create_template_picker_tool(self),
            PickingTools.create_check_job_status_tool(self),
            PickingTools.create_wait_for_job_tool(self),
            PickingTools.create_get_job_log_tool(self),
            PickingTools.create_validate_inputs_tool(self),
            PickingTools.create_reason_about_workflow_tool(self)
        ]
    
    def _get_react_system_prompt(self) -> str:
        """Get the ReAct system prompt for particle picking operations."""
        return "You are a RELION CryoEM particle picking assistant using the ReAct (Reasoning + Acting) framework. Follow the instructions provided in the workflow input."
    
    # Tool implementations
    def _blob_picker_tool(self, input_str: str) -> str:
        """Tool wrapper for blob picking using RELION tools."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            
            # Get input micrographs STAR file (support both JSON param and plain string)
            input_star = params.get('input_star') or params.get('input')
            if not input_star:
                return "❌ Error: input_star parameter is required"
            
            # Get blob picker config from JSON file
            blob_picker_config = self._get_workflow_config().get("blob_picker", {})
            
            # Set parameters with config as default, params as override
            # Use merge_parameters logic: only update if value is not None
            used_params = {
                'input_star': input_star,
                'output_dir': 'AutoPick',
            }
            
            # Merge config parameters, allowing params to override
            config_params = {
                'particle_diameter': blob_picker_config.get('particle_diameter', 200.0),
                'angpix': blob_picker_config.get('angpix', 1.0),
                'threshold': blob_picker_config.get('threshold', 0.25),
                'min_distance': blob_picker_config.get('min_distance', -1),
                'LoG': blob_picker_config.get('LoG', True),
                'LoG_diam_min': blob_picker_config.get('LoG_diam_min', 180.0),
                'LoG_diam_max': blob_picker_config.get('LoG_diam_max', 360.0),
                'LoG_neighbour': blob_picker_config.get('LoG_neighbour', 100.0),
                'LoG_adjust_threshold': blob_picker_config.get('LoG_adjust_threshold', 0.0),
                'LoG_upper_threshold': blob_picker_config.get('LoG_upper_threshold', 99999.0),
                'LoG_use_ctf': blob_picker_config.get('LoG_use_ctf', False),
                'gauss_max': blob_picker_config.get('gauss_max', 0.1),
                'write_fom_maps': blob_picker_config.get('write_fom_maps', False),
                'only_do_unfinished': blob_picker_config.get('only_do_unfinished', False),
                'wait_for_completion': True,
                'timeout': blob_picker_config.get('timeout', 3600),
                'use_backend': blob_picker_config.get('use_backend', self.relion_tools._backend_enabled),
                'conda_env': blob_picker_config.get('conda_env', 'relion-5.0')
            }
            
            # Update with params only if not None (like merge_parameters in test)
            for key, config_value in config_params.items():
                param_value = params.get(key)
                if param_value is not None:
                    used_params[key] = param_value
                else:
                    used_params[key] = config_value
            
            # Handle boolean parameters
            for bool_key in ['LoG', 'LoG_use_ctf', 'write_fom_maps', 'only_do_unfinished', 'wait_for_completion', 'use_backend']:
                if bool_key in used_params:
                    used_params[bool_key] = self._parse_boolean_param(used_params[bool_key])
            
            # Run blob picker
            result = self.relion_tools.blob_picker(**used_params)
            self._record_tool_execution("blob_picker", used_params, result=result)
            
            # Extract relative job directory for tracking
            output_dir_full = result.get("output_dir")
            if output_dir_full:
                relion_dir = self.relion_tools.relion_dir
                job_dir_relative = os.path.relpath(output_dir_full, relion_dir)
            else:
                job_dir_relative = None
            
            # Check if job actually completed (not just started in backend mode)
            job_status = result.get("status")
            if job_status == "completed":
                # Update workflow state only if job is actually completed
                self.workflow_state["blob_picker"]["completed"] = True
                self.workflow_state["blob_picker"]["job_dir"] = job_dir_relative  # Use relative path
                self.workflow_state["blob_picker"]["output_file"] = result.get("output_file")
                return f"✅ Successfully ran blob picker: {result.get('output_dir')}"
            elif job_status == "running":
                # Job started but not completed yet - LLM needs to wait
                self.workflow_state["blob_picker"]["completed"] = False
                self.workflow_state["blob_picker"]["job_dir"] = job_dir_relative  # Use relative path
                return f"🔄 Started blob picker job (still running): {result.get('output_dir')}. Please wait for completion before proceeding."
            else:
                # Job failed or unknown status
                return f"❌ Blob picker job has status: {job_status}. Error: {result.get('error', 'Unknown error')}"
            
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("blob_picker", context, error=str(e))
            return f"❌ Error during blob picking: {str(e)}"
    
    def _particle_extraction_tool(self, input_str: str) -> str:
        """Tool wrapper for particle extraction using RELION tools."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            
            # Get input micrographs STAR file (support both JSON param and plain string)
            input_star = params.get('input_star') or params.get('input')
            if not input_star:
                return "❌ Error: input_star parameter is required"
            
            # Get particle extraction config from JSON file
            extraction_config = self._get_workflow_config().get("particle_extraction", {})
            
            # Determine which round this is and get appropriate coord_list
            is_round2 = self.workflow_state.get("template_picker", {}).get("completed", False)
            
            # Get coord_list from params, or use appropriate picker output_dir
            coord_list = params.get("coord_list")
            if not coord_list:
                if is_round2:
                    # Round 2: use template_picker output
                    template_job_dir = self.workflow_state.get("template_picker", {}).get("job_dir")
                    if template_job_dir:
                        if not os.path.isabs(template_job_dir):
                            relion_dir = self.relion_tools.relion_dir
                            coord_list = os.path.join(relion_dir, template_job_dir)
                        else:
                            coord_list = template_job_dir
                else:
                    # Round 1: use blob_picker output
                    blob_job_dir = self.workflow_state.get("blob_picker", {}).get("job_dir")
                    if blob_job_dir:
                        if not os.path.isabs(blob_job_dir):
                            relion_dir = self.relion_tools.relion_dir
                            coord_list = os.path.join(relion_dir, blob_job_dir)
                        else:
                            coord_list = blob_job_dir
            
            # If still no coord_list, use ASINPUT as default
            if not coord_list:
                coord_list = "ASINPUT"
            
            coord_suffix = params.get('coord_suffix')
            if coord_suffix is None:
                coord_suffix = extraction_config.get('coord_suffix', '_autopick.star')
            
            # Set parameters with config as default, params as override
            # Use merge_parameters logic: only update if value is not None
            used_params = {
                'input_star': input_star,
                'output_dir': 'Extract',
                'coord_list': coord_list,
                'coord_suffix': coord_suffix,
            }
            
            # Merge config parameters, allowing params to override
            # Only include parameters that are not None (from test file merge_parameters logic)
            config_params = {
                'extract_size': extraction_config.get('extract_size', 440),
                'float16': extraction_config.get('float16', True),
                'scale': extraction_config.get('scale', 128),
                'norm': extraction_config.get('norm', True),
                'bg_radius': extraction_config.get('bg_radius', 48),
                'white_dust': extraction_config.get('white_dust', -1),
                'black_dust': extraction_config.get('black_dust', -1),
                'invert_contrast': extraction_config.get('invert_contrast', True),
                'extract_bias_x': extraction_config.get('extract_bias_x', 0.0),
                'extract_bias_y': extraction_config.get('extract_bias_y', 0.0),
                'only_do_unfinished': extraction_config.get('only_do_unfinished', False),
                'wait_for_completion': True,
                'timeout': extraction_config.get('timeout', 3600),
                'use_backend': extraction_config.get('use_backend', self.relion_tools._backend_enabled),
                'conda_env': extraction_config.get('conda_env', 'relion-5.0')
            }
            
            # Update with params only if not None (like merge_parameters in test)
            for key, config_value in config_params.items():
                param_value = params.get(key)
                if param_value is not None:
                    used_params[key] = param_value
                else:
                    used_params[key] = config_value
            
            # Handle boolean parameters
            for bool_key in ['float16', 'norm', 'invert_contrast', 'only_do_unfinished', 'wait_for_completion', 'use_backend']:
                if bool_key in used_params:
                    used_params[bool_key] = self._parse_boolean_param(used_params[bool_key])
            
            # Run particle extraction
            result = self.relion_tools.particle_extraction(**used_params)
            self._record_tool_execution("particle_extraction", used_params, result=result)
            
            # Extract relative job directory for tracking
            output_dir_full = result.get("output_dir")
            if output_dir_full:
                relion_dir = self.relion_tools.relion_dir
                job_dir_relative = os.path.relpath(output_dir_full, relion_dir)
            else:
                job_dir_relative = None
            
            # Determine step name based on round
            step_name = "particle_extraction_2" if is_round2 else "particle_extraction"
            
            # Check if job actually completed (not just started in backend mode)
            job_status = result.get("status")
            if job_status == "completed":
                # Update workflow state only if job is actually completed
                self.workflow_state[step_name]["completed"] = True
                self.workflow_state[step_name]["job_dir"] = job_dir_relative  # Use relative path
                self.workflow_state[step_name]["output_file"] = result.get("output_file")
                return f"✅ Successfully extracted particles: {result.get('output_dir')}"
            elif job_status == "running":
                # Job started but not completed yet - LLM needs to wait
                self.workflow_state[step_name]["completed"] = False
                self.workflow_state[step_name]["job_dir"] = job_dir_relative  # Use relative path
                return f"🔄 Started particle extraction job (still running): {result.get('output_dir')}. Please wait for completion before proceeding."
            else:
                # Job failed or unknown status
                return f"❌ Particle extraction job has status: {job_status}. Error: {result.get('error', 'Unknown error')}"
            
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("particle_extraction", context, error=str(e))
            return f"❌ Error during particle extraction: {str(e)}"
    
    def _classification_2d_tool(self, input_str: str) -> str:
        """Tool wrapper for 2D classification using RELION tools."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            
            # Determine which round this is
            is_round2 = self.workflow_state.get("template_picker", {}).get("completed", False)
            
            # Get input particles STAR file
            # If not provided explicitly, try to get from particle_extraction output (like test file does)
            input_star = params.get('input_star') or params.get('input')
            if not input_star:
                # Try to get from appropriate particle_extraction output
                extraction_step = "particle_extraction_2" if is_round2 else "particle_extraction"
                extraction_output_dir = self.workflow_state.get(extraction_step, {}).get("job_dir")
                if extraction_output_dir:
                    # Convert relative path to absolute
                    relion_dir = self.relion_tools.relion_dir
                    if not os.path.isabs(extraction_output_dir):
                        extraction_output_dir = os.path.join(relion_dir, extraction_output_dir)
                    input_star = os.path.join(extraction_output_dir, "particles.star")
            
            # If input_star is a directory, append particles.star to it
            if input_star and os.path.isdir(input_star):
                input_star = os.path.join(input_star, "particles.star")
            
            if not input_star:
                return "❌ Error: input_star parameter is required or particle_extraction must complete first"
            
            # Get classification config from JSON file
            classification_config = self._get_workflow_config().get("classification_2d", {})
            
            # Set parameters with config as default, params as override
            # Use merge_parameters logic: only update if value is not None
            used_params = {
                'input_star': input_star,
                'output_dir': 'Class2D',
            }
            
            # Merge config parameters, allowing params to override
            config_params = {
                'K': classification_config.get('K', 50),
                'iter': classification_config.get('iter', 25),
                'tau2_fudge': classification_config.get('tau2_fudge', 2.0),
                'particle_diameter': classification_config.get('particle_diameter', 260.0),
                'angpix': classification_config.get('angpix', -1),
                'offset_range': classification_config.get('offset_range', 6.0),
                'offset_step': classification_config.get('offset_step', 2.0),
                'oversampling': classification_config.get('oversampling', 1),
                'healpix_order': classification_config.get('healpix_order', 2),
                'psi_step': classification_config.get('psi_step', -1),
                'skip_align': classification_config.get('skip_align', False),
                'skip_rotate': classification_config.get('skip_rotate', False),
                'ctf': classification_config.get('ctf', True),
                'norm': classification_config.get('norm', True),
                'scale': classification_config.get('scale', True),
                'pool': classification_config.get('pool', 1),
                'j': classification_config.get('j', 1),
                'only_do_unfinished': classification_config.get('only_do_unfinished', False),
                'gpu': classification_config.get('gpu'),  # None if not specified (no GPU), True/string if specified
                'wait_for_completion': True,
                'timeout': classification_config.get('timeout', 7200),
                'use_backend': classification_config.get('use_backend', self.relion_tools._backend_enabled),
                'conda_env': classification_config.get('conda_env', 'relion-5.0')
            }
            
            # Update with params only if not None (like merge_parameters in test)
            for key, config_value in config_params.items():
                param_value = params.get(key)
                if param_value is not None:
                    used_params[key] = param_value
                else:
                    used_params[key] = config_value
            
            # Handle boolean parameters (but not gpu, which can be bool or string)
            for bool_key in ['skip_align', 'skip_rotate', 'ctf', 'norm', 'scale', 'only_do_unfinished', 'wait_for_completion', 'use_backend']:
                if bool_key in used_params:
                    used_params[bool_key] = self._parse_boolean_param(used_params[bool_key])
            
            # Handle gpu parameter specially - can be bool (True) or string (GPU ID)
            if 'gpu' in used_params:
                gpu_val = used_params['gpu']
                if isinstance(gpu_val, bool):
                    # If boolean True, convert to empty string to enable default GPU
                    if gpu_val:
                        used_params['gpu'] = ''
                    else:
                        # If False, remove from params to use default (no GPU)
                        del used_params['gpu']
                elif isinstance(gpu_val, str):
                    # If string, keep as is (could be '' for default or '0' for specific GPU)
                    pass
                elif gpu_val is not None:
                    # If other type (e.g., int), convert to string
                    used_params['gpu'] = str(gpu_val)
                else:
                    # If None, remove from params to use default (no GPU)
                    del used_params['gpu']
            
            # Run 2D classification
            result = self.relion_tools.classification_2d(**used_params)
            self._record_tool_execution("classification_2d", used_params, result=result)
            
            # Extract relative job directory for tracking
            output_dir_full = result.get("output_dir")
            if output_dir_full:
                relion_dir = self.relion_tools.relion_dir
                job_dir_relative = os.path.relpath(output_dir_full, relion_dir)
            else:
                job_dir_relative = None
            
            # Determine step name based on round
            step_name = "classification_2d_2" if is_round2 else "classification_2d"
            
            # Check if job actually completed (not just started in backend mode)
            job_status = result.get("status")
            if job_status == "completed":
                # Update workflow state only if job is actually completed
                self.workflow_state[step_name]["completed"] = True
                self.workflow_state[step_name]["job_dir"] = job_dir_relative  # Use relative path
                self.workflow_state[step_name]["output_file"] = result.get("output_file")
                # Store optimiser_star if available in result (for auto_2d_selection)
                if "optimiser_star" in result:
                    self.workflow_state[step_name]["optimiser_star"] = result.get("optimiser_star")
                return f"✅ Successfully classified particles: {result.get('output_dir')}"
            elif job_status == "running":
                # Job started but not completed yet - LLM needs to wait
                self.workflow_state[step_name]["completed"] = False
                self.workflow_state[step_name]["job_dir"] = job_dir_relative  # Use relative path
                return f"🔄 Started 2D classification job (still running): {result.get('output_dir')}. Please wait for completion before proceeding."
            else:
                # Job failed or unknown status
                return f"❌ 2D classification job has status: {job_status}. Error: {result.get('error', 'Unknown error')}"
            
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("classification_2d", context, error=str(e))
            return f"❌ Error during 2D classification: {str(e)}"
    
    def _auto_2d_selection_tool(self, input_str: str) -> str:
        """Tool wrapper for automatic 2D class selection using RELION tools."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            
            # Determine which round this is
            is_round2 = self.workflow_state.get("template_picker", {}).get("completed", False)
            classification_step = "classification_2d_2" if is_round2 else "classification_2d"
            selection_step = "auto_2d_selection_2" if is_round2 else "auto_2d_selection"
            
            # Get classification state (needed for both checking stored value and final verification)
            classification_state = self.workflow_state.get(classification_step, {})
            
            # Get input_opt: check params first (explicit input_opt or parsed 'input'), then workflow state, then input_str as fallback
            input_opt = params.get('input_opt') or params.get('input')
            if not input_opt:
                input_opt = classification_state.get("optimiser_star")
            if not input_opt:
                # Fallback: use input_str directly if it looks like a path
                if input_str and (input_str.startswith('/') or input_str.startswith('./') or '/' in input_str):
                    input_opt = input_str.strip()
            
            if not input_opt:
                return "❌ Error: input_opt parameter is required. Provide the optimiser.star file path from 2D classification."
            
            print(f"input_opt is used: {input_opt}" )
            
            # Get auto selection config from JSON file
            selection_config = self._get_workflow_config().get("auto_2d_selection", {})
            
            # Set parameters with config as default, params as override
            # Use merge_parameters logic: only update if value is not None
            used_params = {
                'input_opt': input_opt,
                'output_dir': 'Select',
            }
            
            # Merge config parameters, allowing params to override
            config_params = {
                'min_score': selection_config.get('min_score', 0.05),
                'max_score': selection_config.get('max_score', 999.0),
                'select_min_nr_particles': selection_config.get('select_min_nr_particles', -1),
                'select_min_nr_classes': selection_config.get('select_min_nr_classes', -1),
                'relative_thresholds': selection_config.get('relative_thresholds', False),
                'auto_select': selection_config.get('auto_select', True),
                'fn_sel_parts': selection_config.get('fn_sel_parts', 'particles.star'),
                'fn_sel_classavgs': selection_config.get('fn_sel_classavgs', 'class_averages.star'),
                'wait_for_completion': True,
                'timeout': selection_config.get('timeout', 1800),
                'use_backend': selection_config.get('use_backend', self.relion_tools._backend_enabled)
            }
            
            # Update with params only if not None (like merge_parameters in test)
            for key, config_value in config_params.items():
                param_value = params.get(key)
                if param_value is not None:
                    used_params[key] = param_value
                else:
                    used_params[key] = config_value
            
            # Allow explicit input_opt from params to override (handles edge cases)
            if 'input_opt' in params and params['input_opt'] is not None:
                used_params['input_opt'] = params['input_opt']
            
            # Handle boolean parameters
            for bool_key in ['relative_thresholds', 'auto_select', 'wait_for_completion', 'use_backend']:
                if bool_key in used_params:
                    used_params[bool_key] = self._parse_boolean_param(used_params[bool_key])
            
            # Run auto 2D selection
            result = self.relion_tools.auto_2d_selection(**used_params)
            self._record_tool_execution("auto_2d_selection", used_params, result=result)
            
            # Extract relative job directory for tracking
            output_dir_full = result.get("output_dir")
            if output_dir_full:
                relion_dir = self.relion_tools.relion_dir
                job_dir_relative = os.path.relpath(output_dir_full, relion_dir)
            else:
                job_dir_relative = None
            
            # Check if job actually completed (not just started in backend mode)
            job_status = result.get("status")
            if job_status == "completed":
                # Update workflow state only if job is actually completed
                self.workflow_state[selection_step]["completed"] = True
                self.workflow_state[selection_step]["job_dir"] = job_dir_relative  # Use relative path
                self.workflow_state[selection_step]["output_file"] = result.get("output_file")
                return f"✅ Successfully ran auto 2D selection: {result.get('output_dir')}"
            elif job_status == "running":
                # Job started but not completed yet - LLM needs to wait
                self.workflow_state[selection_step]["completed"] = False
                self.workflow_state[selection_step]["job_dir"] = job_dir_relative  # Use relative path
                return f"🔄 Started auto 2D selection job (still running): {result.get('output_dir')}. Please wait for completion before proceeding."
            else:
                # Job failed or unknown status
                return f"❌ Auto 2D selection job has status: {job_status}. Error: {result.get('error', 'Unknown error')}"
            
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("auto_2d_selection", context, error=str(e))
            return f"❌ Error during auto 2D selection: {str(e)}"
    
    def _template_picker_tool(self, input_str: str) -> str:
        """Tool wrapper for template-based picking using RELION tools."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            
            # Get input micrographs STAR file (support both JSON param and plain string)
            input_star = params.get('input_star') or params.get('input')
            if not input_star:
                return "❌ Error: input_star parameter is required"
            
            # Get ref_star from auto_2d_selection output (round 1) - like test file line 241
            ref_star = params.get('ref_star')
            if not ref_star:
                # Try to get from auto_2d_selection output (round 1)
                auto_select_state = self.workflow_state.get("auto_2d_selection", {})
                if auto_select_state.get("class_averages_star"):
                    ref_star = auto_select_state["class_averages_star"]
                else:
                    # Fallback: construct from job directory
                    auto_select_job_dir = auto_select_state.get("job_dir")
                    if auto_select_job_dir:
                        relion_dir = self.relion_tools.relion_dir
                        if not os.path.isabs(auto_select_job_dir):
                            auto_select_job_dir = os.path.join(relion_dir, auto_select_job_dir)
                        ref_star = os.path.join(auto_select_job_dir, "class_averages.star")
            
            if not ref_star:
                return "❌ Error: ref_star parameter is required or auto_2d_selection (round 1) must complete first"
            
            # Get template picker config from JSON file
            template_picker_config = self._get_workflow_config().get("template_picker", {})
            
            # Set parameters with config as default, params as override
            # Use merge_parameters logic: only update if value is not None
            used_params = {
                'input_star': input_star,
                'ref_star': ref_star,
                'output_dir': 'AutoPick',
            }
            
            # Merge config parameters, allowing params to override
            config_params = {
                'pickname': template_picker_config.get('pickname', 'autopick'),
                'fn_topaz_exe': template_picker_config.get('fn_topaz_exe', 'relion_python_topaz'),
                'ang': template_picker_config.get('ang', 5.0),
                'shrink': template_picker_config.get('shrink', 0),
                'lowpass': template_picker_config.get('lowpass', 20.0),
                'threshold': template_picker_config.get('threshold', 0.05),
                'min_distance': template_picker_config.get('min_distance', 100.0),
                'max_stddev_noise': template_picker_config.get('max_stddev_noise', 1.1),
                'invert': template_picker_config.get('invert', True),
                'ctf': template_picker_config.get('ctf', True),
                'gpu': template_picker_config.get('gpu', ''),
                'only_do_unfinished': template_picker_config.get('only_do_unfinished', False),
                'wait_for_completion': True,
                'timeout': template_picker_config.get('timeout', 3600),
                'use_backend': template_picker_config.get('use_backend', self.relion_tools._backend_enabled),
                'conda_env': template_picker_config.get('conda_env', 'relion-5.0')
            }
            
            # Update with params only if not None (like merge_parameters in test)
            for key, config_value in config_params.items():
                param_value = params.get(key)
                if param_value is not None:
                    used_params[key] = param_value
                else:
                    used_params[key] = config_value
            
            # Handle boolean parameters (but not gpu, which can be bool or string)
            for bool_key in ['invert', 'ctf', 'only_do_unfinished', 'wait_for_completion', 'use_backend']:
                if bool_key in used_params:
                    used_params[bool_key] = self._parse_boolean_param(used_params[bool_key])
            
            # Handle gpu parameter specially - can be bool (True) or string (GPU ID)
            if 'gpu' in used_params:
                gpu_val = used_params['gpu']
                if isinstance(gpu_val, bool):
                    # If boolean True, convert to empty string to enable default GPU
                    if gpu_val:
                        used_params['gpu'] = ''
                    else:
                        # If False, remove from params to use default (no GPU)
                        del used_params['gpu']
                elif isinstance(gpu_val, str):
                    # If string, keep as is (could be '' for default or '0' for specific GPU)
                    pass
                elif gpu_val is not None:
                    # If other type (e.g., int), convert to string
                    used_params['gpu'] = str(gpu_val)
                else:
                    # If None, remove from params to use default (no GPU)
                    del used_params['gpu']
            
            # Run template picker
            result = self.relion_tools.template_picker(**used_params)
            self._record_tool_execution("template_picker", used_params, result=result)
            
            # Extract relative job directory for tracking
            output_dir_full = result.get("output_dir")
            if output_dir_full:
                relion_dir = self.relion_tools.relion_dir
                job_dir_relative = os.path.relpath(output_dir_full, relion_dir)
            else:
                job_dir_relative = None
            
            # Check if job actually completed (not just started in backend mode)
            job_status = result.get("status")
            if job_status == "completed":
                # Update workflow state only if job is actually completed
                self.workflow_state["template_picker"]["completed"] = True
                self.workflow_state["template_picker"]["job_dir"] = job_dir_relative  # Use relative path
                self.workflow_state["template_picker"]["output_file"] = result.get("output_file")
                return f"✅ Successfully ran template picker: {result.get('output_dir')}"
            elif job_status == "running":
                # Job started but not completed yet - LLM needs to wait
                self.workflow_state["template_picker"]["completed"] = False
                self.workflow_state["template_picker"]["job_dir"] = job_dir_relative  # Use relative path
                return f"🔄 Started template picker job (still running): {result.get('output_dir')}. Please wait for completion before proceeding."
            else:
                # Job failed or unknown status
                return f"❌ Template picker job has status: {job_status}. Error: {result.get('error', 'Unknown error')}"
            
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("template_picker", context, error=str(e))
            return f"❌ Error during template picking: {str(e)}"
    
    def _check_job_status_tool(self, input_str: str) -> str:
        """Tool wrapper for checking RELION job status."""
        try:
            params = self._parse_tool_input(input_str)
            job_dir = params.get("job_dir")
            
            # Handle case where job directory is passed directly as input
            if not job_dir and "input" in params:
                job_dir = params["input"]
            
            if not job_dir:
                return "❌ Error: job_dir parameter is required"
            
            result = self.relion_tools.get_job_status(job_dir)
            self._record_tool_execution("check_job_status", {"job_dir": job_dir}, result=result)
            return result
            
        except Exception as e:
            self._record_tool_execution("check_job_status", {"input": input_str}, error=str(e))
            return f"❌ Error checking job status: {str(e)}"
    
    def _wait_for_job_tool(self, input_str: str) -> str:
        """Tool wrapper for waiting for job completion."""
        try:
            params = self._parse_tool_input(input_str)
            job_dir = params.get("job_dir")
            
            # Handle case where job directory is passed directly as input
            if not job_dir and "input" in params:
                job_dir = params["input"]
            
            timeout = int(params.get("timeout", 3600))
            check_interval = int(params.get("check_interval", 30))
            
            if not job_dir:
                return "❌ Error: job_dir parameter is required"
            
            result = self.relion_tools.wait_for_job_completion(
                job_dir, timeout=timeout, check_interval=check_interval
            )
            self._record_tool_execution("wait_for_job", {"job_dir": job_dir, "timeout": timeout, "check_interval": check_interval}, result=result)
            return result
            
        except Exception as e:
            self._record_tool_execution("wait_for_job", {"input": input_str}, error=str(e))
            return f"❌ Error waiting for job: {str(e)}"
    
    def _get_job_log_tool(self, input_str: str) -> str:
        """Tool wrapper for reading RELION job logs."""
        try:
            params = self._parse_tool_input(input_str)
            job_dir = params.get("job_dir")
            
            # Handle case where job directory is passed directly as input
            if not job_dir and "input" in params:
                job_dir = params["input"]
            
            if not job_dir:
                return "❌ Error: job_dir parameter is required"
            
            result = self.relion_tools.get_job_log(job_dir)
            self._record_tool_execution("get_job_log", {"job_dir": job_dir}, result=result)
            return result
            
        except Exception as e:
            self._record_tool_execution("get_job_log", {"input": input_str}, error=str(e))
            return f"❌ Error reading job log: {str(e)}"
    
    def _validate_inputs_tool(self, input_str: str) -> str:
        """Tool wrapper for validating input files and parameters."""
        try:
            params = self._parse_tool_input(input_str)
            input_type = params.get("input_type")
            input_path = params.get("input_path")
            
            if not input_type or not input_path:
                return "❌ Error: input_type and input_path parameters are required"
            
            result = self.relion_tools.validate_inputs(input_type, input_path)
            self._record_tool_execution("validate_inputs", {"input_type": input_type, "input_path": input_path}, result=result)
            return result
            
        except Exception as e:
            self._record_tool_execution("validate_inputs", {"input": input_str}, error=str(e))
            return f"❌ Error validating inputs: {str(e)}"
    
    def _reason_about_workflow_tool(self, input_str: str) -> str:
        """Tool wrapper for analyzing current workflow state and determining next steps."""
        try:
            analysis = "🤔 **RELION Particle Picking Workflow Analysis**:\n\n"
            
            for step, state in self.workflow_state.items():
                status = "✅ COMPLETED" if state["completed"] else "⏳ PENDING"
                analysis += f"**{step.replace('_', ' ').title()}**: {status}\n"
                if state["job_dir"]:
                    analysis += f"  - Job directory: {state['job_dir']}\n"
                if state["output_file"]:
                    analysis += f"  - Output file: {state['output_file']}\n"
                analysis += "\n"
            
            # Determine next step
            if not self.workflow_state["blob_picker"]["completed"]:
                analysis += "**Next Step**: Run blob_picker to detect particles in micrographs\n"
                analysis += "- Use blob_picker tool with input_star parameter\n"
                analysis += "- All parameters are loaded from particle_picking_config.json\n"
            elif not self.workflow_state["particle_extraction"]["completed"]:
                analysis += "**Next Step**: Run particle_extraction to extract particle images\n"
                analysis += f"- Input: {self.workflow_state.get('blob_picker', {}).get('output_file', 'N/A')}\n"
            elif not self.workflow_state["classification_2d"]["completed"]:
                analysis += "**Next Step**: Run classification_2d to classify particles\n"
                analysis += f"- Input: {self.workflow_state.get('particle_extraction', {}).get('output_file', 'N/A')}\n"
            elif not self.workflow_state["auto_2d_selection"]["completed"]:
                analysis += "**Next Step**: Run auto_2d_selection to select best classes\n"
                analysis += f"- Input: {self.workflow_state.get('classification_2d', {}).get('output_file', 'N/A')}\n"
            else:
                analysis += "**Status**: All workflow steps completed successfully!\n"
            
            return analysis
            
        except Exception as e:
            self._record_tool_execution("reason_about_workflow", {"input": input_str}, error=str(e))
            return f"❌ Workflow analysis failed: {str(e)}"
    
    def process_workflow_results(self, results: List, context: WorkflowContext) -> Dict[str, Any]:
        """
        Process workflow results and extract stage outputs.
        
        Args:
            results: List of picking workflow results
            context: Workflow context with project/workspace info
            
        Returns:
            Dictionary of stage outputs
        """
        # For RELION picking, we can extract outputs directly from workflow_state
        stage_outputs = {}
        
        # Extract outputs from each completed step
        for step_name, step_state in self.workflow_state.items():
            if step_state.get("completed", False):
                if step_state.get("output_file"):
                    stage_outputs[f"{step_name}_output_file"] = step_state["output_file"]
                if step_state.get("job_dir"):
                    stage_outputs[f"{step_name}_job_dir"] = step_state["job_dir"]
        
        # Add final selected particles (prefer round 2 if completed, otherwise round 1)
        if self.workflow_state.get("auto_2d_selection_2", {}).get("completed"):
            auto_select_output = self.workflow_state["auto_2d_selection_2"].get("output_file")
            if auto_select_output:
                stage_outputs["selected_particles_star"] = auto_select_output
        elif self.workflow_state.get("auto_2d_selection", {}).get("completed"):
            auto_select_output = self.workflow_state["auto_2d_selection"].get("output_file")
            if auto_select_output:
                stage_outputs["selected_particles_star"] = auto_select_output
        
        return stage_outputs
    
    def validate_results(self, stage_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that the picking workflow completed successfully.
        
        Args:
            stage_outputs: Dictionary of stage outputs to validate
            
        Returns:
            Dictionary with 'success' boolean and 'error' message if failed
        """
        import os
        
        # Check if all required steps completed (both rounds - 8 steps total)
        required_steps = [
            # Round 1
            "blob_picker", 
            "particle_extraction", 
            "classification_2d", 
            "auto_2d_selection",
            # Round 2
            "template_picker",
            "particle_extraction_2",
            "classification_2d_2",
            "auto_2d_selection_2"
        ]
        
        failed_steps = []
        missing_steps = []
        
        for step in required_steps:
            step_state = self.workflow_state.get(step, {})
            completed_flag = step_state.get("completed", False)
            
            if not completed_flag:
                missing_steps.append(step)
                continue
            
            # Verify actual job status for completed steps
            job_dir = step_state.get("job_dir")
            if job_dir:
                # Convert relative path to absolute
                if not os.path.isabs(job_dir):
                    relion_dir = self.relion_tools.relion_dir
                    job_dir = os.path.join(relion_dir, job_dir)
                
                # Check actual job status
                try:
                    job_status = self.relion_tools.get_job_status(job_dir)
                    actual_status = job_status.get("status", "unknown")
                    
                    if actual_status == "failed":
                        failed_steps.append(step)
                        self.logger.warning(f"Step {step} marked as completed but job actually failed: {job_dir}")
                    elif actual_status != "completed":
                        # If status is "running" or "unknown", mark as incomplete
                        missing_steps.append(step)
                        self.logger.warning(f"Step {step} marked as completed but job status is {actual_status}: {job_dir}")
                except Exception as e:
                    self.logger.warning(f"Failed to verify job status for {step}: {e}")
                    # If we can't verify, trust the completed flag but log a warning
        
        # Determine overall success
        if failed_steps:
            return {
                "success": False,
                "error": f"Workflow failed. Failed steps: {', '.join(failed_steps)}"
            }
        
        if missing_steps:
            return {
                "success": False,
                "error": f"Workflow incomplete. Missing steps: {', '.join(missing_steps)}"
            }
        
        return {
            "success": True,
            "error": None
        }
    
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
        import datetime
        import os
        from pathlib import Path
        
        # Create output directory if it doesn't exist
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Create picking results dictionary
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        status = "completed" if success else "failed"
        
        # Build step-by-step status information
        step_status = {}
        required_steps = [
            "blob_picker", 
            "particle_extraction", 
            "classification_2d", 
            "auto_2d_selection",
            "template_picker",
            "particle_extraction_2",
            "classification_2d_2",
            "auto_2d_selection_2"
        ]
        
        for step in required_steps:
            step_state = self.workflow_state.get(step, {})
            step_status[step] = {
                "completed": step_state.get("completed", False),
                "job_dir": step_state.get("job_dir"),
                "output_file": step_state.get("output_file"),
                "status": "pending"
            }
            
            # Check actual job status if job_dir exists
            job_dir = step_state.get("job_dir")
            if job_dir:
                # Convert relative path to absolute
                if not os.path.isabs(job_dir):
                    relion_dir = self.relion_tools.relion_dir
                    job_dir = os.path.join(relion_dir, job_dir)
                
                try:
                    job_status_result = self.relion_tools.get_job_status(job_dir)
                    actual_status = job_status_result.get("status", "unknown")
                    step_status[step]["status"] = actual_status
                    step_status[step]["job_status"] = job_status_result
                except Exception as e:
                    step_status[step]["status"] = "unknown"
                    step_status[step]["error"] = str(e)
        
        # Build stage_outputs with job directories
        enhanced_stage_outputs = stage_outputs.copy()
        for step, status_info in step_status.items():
            if status_info.get("job_dir"):
                # Convert relative to absolute path for stage_outputs
                job_dir = status_info["job_dir"]
                if not os.path.isabs(job_dir):
                    relion_dir = self.relion_tools.relion_dir
                    job_dir = os.path.join(relion_dir, job_dir)
                enhanced_stage_outputs[f"{step}_job_dir"] = job_dir
            if status_info.get("output_file"):
                enhanced_stage_outputs[f"{step}_output_file"] = status_info["output_file"]
        
        picking_results = {
            "timestamp": timestamp,
            "status": status,
            "stage": "particle_picking",
            "agent_type": "relion",
            "stage_outputs": enhanced_stage_outputs,
            "step_status": step_status,
            "metadata": {
                "workflow_type": getattr(context, 'workflow_type', 'unknown'),
                "start_time": getattr(context, 'start_time', None),
                "conversation_id": getattr(context, 'conversation_id', None)
            }
        }
        
        # Save to JSON file
        output_file = output_dir / f"particle_picking_results_relion_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(picking_results, f, indent=2)
        
        self.logger.info(f"Particle picking results saved to {output_file}")
        return str(output_file)
