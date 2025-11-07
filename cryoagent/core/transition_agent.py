"""
Transition Agent for seamless conversion between CryoSparc and Relion formats.

This agent handles format conversion when transitioning between stages that use
different backends (CryoSparc <-> Relion). It uses an agentic workflow to monitor
and verify the conversion process.
"""

import json
import os
import logging
import glob
import time
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
from langchain.tools import Tool

from ..config.config_loader import ConfigLoader
from ..tools.cryosparc_tools import CryoSPARCTools
from ..tools.conversion_tool import FileConversionTools
from ..tools.relion_tools import RELIONTools
from .base_react_agent import BaseReActAgent


class BackendType(Enum):
    """Enumeration of backend types."""
    CRYOSPARC = "cryosparc"
    RELION = "relion"


class TransitionMonitoringAgent(BaseReActAgent):
    """Agent for monitoring and verifying transition conversions."""
    
    def __init__(self, transition_agent, config):
        """Initialize the transition monitoring agent."""
        self.transition_agent = transition_agent
        super().__init__(None, config, None)  # No CryoSparc tools needed
        
    def _create_tools(self) -> List[Tool]:
        """Create tools for monitoring transition."""
        return [
            Tool(
                name="verify_star_file",
                description="Verify that a STAR file exists and is valid",
                func=self._verify_star_file_tool
            ),
            Tool(
                name="verify_relion_directory",
                description="Verify that a Relion job directory exists and contains expected files",
                func=self._verify_relion_directory_tool
            ),
            Tool(
                name="verify_config_file",
                description="Verify that a JSON config file exists and is valid",
                func=self._verify_config_file_tool
            ),
            Tool(
                name="check_file_exists",
                description="Check if a file exists at the given path",
                func=self._check_file_exists_tool
            )
        ]
    
    def _get_react_system_prompt(self) -> str:
        """Get the system prompt for transition monitoring."""
        return "You are a transition monitoring assistant. Your role is to verify that format conversions between CryoSparc and Relion have completed successfully. Check that all required files and directories exist and are properly formatted."
    
    def _verify_star_file_tool(self, input_str: str) -> str:
        """Verify that a STAR file exists and is valid."""
        try:
            params = self._parse_tool_input(input_str)
            star_path = params.get("star_file") or params.get("file_path") or input_str
            
            if not star_path or not os.path.exists(star_path):
                return f"❌ STAR file does not exist: {star_path}"
            
            # Check if it's a valid STAR file by reading first few lines
            try:
                with open(star_path, 'r') as f:
                    lines = f.readlines()[:10]
                    if any('data_' in line for line in lines):
                        file_size = os.path.getsize(star_path)
                        return f"✅ STAR file is valid: {star_path} (size: {file_size} bytes)"
                    else:
                        return f"⚠️ File exists but may not be a valid STAR file: {star_path}"
            except Exception as e:
                return f"❌ Error reading STAR file: {e}"
                
        except Exception as e:
            return f"❌ Error verifying STAR file: {str(e)}"
    
    def _verify_relion_directory_tool(self, input_str: str) -> str:
        """Verify that a Relion job directory exists."""
        try:
            params = self._parse_tool_input(input_str)
            job_dir = params.get("job_dir") or params.get("directory") or input_str
            
            if not job_dir or not os.path.exists(job_dir):
                return f"❌ Relion directory does not exist: {job_dir}"
            
            if not os.path.isdir(job_dir):
                return f"❌ Path is not a directory: {job_dir}"
            
            # Check for common Relion files
            star_files = list(Path(job_dir).glob("*.star"))
            if star_files:
                return f"✅ Relion directory exists: {job_dir} (contains {len(star_files)} STAR file(s))"
            else:
                return f"⚠️ Relion directory exists but contains no STAR files: {job_dir}"
                
        except Exception as e:
            return f"❌ Error verifying Relion directory: {str(e)}"
    
    def _verify_config_file_tool(self, input_str: str) -> str:
        """Verify that a JSON config file exists and is valid."""
        try:
            params = self._parse_tool_input(input_str)
            config_path = params.get("config_file") or params.get("file_path") or input_str
            
            if not config_path or not os.path.exists(config_path):
                return f"❌ Config file does not exist: {config_path}"
            
            # Validate JSON
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    return f"✅ Config file is valid JSON: {config_path} (keys: {list(config_data.keys())})"
            except json.JSONDecodeError as e:
                return f"❌ Config file is not valid JSON: {e}"
                
        except Exception as e:
            return f"❌ Error verifying config file: {str(e)}"
    
    def _check_file_exists_tool(self, input_str: str) -> str:
        """Check if a file exists."""
        try:
            params = self._parse_tool_input(input_str)
            file_path = params.get("file_path") or params.get("path") or input_str
            
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                return f"✅ File exists: {file_path} (size: {size} bytes)"
            else:
                return f"❌ File does not exist: {file_path}"
                
        except Exception as e:
            return f"❌ Error checking file: {str(e)}"


class TransitionWorkflow:
    """Agentic workflow for monitoring transition conversions."""
    
    def __init__(
        self,
        transition_agent,
        stage_name: str,
        stage_outputs: Dict[str, Any],
        project_uid: str
    ):
        """Initialize the transition workflow."""
        self.transition_agent = transition_agent
        self.stage_name = stage_name
        self.stage_outputs = stage_outputs
        self.project_uid = project_uid
        self.logger = logging.getLogger("TransitionWorkflow")
        self.conversion_result = {}
    
    def run(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Run the transition workflow."""
        try:
            # Step 1: Perform the actual conversion
            conversion_result = self._perform_conversion()
            
            # Step 2: Use agentic workflow to monitor and verify
            if conversion_result.get("success"):
                self._verify_conversion(conversion_result, conversation_id)
            
            return conversion_result
            
        except Exception as e:
            self.logger.error(f"Transition workflow failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "converted_outputs": {}
            }
    
    def _perform_conversion(self) -> Dict[str, Any]:
        """Perform the actual format conversion."""
        converted_outputs = {}
        relion_dir = Path(self.transition_agent.relion_tools.relion_dir)
        
        if self.stage_name == "preprocessing":
            return self._convert_preprocessing_outputs(relion_dir)
        elif self.stage_name == "particle_picking":
            return self._convert_particle_picking_outputs(relion_dir)
        else:
            return {
                "success": False,
                "error": f"Conversion not implemented for stage: {self.stage_name}",
                "converted_outputs": {}
            }
    
    def _convert_preprocessing_outputs(self, relion_dir: Path) -> Dict[str, Any]:
        """Convert preprocessing outputs (micrographs)."""
        job_uid = self.stage_outputs.get("micrograph_selection_job_uid")
        if not job_uid:
            self.logger.error(f"No micrograph_selection_job_uid found. Available keys: {list(self.stage_outputs.keys())}")
            return {
                "success": False,
                "error": "No micrograph selection job UID found",
                "converted_outputs": {}
            }
        
        self.logger.info(f"Starting conversion for job_uid: {job_uid}, project_uid: {self.project_uid}")
        
        try:
            # Try to get job directory - first from CryoSparc tools if available,
            # otherwise try to get from cached stage_outputs
            job_dir = None
            
            if self.transition_agent.cryosparc_tools:
                try:
                    # Get CryoSparc job output directory using tools
                    job_info = self.transition_agent.cryosparc_tools.get_job_output_directory(
                        self.project_uid, job_uid
                    )
                    job_dir = Path(job_info["job_directory"])
                    self.logger.info(f"Got job directory from CryoSparc tools: {job_dir}")
                except Exception as e:
                    self.logger.warning(f"Could not get job directory from CryoSparc tools: {e}")
                    job_dir = None
            
            # Fallback: try to get from cached data
            if not job_dir:
                # Check if we have output_directory in stage_outputs
                micrograph_location = self.stage_outputs.get("micrograph_location", {})
                self.logger.info(f"Checking micrograph_location: {micrograph_location}")
                output_directory = micrograph_location.get("output_directory")
                
                if output_directory:
                    self.logger.info(f"Found output_directory in cache: {output_directory}")
                    if os.path.exists(output_directory):
                        job_dir = Path(output_directory)
                        self.logger.info(f"Using cached output directory: {job_dir}")
                    else:
                        self.logger.warning(f"Cached output directory does not exist: {output_directory}")
                else:
                    self.logger.warning(f"No output_directory found in micrograph_location")
                
                # Final fallback: try to construct from job_uid if we still don't have it
                if not job_dir:
                    return {
                        "success": False,
                        "error": f"Could not find job directory for {job_uid}. CryoSparc tools not available and no cached output_directory found. Available stage_outputs keys: {list(self.stage_outputs.keys())}",
                        "converted_outputs": {}
                    }
            
            if not job_dir or not job_dir.exists():
                return {
                    "success": False,
                    "error": f"Job directory does not exist: {job_dir}",
                    "converted_outputs": {}
                }
            
            self.logger.info(f"Using job directory: {job_dir}")
            
            # Find exposure/micrograph .cs file - look for exposure files first
            micrographs_cs = None
            exposure_candidates = [
                job_dir / "0_exposures_accepted.cs",
                job_dir / "exposures_accepted.cs",
            ]
            for candidate in exposure_candidates:
                if candidate.exists():
                    micrographs_cs = candidate
                    break
            
            # Fallback: look for any .cs file
            if not micrographs_cs:
                cs_files = list(job_dir.glob("*.cs"))
                # Prefer files with "exposure" in the name
                exposure_files = [f for f in cs_files if "exposure" in f.name.lower()]
                if exposure_files:
                    micrographs_cs = exposure_files[0]
                elif cs_files:
                    micrographs_cs = cs_files[0]
                else:
                    return {
                        "success": False,
                        "error": f"No .cs file found in {job_dir}",
                        "converted_outputs": {}
                    }
            
            self.logger.info(f"Using CryoSparc file: {micrographs_cs}")
            
            # Find passthrough file for exposure data (contains CTF info and micrograph paths)
            passthrough_file = None
            # Try specific patterns first
            passthrough_candidates = [
                job_dir / f"{micrographs_cs.stem}_passthrough_exposures_accepted.cs",
                job_dir / f"{micrographs_cs.stem}_passthrough.cs",
            ]
            for candidate in passthrough_candidates:
                if candidate.exists():
                    passthrough_file = candidate
                    break
            
            # Fallback: glob for any passthrough file
            if not passthrough_file:
                passthrough_glob = list(job_dir.glob("*_passthrough_exposures_accepted.cs"))
                if passthrough_glob:
                    passthrough_file = passthrough_glob[0]
            
            if passthrough_file:
                self.logger.info(f"Found passthrough file: {passthrough_file}")
            
            # Create Relion Select job directory
            relion_job_dir = self.transition_agent.relion_tools._get_next_job_directory("Select")
            relion_job_dir_path = Path(relion_job_dir)
            relion_job_dir_relative = os.path.relpath(relion_job_dir, relion_dir)
            
            # Convert to STAR file in Relion directory
            star_file_name = "micrographs.star"
            star_file_path = relion_job_dir_path / star_file_name
            
            # Convert using passthrough file if available
            # Pass job_directory to help resolve micrograph paths to absolute paths
            self.transition_agent.conversion_tools.convert_cs_to_star(
                micrographs_cs,
                star_file_path,
                passthrough_path=str(passthrough_file) if passthrough_file else None,
                job_directory=str(job_dir)
            )
            
            # Generate JSON config file in Relion directory
            config_file_path = relion_job_dir_path / "transition_config.json"
            config_data = {
                "transition_info": {
                    "from_agent": "cryosparc",
                    "to_agent": "relion",
                    "stage": "preprocessing",
                    "source_job_uid": job_uid,
                    "source_directory": str(job_dir),
                    "conversion_timestamp": datetime.datetime.now().isoformat()
                },
                "relion_job": {
                    "job_dir": relion_job_dir_relative,
                    "job_dir_absolute": relion_job_dir,
                    "star_file": star_file_name,
                    "star_file_absolute": str(star_file_path.absolute())
                },
                "conversion_metadata": {
                    "source_file": str(micrographs_cs),
                    "converter": "FileConversionTools",
                    "relion_dir": str(relion_dir)
                }
            }
            
            with open(config_file_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            # Also save a copy in outputs/transitions folder for easy access
            transitions_dir = Path("outputs") / "transitions"
            transitions_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            transitions_config_file = transitions_dir / f"transition_{self.stage_name}_{timestamp}.json"
            with open(transitions_config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            self.logger.info(f"Created Relion job directory: {relion_job_dir}")
            self.logger.info(f"Converted STAR file: {star_file_path}")
            self.logger.info(f"Created config file: {config_file_path}")
            self.logger.info(f"Saved transition config to transitions folder: {transitions_config_file}")
            
            return {
                "success": True,
                "converted_outputs": {
                    "selected_micrographs_star": str(star_file_path.absolute()),
                    "selection_job_dir": relion_job_dir_relative,
                    "transition_config": str(config_file_path.absolute()),
                    "transition_config_transitions": str(transitions_config_file.absolute()),
                    "transition_info": config_data.get("transition_info")
                },
                "relion_job_dir": relion_job_dir_relative,
                "star_file": str(star_file_path.absolute()),
                "config_file": str(config_file_path.absolute()),
                "transitions_config_file": str(transitions_config_file.absolute())
            }
            
        except Exception as e:
            self.logger.error(f"Failed to convert preprocessing outputs: {e}")
            return {
                "success": False,
                "error": str(e),
                "converted_outputs": {}
            }
    
    def _convert_particle_picking_outputs(self, relion_dir: Path) -> Dict[str, Any]:
        """Convert particle picking outputs (particles)."""
        job_uid = (self.stage_outputs.get("final_selection_job_uid") or 
                  self.stage_outputs.get("selected_particles_job_uid"))
        if not job_uid or not self.transition_agent.cryosparc_tools:
            return {
                "success": False,
                "error": "No particle selection job UID found",
                "converted_outputs": {}
            }
        
        try:
            # Get CryoSparc job output directory
            job_info = self.transition_agent.cryosparc_tools.get_job_output_directory(
                self.project_uid, job_uid
            )
            job_dir = Path(job_info["job_directory"])
            
            # Find particles.cs file
            particles_cs = job_dir / "particles_selected.cs"
            if not particles_cs.exists():
                particles_cs = job_dir / "particles.cs"
                if not particles_cs.exists():
                    cs_files = list(job_dir.glob("*particles*.cs"))
                    if cs_files:
                        particles_cs = cs_files[0]
                    else:
                        return {
                            "success": False,
                            "error": f"No particles .cs file found in {job_dir}",
                            "converted_outputs": {}
                        }
            
            # Look for passthrough file
            passthrough_file = job_dir / "particles_selected_passthrough.cs"
            if not passthrough_file.exists():
                passthrough_file = job_dir / "particles_passthrough.cs"
                if not passthrough_file.exists():
                    passthrough_file = None
            
            # Create Relion Select job directory for particles
            relion_job_dir = self.transition_agent.relion_tools._get_next_job_directory("Select")
            relion_job_dir_path = Path(relion_job_dir)
            relion_job_dir_relative = os.path.relpath(relion_job_dir, relion_dir)
            
            # Convert to STAR file in Relion directory
            star_file_name = "particles.star"
            star_file_path = relion_job_dir_path / star_file_name
            
            self.transition_agent.conversion_tools.convert_cs_to_star(
                particles_cs,
                star_file_path,
                passthrough_path=passthrough_file if passthrough_file and passthrough_file.exists() else None
            )
            
            # Generate JSON config file in Relion directory
            config_file_path = relion_job_dir_path / "transition_config.json"
            config_data = {
                "transition_info": {
                    "from_agent": "cryosparc",
                    "to_agent": "relion",
                    "stage": "particle_picking",
                    "source_job_uid": job_uid,
                    "source_directory": str(job_dir),
                    "conversion_timestamp": datetime.datetime.now().isoformat()
                },
                "relion_job": {
                    "job_dir": relion_job_dir_relative,
                    "job_dir_absolute": relion_job_dir,
                    "star_file": star_file_name,
                    "star_file_absolute": str(star_file_path.absolute())
                },
                "conversion_metadata": {
                    "source_file": str(particles_cs),
                    "passthrough_file": str(passthrough_file) if passthrough_file else None,
                    "converter": "FileConversionTools",
                    "relion_dir": str(relion_dir)
                }
            }
            
            with open(config_file_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            # Also save a copy in outputs folder for easy access
            outputs_dir = Path("outputs")
            outputs_dir.mkdir(exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            outputs_config_file = outputs_dir / f"transition_particle_picking_{timestamp}.json"
            with open(outputs_config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            self.logger.info(f"Created Relion job directory: {relion_job_dir}")
            self.logger.info(f"Converted STAR file: {star_file_path}")
            self.logger.info(f"Created config file: {config_file_path}")
            self.logger.info(f"Saved transition config to outputs: {outputs_config_file}")
            
            return {
                "success": True,
                "converted_outputs": {
                    "final_star_file": str(star_file_path.absolute()),
                    "transition_config": str(config_file_path.absolute()),
                    "transition_config_outputs": str(outputs_config_file.absolute())
                },
                "relion_job_dir": relion_job_dir_relative,
                "star_file": str(star_file_path.absolute()),
                "config_file": str(config_file_path.absolute()),
                "outputs_config_file": str(outputs_config_file.absolute())
            }
            
        except Exception as e:
            self.logger.error(f"Failed to convert particle picking outputs: {e}")
            return {
                "success": False,
                "error": str(e),
                "converted_outputs": {}
            }
    
    def _verify_conversion(self, conversion_result: Dict[str, Any], conversation_id: Optional[str] = None):
        """Use agentic workflow to verify the conversion."""
        try:
            # Initialize monitoring agent
            config_loader = ConfigLoader(
                config_path="configs/relion/preprocessing_config.json",
                master_config_path=self.transition_agent.master_config_path
            )
            config = config_loader.load_config()
            
            monitoring_agent = TransitionMonitoringAgent(self.transition_agent, config)
            
            # Create verification workflow input
            star_file = conversion_result.get("star_file")
            job_dir = conversion_result.get("relion_job_dir")
            config_file = conversion_result.get("config_file")
            
            workflow_input = f"""
            Verify that the transition conversion completed successfully:
            
            1. Check that the STAR file exists: {star_file}
            2. Check that the Relion job directory exists: {job_dir}
            3. Check that the config file exists: {config_file}
            4. Verify all files are valid and properly formatted
            
            Use the verification tools to check each requirement.
            """
            
            # Run verification
            verification_result = monitoring_agent.run_react_workflow(workflow_input, conversation_id)
            self.logger.info(f"Verification result: {verification_result}")
            
            # Update conversion result with verification status
            conversion_result["verification"] = {
                "verified": "✅" in verification_result or "valid" in verification_result.lower(),
                "message": verification_result
            }
            
        except Exception as e:
            self.logger.warning(f"Verification workflow failed: {e}")
            conversion_result["verification"] = {
                "verified": False,
                "error": str(e)
            }


class TransitionAgent:
    """Agent for handling format transitions between CryoSparc and Relion."""
    
    def __init__(self, master_config_path: str = "configs/master_config.json", cryosparc_tools: Optional[Any] = None, relion_tools: Optional[Any] = None):
        """
        Initialize the transition agent.
        
        Args:
            master_config_path: Path to the master configuration file
            cryosparc_tools: Optional pre-initialized CryoSparc tools (if available from master orchestrator)
            relion_tools: Optional pre-initialized Relion tools (if available from master orchestrator)
        """
        self.master_config_path = master_config_path
        self.master_config = None
        self.logger = logging.getLogger("TransitionAgent")
        self.conversion_tools = FileConversionTools()
        self.cryosparc_tools = cryosparc_tools
        self.relion_tools = relion_tools
        
        # Load master config
        self._load_master_config()
        
        # Initialize backend tools if needed (only if not provided)
        if not self.cryosparc_tools or not self.relion_tools:
            self._initialize_tools()
    
    def _load_master_config(self):
        """Load the master configuration file."""
        try:
            with open(self.master_config_path, 'r') as f:
                self.master_config = json.load(f)
            self.logger.info(f"Loaded master config from {self.master_config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load master config: {e}")
            raise
    
    def _initialize_tools(self):
        """Initialize CryoSparc and Relion tools if needed."""
        # Only initialize if not already provided
        if not self.cryosparc_tools:
            try:
                # Initialize CryoSparc tools if config exists
                if "cryosparc" in self.master_config:
                    from ..config.config_loader import CryoSPARCSettings
                    from ..tools.cryosparc_tools import CryoSPARCTools
                    cryosparc_config = self.master_config["cryosparc"]
                    settings = CryoSPARCSettings(
                        host=cryosparc_config.get("host", "localhost"),
                        base_port=cryosparc_config.get("base_port", 39000),
                        username=cryosparc_config.get("username", ""),
                        password=cryosparc_config.get("password", ""),
                        license_id=cryosparc_config.get("license_id", "")
                    )
                    self.cryosparc_tools = CryoSPARCTools(settings)
                    self.logger.info("Initialized CryoSparc tools")
            
            except Exception as e:
                self.logger.warning(f"Could not initialize CryoSparc tools: {e}")
        
        if not self.relion_tools:
            try:
                # Initialize Relion tools if config exists
                if "relion" in self.master_config:
                    from ..config.config_loader import ConfigLoader
                    from ..tools.relion_tools import RELIONTools
                    config_loader = ConfigLoader(
                        config_path="configs/relion/preprocessing_config.json",
                        master_config_path=self.master_config_path
                    )
                    self.relion_tools = RELIONTools(
                        config_loader.get_relion_settings(),
                        config_loader
                    )
                    self.logger.info("Initialized Relion tools")
            
            except Exception as e:
                self.logger.warning(f"Could not initialize Relion tools: {e}")
    
    def _resolve_relion_path(self, path: Optional[str]) -> Optional[str]:
        """Resolve a path that may be relative to the RELION project directory."""
        if not path:
            return None

        try:
            path_obj = Path(path).expanduser()
        except Exception:
            return path

        if path_obj.is_absolute():
            return str(path_obj)

        relion_dir = Path(getattr(self.relion_tools, "relion_dir", Path.cwd()))
        candidate = relion_dir / path_obj
        if candidate.exists():
            return str(candidate.resolve())

        # Fall back to joining with RELION dir even if it does not exist yet
        return str(candidate)

    def _load_particle_reextraction_params(self) -> Dict[str, Any]:
        """Load particle re-extraction parameters from configuration with sensible defaults."""
        if hasattr(self, "_particle_reextraction_params"):
            return dict(self._particle_reextraction_params)

        params: Dict[str, Any] = {
            "extract_size": 0,
            "norm": True,
            "bg_radius": -1,
            "white_dust": -1,
            "black_dust": -1,
            "invert_contrast": False,
            "only_do_unfinished": False,
            "float16": True,
            "timeout": 86400,
            "use_backend": False,
        }

        config_path = Path("configs/relion/reconstruction_config.json")
        try:
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                workflow_cfg = config_data.get("workflow", {})
                reextract_cfg = workflow_cfg.get("particle_reextraction", {})
                for key, value in reextract_cfg.items():
                    if value is not None:
                        params[key] = value
        except Exception as exc:
            self.logger.warning(f"Could not load particle re-extraction configuration: {exc}")

        self._particle_reextraction_params = dict(params)
        return dict(params)

    def _load_relion_particle_results(self, stage_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load cached RELION particle picking results from disk when stage outputs are unavailable."""
        candidate_paths: List[Path] = []

        result_file = stage_outputs.get("result_file")
        if isinstance(result_file, str):
            candidate_paths.append(Path(result_file))

        # Allow stage outputs to directly provide a JSON blob
        serialized = stage_outputs.get("relion_particle_results")
        if isinstance(serialized, dict):
            return dict(serialized)

        outputs_dir = Path("outputs")
        if outputs_dir.exists():
            pattern = "particle_picking_results_relion_*.json"
            files = sorted(outputs_dir.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
            candidate_paths.extend(files)

        seen: set[Path] = set()
        for path in candidate_paths:
            try:
                if not path:
                    continue
                resolved = path.expanduser().resolve()
                if resolved in seen or not resolved.exists():
                    continue
                seen.add(resolved)
                with open(resolved, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception as exc:
                self.logger.debug(f"Could not load particle picking results from {path}: {exc}")

        return {}

    def get_stage_backend(self, stage_name: str) -> Optional[BackendType]:
        """
        Get the backend type for a given stage from master config.
        
        Args:
            stage_name: Name of the stage (e.g., "preprocessing", "particle_picking")
            
        Returns:
            BackendType or None if not found
        """
        if not self.master_config or "master_workflow" not in self.master_config:
            return None
        
        stages = self.master_config["master_workflow"].get("stages", [])
        for stage in stages:
            if stage.get("name") == stage_name:
                agent_group = stage.get("agent_group", "").lower()
                if "relion" in agent_group:
                    return BackendType.RELION
                elif "cryosparc" in agent_group:
                    return BackendType.CRYOSPARC
                else:
                    # Default to CryoSparc if not specified
                    return BackendType.CRYOSPARC
        
        return None
    
    def check_transition_needed(
        self,
        current_stage: str,
        next_stage: str
    ) -> Tuple[bool, Optional[BackendType], Optional[BackendType]]:
        """
        Check if a transition is needed between stages.
        
        Args:
            current_stage: Name of the current stage
            next_stage: Name of the next stage
            
        Returns:
            Tuple of (needs_transition, current_backend, next_backend)
        """
        current_backend = self.get_stage_backend(current_stage)
        next_backend = self.get_stage_backend(next_stage)
        
        if current_backend is None or next_backend is None:
            self.logger.warning(f"Could not determine backend for stages: {current_stage} -> {next_stage}")
            return False, current_backend, next_backend
        
        needs_transition = current_backend != next_backend
        
        if needs_transition:
            self.logger.info(
                f"Transition needed: {current_stage} ({current_backend.value}) -> "
                f"{next_stage} ({next_backend.value})"
            )
        else:
            self.logger.info(
                f"No transition needed: {current_stage} ({current_backend.value}) -> "
                f"{next_stage} ({next_backend.value})"
            )
        
        return needs_transition, current_backend, next_backend
    
    def check_existing_transition(
        self,
        current_stage: str,
        next_stage: str,
        current_stage_outputs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Check if a transition has already been performed by reading transition files.
        
        Args:
            current_stage: Name of the current stage
            next_stage: Name of the next stage
            current_stage_outputs: Dictionary of outputs from the current stage
            
        Returns:
            Dictionary with existing transition data if found, None otherwise
        """
        self.logger.info(f"🔍 [check_existing_transition] Starting check for {current_stage} -> {next_stage}")
        
        # Use absolute path to ensure we find the transitions directory
        transitions_dir = Path("outputs") / "transitions"
        if not transitions_dir.is_absolute():
            # Try to resolve relative to current working directory
            transitions_dir = Path.cwd() / transitions_dir
        
        self.logger.info(f"📁 [check_existing_transition] Looking for transition files in: {transitions_dir.absolute()}")
        self.logger.info(f"📁 [check_existing_transition] Current working directory: {Path.cwd()}")
        
        if not transitions_dir.exists():
            self.logger.warning(f"❌ [check_existing_transition] Transitions directory does not exist: {transitions_dir.absolute()}")
            return None
        
        self.logger.info(f"✅ [check_existing_transition] Transitions directory exists: {transitions_dir.absolute()}")
        
        # Get backend types to determine transition direction
        needs_transition, current_backend, next_backend = self.check_transition_needed(
            current_stage, next_stage
        )
        
        if not needs_transition:
            return None
        
        # Look for transition files matching the stage and direction
        pattern = f"transition_{current_stage}_*.json"
        self.logger.info(f"🔎 [check_existing_transition] Searching for files matching pattern: {pattern}")
        transition_files = list(transitions_dir.glob(pattern))
        
        self.logger.info(f"📊 [check_existing_transition] Found {len(transition_files)} files matching pattern '{pattern}' in {transitions_dir}")
        if transition_files:
            self.logger.info(f"📋 [check_existing_transition] Files found: {[f.name for f in transition_files]}")
        
        if not transition_files:
            self.logger.info("No transition files found")
            return None
        
        # Sort by modification time (most recent first)
        transition_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Check each transition file to see if it matches
        for transition_file in transition_files:
            try:
                with open(transition_file, 'r') as f:
                    transition_data = json.load(f)
                
                transition_info = transition_data.get("transition_info", {})
                if not transition_info:
                    self.logger.warning(f"Transition file {transition_file.name} has no transition_info, skipping")
                    continue
                
                file_from_agent = transition_info.get("from_agent", "").lower().strip()
                file_to_agent = transition_info.get("to_agent", "").lower().strip()
                file_stage = transition_info.get("stage", "").strip()
                
                # Check if this transition matches what we need
                expected_from = current_backend.value.lower() if current_backend else ""
                expected_to = next_backend.value.lower() if next_backend else ""
                
                self.logger.info(f"📄 Checking transition file {transition_file.name}:")
                self.logger.info(f"   File transition_info: from='{file_from_agent}', to='{file_to_agent}', stage='{file_stage}'")
                self.logger.info(f"   Expected: from='{expected_from}', to='{expected_to}', stage='{current_stage}'")
                
                # Check if direction and stage match
                direction_match = (file_from_agent == expected_from and file_to_agent == expected_to)
                stage_match = (file_stage == current_stage)
                
                self.logger.info(f"   Direction match: {direction_match} (from: {file_from_agent == expected_from}, to: {file_to_agent == expected_to})")
                self.logger.info(f"   Stage match: {stage_match}")
                
                if direction_match and stage_match:
                    self.logger.info(f"✅ Found matching transition file (direction and stage match): {transition_file.name}")
                    
                    # For CryoSparc -> Relion, check if source_job_uid matches
                    if current_backend == BackendType.CRYOSPARC:
                        source_job_uid = transition_info.get("source_job_uid")
                        current_job_uid = current_stage_outputs.get("micrograph_selection_job_uid")
                        self.logger.debug(f"CryoSparc->Relion: source_job_uid={source_job_uid}, current_job_uid={current_job_uid}")
                        if source_job_uid and current_job_uid and source_job_uid == current_job_uid:
                            self.logger.info(f"Found existing transition with matching job UID: {transition_file}")
                            return transition_data
                        # Fallback: check if STAR file exists
                        relion_job = transition_data.get("relion_job", {})
                        star_file = relion_job.get("star_file_absolute")
                        if star_file and Path(star_file).exists():
                            self.logger.info(f"Found existing transition with valid STAR file: {transition_file}")
                            return transition_data
                    
                    # For Relion -> CryoSparc, check if motion_correction_job_dir matches OR if CryoSparc job exists
                    elif current_backend == BackendType.RELION:
                        # First try to match by motion_correction_job_dir
                        relion_data = transition_data.get("relion_data", {})
                        file_motion_corr_dir = relion_data.get("motion_correction_job_dir")
                        current_motion_corr_dir = current_stage_outputs.get("motion_correction_job_dir")
                        self.logger.debug(f"Relion->CryoSparc: file_motion_corr_dir={file_motion_corr_dir}, current_motion_corr_dir={current_motion_corr_dir}")
                        
                        if file_motion_corr_dir and current_motion_corr_dir and file_motion_corr_dir == current_motion_corr_dir:
                            self.logger.info(f"Found existing transition with matching motion_correction_job_dir: {transition_file}")
                            return transition_data
                        
                        # Fallback: check if CryoSparc jobs still exist (this is more reliable)
                        # For Relion -> CryoSparc, if we have a matching transition file with the right direction/stage,
                        # we should use it. The job verification is optional.
                        cryosparc_jobs = transition_data.get("cryosparc_jobs", {})
                        selection_job_uid = cryosparc_jobs.get("selection_job_uid")
                        preprocessing_outputs = transition_data.get("preprocessing_outputs", {})
                        # Also check preprocessing_outputs for the job UID
                        if not selection_job_uid:
                            selection_job_uid = preprocessing_outputs.get("micrograph_selection_job_uid")
                        
                        self.logger.info(f"Relion->CryoSparc transition found: selection_job_uid={selection_job_uid}")
                        
                        # If we have a selection_job_uid, try to verify it exists
                        if selection_job_uid:
                            if self.cryosparc_tools:
                                try:
                                    project_uid = current_stage_outputs.get("project_uid") or preprocessing_outputs.get("project_uid", "P1")
                                    self.logger.info(f"Verifying CryoSparc job {selection_job_uid} in project {project_uid}")
                                    job_info = self.cryosparc_tools.get_job_output_directory(
                                        project_uid,
                                        selection_job_uid
                                    )
                                    if job_info:
                                        self.logger.info(f"✓ Found existing transition with valid CryoSparc job {selection_job_uid}: {transition_file}")
                                        return transition_data
                                    else:
                                        self.logger.warning(f"Job {selection_job_uid} not found, but transition file exists")
                                except Exception as e:
                                    self.logger.warning(f"Could not verify CryoSparc job {selection_job_uid}: {e}, but using transition file anyway")
                                    # Even if verification fails, use the transition file if it matches
                                    self.logger.info(f"Using existing transition file despite verification error: {transition_file}")
                                    return transition_data
                            else:
                                # If we can't verify, but the transition file exists and matches direction/stage, use it
                                # This handles cases where CryoSparc tools aren't available but we trust the transition file
                                self.logger.info(f"Using existing transition file (CryoSparc tools not available for verification): {transition_file}")
                                return transition_data
                        else:
                            # Even without job UID, if the file matches direction/stage, use it
                            # The transition was successful before, so trust it
                            self.logger.info(f"Using existing transition file (no job UID to verify, but matches direction/stage): {transition_file}")
                            return transition_data
                
            except Exception as e:
                self.logger.warning(f"Error reading transition file {transition_file}: {e}")
                import traceback
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
                continue
        
        self.logger.warning("No matching existing transition found after checking all files")
        self.logger.warning(f"Checked {len(transition_files)} transition files, none matched the criteria")
        return None
    
    def perform_transition(
        self,
        current_stage: str,
        next_stage: str,
        current_stage_outputs: Dict[str, Any],
        project_uid: str,
        workspace_uid: str,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform format transition between stages if needed.
        
        Args:
            current_stage: Name of the current stage
            next_stage: Name of the next stage
            current_stage_outputs: Dictionary of outputs from the current stage
            project_uid: CryoSparc project UID
            workspace_uid: CryoSparc workspace UID
            conversation_id: Optional conversation ID for tracking
            
        Returns:
            Dictionary with success status, converted_outputs, and error if any
        """
        # Check if transition is needed
        needs_transition, current_backend, next_backend = self.check_transition_needed(
            current_stage, next_stage
        )
        
        if not needs_transition:
            # No transition needed, return success with original outputs
            return {
                "success": True,
                "converted_outputs": current_stage_outputs,
                "transition_info": None
            }
        
        # Check if transition has already been done
        self.logger.info(f"🔍 [perform_transition] Checking for existing transition: {current_stage} ({current_backend.value if current_backend else 'unknown'}) -> {next_stage} ({next_backend.value if next_backend else 'unknown'})")
        self.logger.info(f"   [perform_transition] Current stage outputs keys: {list(current_stage_outputs.keys())[:10]}...")  # Show first 10 keys
        
        existing_transition = self.check_existing_transition(
            current_stage, next_stage, current_stage_outputs
        )
        
        if existing_transition:
            self.logger.info(f"✅ [perform_transition] Found existing transition, skipping conversion")
            self.logger.info(f"Using existing transition for {current_stage} -> {next_stage}")
            # Extract converted outputs from existing transition
            converted_outputs = {}
            
            if current_backend == BackendType.CRYOSPARC and next_backend == BackendType.RELION:
                # CryoSparc -> Relion: extract Relion job info
                relion_job = existing_transition.get("relion_job", {})
                converted_outputs["selected_micrographs_star"] = relion_job.get("star_file_absolute")
                converted_outputs["selection_job_dir"] = relion_job.get("job_dir")
                converted_outputs["transition_config"] = str(Path(relion_job.get("job_dir_absolute", "")) / "transition_config.json")
                converted_outputs["transition_info"] = existing_transition.get("transition_info")
            elif current_backend == BackendType.RELION and next_backend == BackendType.CRYOSPARC:
                # Relion -> CryoSparc: extract CryoSparc job info
                cryosparc_jobs = existing_transition.get("cryosparc_jobs", {})
                preprocessing_outputs = existing_transition.get("preprocessing_outputs", {})
                converted_outputs["micrograph_selection_job_uid"] = preprocessing_outputs.get("micrograph_selection_job_uid") or cryosparc_jobs.get("selection_job_uid")
                converted_outputs["import_job_uid"] = cryosparc_jobs.get("import_job_uid")
                converted_outputs["ctf_job_uid"] = cryosparc_jobs.get("ctf_job_uid")
                converted_outputs["project_uid"] = preprocessing_outputs.get("project_uid")
                converted_outputs["workspace_uid"] = preprocessing_outputs.get("workspace_uid")
                converted_outputs["transition_info"] = existing_transition.get("transition_info")
            
            return {
                "success": True,
                "converted_outputs": converted_outputs,
                "transition_info": existing_transition.get("transition_info"),
                "from_cache": True
            }
        else:
            self.logger.info(f"❌ [perform_transition] No existing transition found, will perform new conversion")
        
        try:
            # Perform the appropriate conversion
            if current_backend == BackendType.CRYOSPARC and next_backend == BackendType.RELION:
                # CryoSparc -> Relion
                result = self.convert_cryosparc_to_relion(
                    stage_name=current_stage,
                    stage_outputs=current_stage_outputs,
                    conversation_id=conversation_id
                )
            elif current_backend == BackendType.RELION and next_backend == BackendType.CRYOSPARC:
                # Relion -> CryoSparc
                result = self.convert_relion_to_cryosparc(
                    stage_name=current_stage,
                    stage_outputs=current_stage_outputs,
                    project_uid=project_uid,
                    workspace_uid=workspace_uid
                )
            else:
                return {
                    "success": False,
                    "error": f"Unsupported transition: {current_backend} -> {next_backend}",
                    "converted_outputs": {}
                }
            
            # Ensure result has success field
            if "success" not in result:
                result["success"] = True
            
            return result
            
        except Exception as e:
            self.logger.error(f"Transition failed: {e}")
            import traceback
            self.logger.error(f"Transition error traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "converted_outputs": {}
            }
    
    def convert_cryosparc_to_relion(
        self,
        stage_name: str,
        stage_outputs: Dict[str, Any],
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convert CryoSparc outputs to Relion format (STAR files) with proper directory structure.
        
        Creates:
        - Relion job directory (e.g., Select/job001, Import/job001)
        - STAR file in the Relion directory
        - JSON config file describing the conversion
        
        Uses agentic workflow to monitor and verify the conversion.
        
        Args:
            stage_name: Name of the stage (preprocessing, particle_picking, reconstruction)
            stage_outputs: Dictionary of stage outputs from CryoSparc
            conversation_id: Optional conversation ID for tracking
            
        Returns:
            Dictionary with converted Relion format outputs
        """
        if not self.relion_tools:
            self.logger.error("Relion tools not initialized")
            return {
                "success": False,
                "error": "Relion tools not initialized",
                "converted_outputs": {}
            }
        
        relion_dir = Path(self.relion_tools.relion_dir)
        project_uid = stage_outputs.get("project_uid", self.master_config.get("workflow", {}).get("project_uid", "P1"))
        
        # Create transition workflow for agentic monitoring
        workflow = TransitionWorkflow(
            transition_agent=self,
            stage_name=stage_name,
            stage_outputs=stage_outputs,
            project_uid=project_uid
        )
        
        # Run the conversion workflow
        result = workflow.run(conversation_id=conversation_id)
        
        # Return the full result dictionary (includes success, converted_outputs, etc.)
        return result
    
    def convert_relion_to_cryosparc(
        self,
        stage_name: str,
        stage_outputs: Dict[str, Any],
        project_uid: str,
        workspace_uid: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convert Relion outputs to CryoSparc format (import STAR files).
        
        Args:
            stage_name: Name of the stage (preprocessing, particle_picking, reconstruction)
            stage_outputs: Dictionary of stage outputs from Relion
            project_uid: CryoSparc project UID
            workspace_uid: CryoSparc workspace UID
            output_dir: Directory to save intermediate files (default: outputs/transitions)
            
        Returns:
            Dictionary with converted CryoSparc format outputs (job UIDs)
        """
        # Try to initialize CryoSparc tools if not already initialized
        if not self.cryosparc_tools:
            self.logger.warning("CryoSparc tools not initialized, attempting to initialize...")
            try:
                self._initialize_tools()
            except Exception as e:
                self.logger.error(f"Failed to initialize CryoSparc tools: {e}")
        
        if not self.cryosparc_tools:
            self.logger.error("CryoSparc tools not initialized and could not be initialized")
            return {}
        
        if output_dir is None:
            output_dir = Path("outputs") / "transitions"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        converted_outputs = {}
        
        if stage_name == "preprocessing":
            # Get MotionCorr job directory from RELION preprocessing outputs
            motion_correction_job_dir = stage_outputs.get("motion_correction_job_dir")
            if not motion_correction_job_dir or not os.path.exists(motion_correction_job_dir):
                self.logger.error(f"Motion correction job directory not found: {motion_correction_job_dir}")
                converted_outputs["conversion_status"] = "failed"
                converted_outputs["error"] = "Motion correction job directory not found"
                return converted_outputs
            
            try:
                self.logger.info(f"Converting RELION preprocessing to CryoSparc format")
                self.logger.info(f"Motion correction directory: {motion_correction_job_dir}")
                
                # Get MotionCorr path
                motion_corr_path = Path(motion_correction_job_dir)
                
                # Create micrograph folder as a subfolder under MotionCorr
                micrograph_folder = motion_corr_path / "Micrographs"
                
                # Clean up the Micrographs folder first (remove all existing links/files)
                if micrograph_folder.exists():
                    self.logger.info(f"Cleaning up existing Micrographs folder: {micrograph_folder}")
                    import shutil
                    try:
                        # Remove all contents
                        for item in micrograph_folder.iterdir():
                            try:
                                if item.is_symlink() or item.is_file():
                                    item.unlink()
                                elif item.is_dir():
                                    shutil.rmtree(item)
                            except Exception as e:
                                self.logger.warning(f"Could not remove {item}: {e}")
                    except Exception as e:
                        self.logger.warning(f"Error cleaning up Micrographs folder: {e}")
                else:
                    micrograph_folder.mkdir(exist_ok=True)
                    self.logger.info(f"Created micrograph folder: {micrograph_folder}")
                
                # Find all *.mrc files in MotionCorr directory and subdirectories (but not *_PS.mrc)
                # Exclude the Micrographs folder itself from the search
                import glob
                motion_corr_mrc_files = []
                micrograph_folder_str = str(micrograph_folder)
                
                # Search in the MotionCorr directory and all subdirectories
                for mrc_file in glob.glob(str(motion_corr_path / "**" / "*.mrc"), recursive=True):
                    # Skip files in the Micrographs folder and *_PS.mrc files
                    if micrograph_folder_str in mrc_file:
                        continue
                    if not mrc_file.endswith("_PS.mrc"):
                        motion_corr_mrc_files.append(Path(mrc_file))
                
                if not motion_corr_mrc_files:
                    raise ValueError(f"No motion-corrected .mrc files found in {motion_corr_path} or subdirectories (excluding *_PS.mrc and Micrographs folder)")
                
                self.logger.info(f"Found {len(motion_corr_mrc_files)} motion-corrected micrograph files")
                
                # Link all *.mrc files (except *_PS.mrc) to the micrograph folder
                linked_files = []
                for mrc_file in motion_corr_mrc_files:
                    # Get absolute path without resolving symlinks (to avoid loops)
                    mrc_file_abs = Path(mrc_file).absolute()
                    link_path = micrograph_folder / mrc_file_abs.name
                    
                    # Create symbolic link (use absolute path for target)
                    try:
                        link_path.symlink_to(mrc_file_abs)
                        linked_files.append(link_path)
                        self.logger.debug(f"Linked {mrc_file_abs.name} -> {link_path}")
                    except FileExistsError:
                        # If it exists, remove and try again
                        try:
                            link_path.unlink()
                            link_path.symlink_to(mrc_file_abs)
                            linked_files.append(link_path)
                            self.logger.debug(f"Re-linked {mrc_file_abs.name} -> {link_path}")
                        except Exception as e:
                            self.logger.error(f"Failed to create symlink {link_path} -> {mrc_file_abs}: {e}")
                            raise
                    except Exception as e:
                        self.logger.error(f"Failed to create symlink {link_path} -> {mrc_file_abs}: {e}")
                        raise
                
                self.logger.info(f"Linked {len(linked_files)} micrograph files to {micrograph_folder}")
                
                # Get pixel size and other parameters from STAR file or use defaults
                star_file = stage_outputs.get("selected_micrographs_star")
                pixel_size = 1.0
                voltage = 300.0
                cs_mm = 2.7
                dose = 53.0  # Default dose
                
                # Try to load dose from microscope_config.json
                try:
                    microscope_config_path = self.master_config.get("workflow", {}).get("microscope_config_path", "configs/microscope_config.json")
                    if not Path(microscope_config_path).is_absolute():
                        microscope_config_path = Path.cwd() / microscope_config_path
                    
                    if Path(microscope_config_path).exists():
                        with open(microscope_config_path, 'r') as f:
                            microscope_config = json.load(f)
                            dose = microscope_config.get("microscope_parameters", {}).get("dose", dose)
                            self.logger.info(f"Loaded dose from microscope config: {dose} e-/Å²")
                except Exception as e:
                    self.logger.warning(f"Could not load dose from microscope config, using default: {e}")
                
                if star_file and os.path.exists(star_file):
                    try:
                        import pandas as pd
                        # Read the STAR file to get metadata
                        with open(star_file, 'r') as f:
                            lines = f.readlines()
                        
                        in_optics = False
                        optics_headers = []
                        optics_data = []
                        
                        for line in lines:
                            line_stripped = line.strip()
                            if not line_stripped or line_stripped.startswith("#"):
                                continue
                            
                            if line_stripped.startswith("data_optics"):
                                in_optics = True
                                continue
                            elif line_stripped.startswith("data_"):
                                in_optics = False
                                continue
                            
                            if line_stripped.startswith("loop_"):
                                continue
                            
                            if in_optics and line_stripped.startswith("_rln"):
                                optics_headers.append(line_stripped.split()[0][1:])
                                continue
                            
                            if in_optics and line_stripped and not line_stripped.startswith("_"):
                                optics_data.append(line_stripped.split())
                        
                        if optics_headers and optics_data:
                            optics_df = pd.DataFrame(optics_data, columns=optics_headers[:len(optics_data[0]) if optics_data else 0])
                            if len(optics_df) > 0:
                                pixel_size = optics_df.iloc[0].get("rlnMicrographOriginalPixelSize", pixel_size)
                                voltage = optics_df.iloc[0].get("rlnVoltage", voltage)
                                cs_mm = optics_df.iloc[0].get("rlnSphericalAberration", cs_mm)
                                
                                # Convert to float and handle NaN
                                if pd.notna(pixel_size):
                                    pixel_size = float(pixel_size)
                                if pd.notna(voltage):
                                    voltage = float(voltage)
                                if pd.notna(cs_mm):
                                    cs_mm = float(cs_mm)
                    except Exception as e:
                        self.logger.warning(f"Could not read metadata from STAR file, using defaults: {e}")
                
                # Create pattern for all linked micrograph files
                micrograph_pattern = str(micrograph_folder / "*.mrc")
                
                self.logger.info(f"Importing micrographs using pattern: {micrograph_pattern}")
                self.logger.info(f"Parameters: pixel_size={pixel_size}, voltage={voltage}, cs_mm={cs_mm}, dose={dose} e-/Å²")
                
                # Import micrographs into CryoSparc using import_micrographs
                project = self.cryosparc_tools.cs.find_project(project_uid)
                workspace = project.find_workspace(workspace_uid)
                
                job_params = {
                    "blob_paths": micrograph_pattern,
                    "psize_A": float(pixel_size),
                    "accel_kv": float(voltage),
                    "cs_mm": float(cs_mm),
                    "total_dose_e_per_A2": float(dose),
                }
                
                self.logger.info(f"Creating import_micrographs job")
                import_job = workspace.create_job("import_micrographs", params=job_params)
                # Queue with default lane (auto-detect if needed)
                try:
                    import_job.queue()
                except Exception as queue_error:
                    message = str(queue_error)
                    if "Must specify a lane" in message:
                        try:
                            lanes = self.cryosparc_tools.cs.get_lanes()
                            if lanes:
                                used_lane = lanes[0]["name"]
                                self.logger.info(f"No lane specified; using default lane '{used_lane}'")
                                import_job.queue(lane=used_lane)
                            else:
                                raise queue_error
                        except Exception:
                            raise queue_error
                    else:
                        raise queue_error
                import_job_uid = import_job.uid
                self.logger.info(f"Queued import_micrographs job: {import_job_uid}")
                
                # Wait for import to complete
                self.logger.info(f"Waiting for import_micrographs job {import_job_uid} to complete...")
                final_status = self.cryosparc_tools.wait_for_job_completion(
                    project_uid=project_uid,
                    job_uid=import_job_uid,
                    workspace_uid=workspace_uid,
                    timeout=3600,
                    check_interval=30
                )
                
                if final_status.get("status") != "completed":
                    raise RuntimeError(f"Import job finished with status: {final_status.get('status')}")
                
                self.logger.info(f"Successfully imported micrographs: job {import_job_uid}")
                
                # Run CTF estimation in CryoSparc
                # Note: import_micrographs outputs "exposures", so we need to connect to that
                self.logger.info("Running CTF estimation in CryoSparc...")
                
                # Create CTF estimation job manually to ensure correct connection
                project = self.cryosparc_tools.cs.find_project(project_uid)
                workspace = project.find_workspace(workspace_uid)
                
                # Get actual output labels from the import job
                available_output_labels = []
                try:
                    import_job = project.find_job(import_job_uid)
                    import_job.refresh()
                    job_doc = getattr(import_job, "doc", {})
                    output_result_groups = job_doc.get("output_result_groups", [])
                    for group in output_result_groups:
                        label = group.get("name")
                        if label:
                            available_output_labels.append(label)
                    self.logger.info(f"Available output labels from import job {import_job_uid}: {available_output_labels}")
                except Exception as e:
                    self.logger.warning(f"Could not get output labels from job, using defaults: {e}")
                
                # Try different output labels from import_micrographs
                # Start with available labels, then fall back to defaults
                output_labels = available_output_labels if available_output_labels else ["exposures", "micrographs", "exposures_accepted"]
                ctf_job = None
                last_error = None
                
                for output_label in output_labels:
                    try:
                        ctf_job = workspace.create_job(
                            "patch_ctf_estimation_multi",
                            connections={"exposures": (import_job_uid, output_label)}
                        )
                        self.logger.info(f"Successfully connected CTF estimation to import job with label: {output_label}")
                        break
                    except Exception as e:
                        last_error = str(e)
                        self.logger.debug(f"Failed to connect with label {output_label}: {e}")
                        continue
                
                if not ctf_job:
                    error_msg = f"Could not create CTF estimation job with any output label from import job {import_job_uid}"
                    if available_output_labels:
                        error_msg += f". Available labels: {available_output_labels}"
                    if last_error:
                        error_msg += f". Last error: {last_error}"
                    raise RuntimeError(error_msg)
                
                # Queue with default lane (auto-detect if needed)
                try:
                    ctf_job.queue()
                except Exception as queue_error:
                    message = str(queue_error)
                    if "Must specify a lane" in message:
                        try:
                            lanes = self.cryosparc_tools.cs.get_lanes()
                            if lanes:
                                used_lane = lanes[0]["name"]
                                self.logger.info(f"No lane specified; using default lane '{used_lane}'")
                                ctf_job.queue(lane=used_lane)
                            else:
                                raise queue_error
                        except Exception:
                            raise queue_error
                    else:
                        raise queue_error
                ctf_job_uid = ctf_job.uid
                self.logger.info(f"Queued CTF estimation job: {ctf_job_uid}")
                
                # Wait for CTF estimation to complete
                final_status = self.cryosparc_tools.wait_for_job_completion(
                    project_uid=project_uid,
                    job_uid=ctf_job_uid,
                    workspace_uid=workspace_uid,
                    timeout=3600,
                    check_interval=30
                )
                
                if final_status.get("status") != "completed":
                    raise RuntimeError(f"CTF estimation failed with status: {final_status.get('status')}")
                
                self.logger.info(f"CTF estimation completed: job {ctf_job_uid}")
                
                # Run micrograph selection in CryoSparc
                self.logger.info("Running micrograph selection in CryoSparc...")
                
                # Create micrograph selection job manually
                selection_job = workspace.create_job(
                    "curate_exposures_v2",
                    connections={"exposures": (ctf_job_uid, "exposures")}
                )
                
                # Queue with default lane (auto-detect if needed)
                try:
                    selection_job.queue()
                except Exception as queue_error:
                    message = str(queue_error)
                    if "Must specify a lane" in message:
                        try:
                            lanes = self.cryosparc_tools.cs.get_lanes()
                            if lanes:
                                used_lane = lanes[0]["name"]
                                self.logger.info(f"No lane specified; using default lane '{used_lane}'")
                                selection_job.queue(lane=used_lane)
                            else:
                                raise queue_error
                        except Exception:
                            raise queue_error
                    else:
                        raise queue_error
                selection_job_uid = selection_job.uid
                self.logger.info(f"Queued micrograph selection job: {selection_job_uid}")
                
                # Wait for job to reach waiting status (interactive mode)
                selection_job.wait_for_status("waiting")
                self.logger.info(f"Job {selection_job_uid} reached waiting status, configuring thresholds...")
                
                # Get fields and thresholds data
                data = selection_job.interact("get_fields_and_thresholds")
                
                # Load min_resolution from preprocessing config (default to 5.0)
                min_resolution = 5.0
                try:
                    preprocessing_config_path = self.master_config.get("workflow", {}).get("stages", {}).get("preprocessing", {}).get("config_path")
                    if preprocessing_config_path and "cryosparc" in preprocessing_config_path:
                        import json as json_module
                        if not Path(preprocessing_config_path).is_absolute():
                            preprocessing_config_path = Path.cwd() / preprocessing_config_path
                        if Path(preprocessing_config_path).exists():
                            with open(preprocessing_config_path, 'r') as f:
                                preprocessing_config = json_module.load(f)
                                min_resolution = preprocessing_config.get("workflow", {}).get("micrograph_selection", {}).get("min_resolution", 5.0)
                                self.logger.info(f"Loaded min_resolution from config: {min_resolution} Å")
                except Exception as e:
                    self.logger.warning(f"Could not load min_resolution from config, using default 5.0 Å: {e}")
                
                # Find the CTF resolution field and set threshold
                from cryosparc.util import first
                ctf_res_field = first(field for field in data["fields"] if field["name"] == "ctf_fit_to_A")
                
                if ctf_res_field:
                    # Set threshold to filter micrographs with resolution better than min_resolution Angstroms
                    # Keep micrographs with resolution 1 to min_resolution Å
                    ctf_res_field["thresholds"] = [1, float(min_resolution)]
                    ctf_res_field["active"] = True
                    self.logger.info(f"Set CTF resolution threshold to {min_resolution} Angstroms (from preprocessing config)")
                else:
                    self.logger.warning("Could not find 'ctf_fit_to_A' field in CTF data")
                
                # Apply the thresholds
                selection_job.interact("set_thresholds", data)
                selection_job.interact("shutdown_interactive")
                self.logger.info("Applied thresholds and submitted micrograph selection")
                
                # Wait for selection to complete
                final_status = self.cryosparc_tools.wait_for_job_completion(
                    project_uid=project_uid,
                    job_uid=selection_job_uid,
                    workspace_uid=workspace_uid,
                    timeout=3600,
                    check_interval=30
                )
                
                if final_status.get("status") != "completed":
                    raise RuntimeError(f"Micrograph selection failed with status: {final_status.get('status')}")
                
                self.logger.info(f"Micrograph selection completed: job {selection_job_uid}")
                
                # Generate transition JSON file in outputs/transitions directory
                transitions_dir = Path("outputs") / "transitions"
                transitions_dir.mkdir(parents=True, exist_ok=True)
                transition_json_path = transitions_dir / f"transition_{stage_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                transition_data = {
                    "transition_info": {
                        "from_agent": "relion",
                        "to_agent": "cryosparc",
                        "stage": stage_name,
                        "conversion_timestamp": datetime.datetime.now().isoformat()
                    },
                    "relion_data": {
                        "motion_correction_job_dir": str(motion_correction_job_dir),
                        "micrograph_folder": str(micrograph_folder),
                        "num_micrographs": len(linked_files)
                    },
                    "cryosparc_jobs": {
                        "import_job_uid": import_job_uid,
                        "ctf_job_uid": ctf_job_uid,
                        "selection_job_uid": selection_job_uid
                    },
                    "conversion_metadata": {
                        "pixel_size": pixel_size,
                        "voltage": voltage,
                        "cs_mm": cs_mm,
                        "dose": dose
                    },
                    "preprocessing_outputs": {
                        "micrograph_selection_job_uid": selection_job_uid,
                        "project_uid": project_uid,
                        "workspace_uid": workspace_uid
                    }
                }
                
                with open(transition_json_path, 'w') as f:
                    json.dump(transition_data, f, indent=2)
                
                self.logger.info(f"Generated transition JSON: {transition_json_path}")
                
                # Return converted outputs in the format expected by the picking agent
                # The picking agent expects 'micrograph_selection_job_uid' in preprocessing_outputs
                converted_outputs["micrograph_selection_job_uid"] = selection_job_uid
                converted_outputs["project_uid"] = project_uid
                converted_outputs["workspace_uid"] = workspace_uid
                converted_outputs["import_job_uid"] = import_job_uid
                converted_outputs["ctf_job_uid"] = ctf_job_uid
                converted_outputs["conversion_status"] = "completed"
                converted_outputs["num_micrographs"] = len(linked_files)
                converted_outputs["micrograph_folder"] = str(micrograph_folder)
                converted_outputs["transition_json"] = str(transition_json_path)
                
                # Also include micrograph_location for compatibility with preprocessing parser format
                try:
                    # Get the job output directory for micrograph_location
                    job_info = self.cryosparc_tools.get_job_output_directory(
                        project_uid, selection_job_uid
                    )
                    micrograph_output_directory = job_info.get("job_directory")
                    
                    converted_outputs["micrograph_location"] = {
                        "description": "Location of processed micrographs in CryoSPARC (converted from RELION)",
                        "project_uid": project_uid,
                        "workspace_uid": workspace_uid,
                        "final_selection_job_uid": selection_job_uid,
                        "output_directory": micrograph_output_directory,
                        "path_pattern": f"Project {project_uid}, Workspace {workspace_uid}, Job {selection_job_uid}",
                        "source": "relion_preprocessing_transition"
                    }
                except Exception as e:
                    self.logger.warning(f"Could not get micrograph output directory: {e}")
                    # Still include basic micrograph_location info
                    converted_outputs["micrograph_location"] = {
                        "description": "Location of processed micrographs in CryoSPARC (converted from RELION)",
                        "project_uid": project_uid,
                        "workspace_uid": workspace_uid,
                        "final_selection_job_uid": selection_job_uid,
                        "source": "relion_preprocessing_transition"
                    }
                
            except Exception as e:
                self.logger.error(f"Failed to convert Relion preprocessing output: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                converted_outputs["conversion_status"] = "failed"
                converted_outputs["error"] = str(e)
        
        elif stage_name == "particle_picking":
            try:
                reextraction_params = self._load_particle_reextraction_params()

                # Resolve selected particles STAR file (from RELION Select job)
                particle_star_candidates: List[str] = []
                for key in [
                    "reextracted_particles_star",
                    "selected_particles_star",
                    "final_star_file",
                    "auto_2d_selection_2_output_file",
                    "auto_2d_selection_output_file",
                ]:
                    value = stage_outputs.get(key)
                    if isinstance(value, str):
                        particle_star_candidates.append(value)

                cached_results = self._load_relion_particle_results(stage_outputs)
                if cached_results:
                    stage_outputs.setdefault("relion_particle_results", cached_results)
                    for key in [
                        "reextracted_particles_star",
                        "selected_particles_star",
                        "final_star_file",
                    ]:
                        cache_val = cached_results.get(key)
                        if isinstance(cache_val, str):
                            particle_star_candidates.append(cache_val)

                # Derive from job directories if necessary
                for job_key in ["auto_2d_selection_2_job_dir", "auto_2d_selection_job_dir"]:
                    job_dir_value = stage_outputs.get(job_key)
                    job_dir_path = self._resolve_relion_path(job_dir_value) if job_dir_value else None
                    if job_dir_path:
                        candidate = Path(job_dir_path) / "particles.star"
                        particle_star_candidates.append(str(candidate))

                reextract_data_star: Optional[str] = None
                for candidate in particle_star_candidates:
                    resolved = self._resolve_relion_path(candidate)
                    if resolved:
                        reextract_data_star = resolved
                        break

                if not reextract_data_star:
                    raise FileNotFoundError(
                        "Could not locate RELION particles STAR file to re-extract. "
                        "Expected keys include 'selected_particles_star' or 'final_star_file'."
                    )

                # Resolve micrographs STAR file used for extraction (original pixel size)
                micrographs_star = stage_outputs.get("selected_micrographs_star")
                micrographs_star = self._resolve_relion_path(micrographs_star)

                if not micrographs_star and cached_results:
                    micro_candidate = cached_results.get("micrograph_location")
                    if isinstance(micro_candidate, dict):
                        for key in [
                            "selected_micrographs_star",
                            "micrographs_star",
                            "path",
                            "output_file",
                        ]:
                            val = micro_candidate.get(key)
                            if isinstance(val, str):
                                micrographs_star = self._resolve_relion_path(val)
                                if micrographs_star:
                                    break
                    elif isinstance(micro_candidate, str):
                        micrographs_star = self._resolve_relion_path(micro_candidate)

                if not micrographs_star:
                    # Try to infer from Select job directories
                    select_dirs = []
                    for job_key in ["auto_2d_selection_2_job_dir", "auto_2d_selection_job_dir"]:
                        job_dir_value = stage_outputs.get(job_key)
                        job_dir_path = self._resolve_relion_path(job_dir_value) if job_dir_value else None
                        if job_dir_path:
                            select_dirs.append(Path(job_dir_path))

                    if not select_dirs:
                        relion_dir = Path(getattr(self.relion_tools, "relion_dir", Path.cwd()))
                        select_dirs = sorted(relion_dir.glob("Select/job*"), reverse=True)

                    for select_dir in select_dirs:
                        candidate = select_dir / "micrographs.star"
                        if candidate.exists():
                            micrographs_star = str(candidate.resolve())
                            break

                if not micrographs_star:
                    raise FileNotFoundError(
                        "Could not locate micrographs STAR file required for re-extraction."
                    )

                extract_size = int(reextraction_params.get("extract_size", 0) or 0)
                if extract_size <= 0:
                    raise ValueError(
                        "Re-extraction parameters must define a positive 'extract_size'."
                    )

                used_reextract_params = {
                    "reextract_data_star": reextract_data_star,
                    "micrographs_star": micrographs_star,
                    "output_dir": "ReExtract",
                    "extract_size": extract_size,
                    "norm": bool(reextraction_params.get("norm", True)),
                    "bg_radius": float(reextraction_params.get("bg_radius", -1) or -1),
                    "white_dust": float(reextraction_params.get("white_dust", -1) or -1),
                    "black_dust": float(reextraction_params.get("black_dust", -1) or -1),
                    "invert_contrast": bool(reextraction_params.get("invert_contrast", False)),
                    "only_do_unfinished": bool(reextraction_params.get("only_do_unfinished", False)),
                    "float16": bool(reextraction_params.get("float16", True)),
                    "timeout": int(reextraction_params.get("timeout", 86400)),
                    "use_backend": bool(reextraction_params.get("use_backend", False)),
                }

                reextract_result = self.relion_tools.reextract_particles_original_pixelsize(**used_reextract_params)

                reextract_output_dir = reextract_result.get("output_dir")
                if not reextract_output_dir:
                    raise RuntimeError("RELION re-extraction did not return an output directory")

                reextract_output_dir_path = Path(reextract_output_dir)
                relion_dir = Path(getattr(self.relion_tools, "relion_dir", Path.cwd()))
                try:
                    reextract_job_relative = os.path.relpath(reextract_output_dir_path, relion_dir)
                except ValueError:
                    reextract_job_relative = str(reextract_output_dir_path)

                if reextract_result.get("status") == "running" and used_reextract_params["use_backend"]:
                    timeout = used_reextract_params["timeout"]
                    check_interval = getattr(self.relion_tools, "_backend_check_interval", 30)
                    wait_status = self.relion_tools.wait_for_job_completion(
                        str(reextract_output_dir_path),
                        timeout=timeout,
                        check_interval=check_interval,
                    )
                    if wait_status.get("status") != "completed":
                        raise RuntimeError(
                            f"RELION re-extraction job did not complete successfully: {wait_status}"
                        )

                reextracted_particles_star = reextract_result.get("particles_star")
                if not reextracted_particles_star:
                    reextracted_particles_star = str(reextract_output_dir_path / "particles.star")

                reextracted_particles_star = self._resolve_relion_path(reextracted_particles_star)
                if not reextracted_particles_star or not Path(reextracted_particles_star).exists():
                    raise FileNotFoundError(
                        "Re-extracted particles STAR file not found after RELION re-extraction"
                    )

                # Import re-extracted particles into CryoSPARC
                data_sign = "negative" if used_reextract_params["invert_contrast"] else "positive"
                import_result = self.cryosparc_tools.import_particles_from_star(
                    project_uid=project_uid,
                    workspace_uid=workspace_uid,
                    star_path=reextracted_particles_star,
                    data_sign=data_sign,
                    wait_for_completion=True,
                    timeout=1800,
                )

                import_job_uid = import_result.get("job_uid")
                if not import_job_uid:
                    raise RuntimeError("CryoSPARC import_particles job did not return a job UID")

                # Record transition metadata
                transitions_dir = Path("outputs") / "transitions"
                transitions_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                transition_json_path = transitions_dir / f"transition_{stage_name}_{timestamp}.json"

                transition_data = {
                    "transition_info": {
                        "from_agent": "relion",
                        "to_agent": "cryosparc",
                        "stage": stage_name,
                        "conversion_timestamp": datetime.datetime.now().isoformat(),
                    },
                    "relion_data": {
                        "reextract_data_star": reextract_data_star,
                        "micrographs_star": micrographs_star,
                        "reextracted_particles_star": reextracted_particles_star,
                        "reextraction_job_dir": reextract_job_relative,
                    },
                    "cryosparc_jobs": {
                        "import_job_uid": import_job_uid,
                        "import_status": import_result.get("status"),
                    },
                    "conversion_metadata": {
                        "reextraction_params": used_reextract_params,
                        "data_sign": data_sign,
                    },
                }

                with open(transition_json_path, "w", encoding="utf-8") as f:
                    json.dump(transition_data, f, indent=2)

                converted_outputs.update({
                    "reextracted_particles_star": reextracted_particles_star,
                    "reextraction_job_dir": reextract_job_relative,
                    "reextraction_params": used_reextract_params,
                    "import_particles_job_uid": import_job_uid,
                    "import_job_uid": import_job_uid,
                    "import_particles_status": import_result.get("status"),
                    "particles_job_uid": import_job_uid,
                    "selected_particles_job_uid": import_job_uid,
                    "final_selection_job_uid": import_job_uid,
                    "final_star_file": reextracted_particles_star,
                    "particles_star_file": reextracted_particles_star,
                    "conversion_status": "completed",
                    "transition_json": str(transition_json_path),
                    "project_uid": project_uid,
                    "workspace_uid": workspace_uid,
                })

                converted_outputs.setdefault("outputs", {})
                converted_outputs["outputs"]["selected_particles_job_uid"] = import_job_uid
                converted_outputs.setdefault("job_uids", {})
                converted_outputs["job_uids"]["imported_particles"] = import_job_uid

                # Update cached RELION particle picking results with CryoSPARC import info
                try:
                    outputs_dir = Path("outputs")
                    if outputs_dir.exists():
                        result_files = sorted(
                            outputs_dir.glob("particle_picking_results_relion_*.json"),
                            key=lambda f: f.stat().st_mtime,
                            reverse=True,
                        )
                        if result_files:
                            latest_file = result_files[0]
                            with open(latest_file, "r", encoding="utf-8") as f:
                                picking_data = json.load(f)
                            picking_data.setdefault("conversion", {})
                            picking_data["conversion"].update({
                                "import_job_uid": import_job_uid,
                                "import_status": import_result.get("status"),
                                "reextracted_particles_star": reextracted_particles_star,
                                "reextraction_job_dir": reextract_job_relative,
                                "timestamp": datetime.datetime.now().isoformat(),
                            })
                            picking_data.setdefault("job_uids", {})
                            picking_data["job_uids"]["imported_particles"] = import_job_uid
                            picking_data.setdefault("outputs", {})
                            picking_data["outputs"]["selected_particles_job_uid"] = import_job_uid
                            with open(latest_file, "w", encoding="utf-8") as f:
                                json.dump(picking_data, f, indent=2)
                            self.logger.info(
                                f"Updated particle picking results with CryoSPARC import job: {import_job_uid}"
                            )
                except Exception as update_exc:
                    self.logger.warning(
                        f"Could not update particle picking results with import job info: {update_exc}"
                    )

            except Exception as e:
                self.logger.error(f"Failed to convert Relion particle picking output: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                converted_outputs["conversion_status"] = "failed"
                converted_outputs["error"] = str(e)

        elif stage_name == "reconstruction":
            # For reconstruction, volumes are typically in MRC format which both can use
            self.logger.info("Reconstruction stage conversion not yet implemented")
        
        return converted_outputs
    
    def perform_transition(
        self,
        current_stage: str,
        next_stage: str,
        current_stage_outputs: Dict[str, Any],
        project_uid: Optional[str] = None,
        workspace_uid: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform the transition between stages if needed.
        
        Args:
            current_stage: Name of the current stage
            next_stage: Name of the next stage
            current_stage_outputs: Outputs from the current stage
            project_uid: CryoSparc project UID (if needed)
            workspace_uid: CryoSparc workspace UID (if needed)
            
        Returns:
            Dictionary with converted outputs ready for the next stage
        """
        needs_transition, current_backend, next_backend = self.check_transition_needed(
            current_stage, next_stage
        )
        
        if not needs_transition:
            self.logger.info(f"No transition needed between {current_stage} and {next_stage}")
            return current_stage_outputs
        
        # Check if transition has already been done
        self.logger.info(f"🔍 [perform_transition] Checking for existing transition: {current_stage} ({current_backend.value if current_backend else 'unknown'}) -> {next_stage} ({next_backend.value if next_backend else 'unknown'})")
        existing_transition = self.check_existing_transition(
            current_stage, next_stage, current_stage_outputs
        )
        
        if existing_transition:
            self.logger.info(f"✅ [perform_transition] Found existing transition, using cached results")
            # Extract converted outputs from existing transition
            if current_backend == BackendType.CRYOSPARC and next_backend == BackendType.RELION:
                relion_job = existing_transition.get("relion_job", {})
                converted_outputs = {
                    "selected_micrographs_star": relion_job.get("star_file_absolute"),
                    "selection_job_dir": relion_job.get("job_dir"),
                    "transition_config": str(Path(relion_job.get("job_dir_absolute", "")) / "transition_config.json"),
                }
            elif current_backend == BackendType.RELION and next_backend == BackendType.CRYOSPARC:
                cryosparc_jobs = existing_transition.get("cryosparc_jobs", {})
                preprocessing_outputs = existing_transition.get("preprocessing_outputs", {})
                converted_outputs = {
                    "micrograph_selection_job_uid": preprocessing_outputs.get("micrograph_selection_job_uid") or cryosparc_jobs.get("selection_job_uid"),
                    "import_job_uid": cryosparc_jobs.get("import_job_uid"),
                    "ctf_job_uid": cryosparc_jobs.get("ctf_job_uid"),
                    "project_uid": preprocessing_outputs.get("project_uid"),
                    "workspace_uid": preprocessing_outputs.get("workspace_uid"),
                }
            else:
                converted_outputs = {}
            
            # Merge with original outputs
            result = {**current_stage_outputs, **converted_outputs}
            result["transition_info"] = existing_transition.get("transition_info", {})
            return result
        
        self.logger.info(f"❌ [perform_transition] No existing transition found, will perform new conversion")
        
        if current_backend == BackendType.CRYOSPARC and next_backend == BackendType.RELION:
            self.logger.info(f"Converting CryoSparc -> Relion for {current_stage} -> {next_stage}")
            conversion_result = self.convert_cryosparc_to_relion(
                current_stage,
                current_stage_outputs
            )
            
            if not conversion_result.get("success"):
                self.logger.error(f"Conversion failed: {conversion_result.get('error')}")
                return current_stage_outputs
            
            # Merge converted outputs with original (converted takes precedence)
            converted_outputs = conversion_result.get("converted_outputs", {})
            result = {**current_stage_outputs, **converted_outputs}
            result["transition_info"] = {
                "from_agent": "cryosparc",
                "to_agent": "relion",
                "stage": current_stage,
                "converted": list(converted_outputs.keys()),
                "relion_job_dir": conversion_result.get("relion_job_dir"),
                "config_file": conversion_result.get("config_file"),
                "verification": conversion_result.get("verification")
            }
            return result
        
        elif current_backend == BackendType.RELION and next_backend == BackendType.CRYOSPARC:
            self.logger.info(f"Converting Relion -> CryoSparc for {current_stage} -> {next_stage}")
            
            # Get project/workspace from config if not provided
            if not project_uid:
                project_uid = self.master_config.get("workflow", {}).get("project_uid", "P1")
            if not workspace_uid:
                workspace_uid = self.master_config.get("workflow", {}).get("workspace_uid", "W1")
            
            converted = self.convert_relion_to_cryosparc(
                current_stage,
                current_stage_outputs,
                project_uid,
                workspace_uid
            )
            # Merge converted outputs with original (converted takes precedence)
            result = {**current_stage_outputs, **converted}
            result["transition_info"] = {
                "from_agent": "relion",
                "to_agent": "cryosparc",
                "stage": current_stage,
                "converted": list(converted.keys())
            }
            return result
        
        else:
            self.logger.warning(f"Unknown transition: {current_backend} -> {next_backend}")
            return current_stage_outputs
    
    def get_transition_info(self, stage_outputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get transition information from stage outputs if transition was performed.
        
        Args:
            stage_outputs: Stage outputs dictionary
            
        Returns:
            Transition info dictionary or None
        """
        return stage_outputs.get("transition_info")
