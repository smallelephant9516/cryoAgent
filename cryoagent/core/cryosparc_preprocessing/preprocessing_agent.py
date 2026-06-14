"""ReAct-based preprocessing agent for CryoEM data processing."""

import json
import logging
import re
from typing import Dict, Any, List
from langchain.tools import Tool
from langchain_core.language_models import BaseLanguageModel
from typing import Optional
from pathlib import Path

from ..base_react_agent import BaseReActAgent
from .preprocessing_tools import PreprocessingTools
from ...tools.cryosparc_tools import CryoSPARCTools
from ...tools.cryosparc_parser_tools import CryoSPARCPreprocessingParser, WorkflowContext
from ...config.config_loader import CryoAgentConfig
from ...prompts.prompt_loader import load_prompt


class PreprocessingAgent(BaseReActAgent):
    """ReAct-based agent for CryoEM preprocessing operations."""
    
    def __init__(
        self,
        cryosparc_tools: CryoSPARCTools,
        config: CryoAgentConfig,
        llm: Optional[BaseLanguageModel] = None
    ):
        """
        Initialize the preprocessing agent.
        
        Args:
            cryosparc_tools: CryoSPARC tools instance
            config: Complete configuration object
            llm: Language model for the agent
        """
        super().__init__(cryosparc_tools, config, llm)
        # Initialize logger for this agent
        self.logger = logging.getLogger("PreprocessingAgent")
        # Load preprocessing workflow configuration (stage config)
        self.preprocessing_config = self._load_preprocessing_config()
        self.stage_config = self.preprocessing_config
        self.stage_workflow = self.stage_config.get("workflow", {})
        # Extract microscope-related defaults from stage config and apply global overrides when requested
        stage_defaults: Dict[str, Any] = {}
        stage_defaults.update(self.stage_config.get("microscope_parameters", {}) or {})
        import_defaults = (
            self.preprocessing_config
            .get("workflow", {})
            .get("import_movies", {})
        )
        if isinstance(import_defaults, dict):
            stage_defaults.update(import_defaults)
        self.microscope_config = self._resolve_microscope_defaults(stage_defaults, update_cache=True)
        # Track import method for determining group_job_uid in CTF estimation
        self.import_micrographs_job_uid: Optional[str] = None
        self.import_movies_job_uids: List[str] = []
    
    def _load_preprocessing_config(self) -> Dict[str, Any]:
        """Load preprocessing workflow configuration from separate config file."""
        try:
            # Default path for preprocessing config
            preprocessing_config_path = 'configs/cryosparc/preprocessing_config.json'
            
            # If it's a relative path, make it relative to the project root
            if not Path(preprocessing_config_path).is_absolute():
                preprocessing_config_path = Path.cwd() / preprocessing_config_path
            
            config_path = Path(preprocessing_config_path)
            
            if not config_path.exists():
                print(f"Warning: Preprocessing configuration file not found: {config_path}")
                # Return default values if loading fails
                return {
                    "motion_correction": {
                        "binning": 1,
                        "patch_size": 5
                    },
                    "ctf_estimation": {
                        "min_res": 30.0,
                        "max_res": 4.0
                    }
                }
            
            with open(config_path, 'r') as f:
                preprocessing_data = json.load(f)
            
            # Return the full preprocessing configuration
            return preprocessing_data
            
        except Exception as e:
            print(f"Warning: Could not load preprocessing configuration: {e}")
            # Return default values if loading fails
            return {
                "motion_correction": {
                    "binning": 1,
                    "patch_size": 5
                },
                "ctf_estimation": {
                    "min_res": 30.0,
                    "max_res": 4.0
                }
            }
    
    def _parse_boolean_param(self, value: Any, default: bool = False) -> bool:
        """Parse boolean parameter that might be string, number, or boolean."""
        if value is None or value == "":
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {'true', '1', 'yes', 'on'}:
                return True
            if lowered in {'false', '0', 'no', 'off'}:
                return False
        return bool(value)

    def _parse_int_param(self, value: Any, default: int = 0, param_name: str = "value") -> int:
        """Parse integer parameter that might be string/number/bool."""
        if value is None or value == "":
            return default
        try:
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            if isinstance(value, str):
                stripped = value.strip()
                if stripped == "":
                    return default
                lowered = stripped.lower()
                if lowered in {'true', 'yes', 'on'}:
                    return 1
                if lowered in {'false', 'no', 'off'}:
                    return 0
                return int(float(stripped))
        except (TypeError, ValueError):
            self.logger.warning("Invalid %s value '%s'; defaulting to %s", param_name, value, default)
        return default

    def _parse_float_param(self, value: Any, default: float = 0.0, param_name: str = "value") -> float:
        """Parse float parameter; accepts strings with trailing junk (e.g. '40.0)', '40.0 e-/Å²')."""
        if value is None or value == "":
            return default
        try:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                stripped = value.strip()
                if stripped == "":
                    return default
                # Extract leading numeric part (handles "40.0)", "40.0 e-/Å²", etc.)
                match = re.match(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", stripped)
                if match:
                    return float(match.group(0))
                return float(stripped)
        except (TypeError, ValueError):
            self.logger.warning("Invalid %s value '%s'; defaulting to %s", param_name, value, default)
        return default

    @staticmethod
    def _sanitize_uid(uid: str) -> str:
        """Strip trailing punctuation from project/workspace UID (e.g. 'W1.' -> 'W1')."""
        if not uid or not isinstance(uid, str):
            return uid
        return uid.rstrip(".).;,:!?")

    def _resolve_gain_reference_params(
        self,
        params: Dict[str, Any],
        movie_set: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Derive CryoSPARC gain-reference orientation flags for a movie set."""
        gain_rot_value = params.get("gain_rot", movie_set.get("gain_rot", 0))
        gain_rot = self._parse_int_param(gain_rot_value, default=0, param_name="gain_rot") % 4

        gain_flip_value = params.get("gain_flip", movie_set.get("gain_flip", 0))
        relion_gain_flip = self._parse_int_param(gain_flip_value, default=0, param_name="gain_flip")

        relion_flip_y = bool(relion_gain_flip & 0b01)
        relion_flip_x = bool(relion_gain_flip & 0b10)

        default_flip_y = not relion_flip_y
        default_flip_x = relion_flip_x

        return {
            "gain_rot": gain_rot,
            "relion_gain_flip": relion_gain_flip,
            "gainref_flip_y": self._parse_boolean_param(params.get("gainref_flip_y"), default=default_flip_y),
            "gainref_flip_x": self._parse_boolean_param(params.get("gainref_flip_x"), default=default_flip_x),
            "gainref_rotate_num": self._parse_int_param(
                params.get("gainref_rotate_num", gain_rot),
                default=gain_rot,
                param_name="gainref_rotate_num",
            ) % 4,
        }

    def _collect_import_movies_job_uids(self) -> List[str]:
        """Collect import job UIDs from prior tool executions in this session."""
        job_uids: List[str] = []
        seen = set()
        for entry in self.tool_execution_log:
            if entry.get("tool") != "import_movies":
                continue
            result = entry.get("result")
            if not isinstance(result, dict):
                continue
            candidates = result.get("job_uids") or []
            if not candidates and result.get("job_uid"):
                candidates = [result["job_uid"]]
            for uid in candidates:
                if uid and uid not in seen:
                    seen.add(uid)
                    job_uids.append(uid)
        return job_uids

    def _parse_job_uid_list(self, params: Dict[str, Any], input_str: str) -> List[str]:
        """Parse one or more CryoSPARC job UIDs from tool params or free text."""
        raw_values: List[str] = []
        for key in ("movies_job_uids", "movies_job_uid"):
            value = params.get(key)
            if value is None:
                continue
            if isinstance(value, list):
                raw_values.extend(str(item).strip() for item in value if str(item).strip())
            else:
                raw_values.extend(
                    part.strip()
                    for part in str(value).split(",")
                    if part.strip()
                )

        if not raw_values:
            matches = re.findall(r"\bJ\d+\b", input_str)
            raw_values.extend(matches)

        deduped: List[str] = []
        seen = set()
        for uid in raw_values:
            if uid not in seen:
                seen.add(uid)
                deduped.append(uid)
        return deduped

    def _create_tools(self) -> List[Tool]:
        """Create preprocessing-specific tools."""
        return [
            PreprocessingTools.create_import_movies_tool(self),
            PreprocessingTools.create_import_micrographs_tool(self),
            PreprocessingTools.create_motion_correction_tool(self),
            PreprocessingTools.create_ctf_estimation_tool(self),
            PreprocessingTools.create_micrograph_selection_tool(self),
            PreprocessingTools.create_get_job_status_tool(self),
            PreprocessingTools.create_wait_for_job_tool(self),
            PreprocessingTools.create_get_job_log_tool(self),
            PreprocessingTools.create_reason_about_workflow_tool(self)
        ]
    
    def _get_system_prompt_context(self) -> Dict[str, Any]:
        microscope_config = getattr(self, "microscope_config", {})
        movie_sets = self._get_movie_sets()
        movie_sets_summary = (
            "; ".join(
                f"{movie_set.get('name', f'set_{index + 1}')}: {movie_set.get('movies_path', 'N/A')}"
                for index, movie_set in enumerate(movie_sets)
            )
            if movie_sets
            else "N/A"
        )
        micrographs_path = microscope_config.get("micrographs_path")
        return {
            "project_uid": self.config.workflow.project_uid,
            "workspace_uid": self.config.workflow.workspace_uid,
            "movie_sets_summary": movie_sets_summary,
            "movies_path": microscope_config.get("movies_path", "N/A"),
            "micrographs_path_display": (
                micrographs_path if micrographs_path else "Not set (will use movies_path)"
            ),
            "gain_ref_path": microscope_config.get("gain_ref_path", "N/A"),
            "pixel_size": microscope_config.get("pixel_size", "N/A"),
            "voltage": microscope_config.get("voltage", "N/A"),
        }

    def _get_react_system_prompt(self) -> str:
        """Get the preprocessing-specific ReAct system prompt."""
        return load_prompt(
            "cryosparc/preprocessing/system.md",
            self._get_system_prompt_context(),
        )
    
    # Tool implementation methods
    def _import_movies_tool(self, input_str: str) -> str:
        """Tool wrapper for importing movies."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            project_uid = self._sanitize_uid(
                params.get("project_uid", self.config.workflow.project_uid)
            )
            workspace_uid = self._sanitize_uid(
                params.get("workspace_uid", self.config.workflow.workspace_uid)
            )

            microscope_config = getattr(self, 'microscope_config', {})
            movie_sets = self._get_movie_sets()
            if not movie_sets:
                raise ValueError(
                    "No movie sets configured. Set movies_path in microscope_config.json."
                )

            explicit_movies_path = params.get("movies_path")
            set_index = params.get("set_index")
            if set_index is not None:
                try:
                    selected_set = movie_sets[int(set_index)]
                except (TypeError, ValueError, IndexError) as exc:
                    raise ValueError(
                        f"Invalid set_index '{set_index}'. "
                        f"Expected 0..{len(movie_sets) - 1}."
                    ) from exc
                sets_to_import = [selected_set]
            elif explicit_movies_path:
                sets_to_import = [{
                    "name": params.get("set_name", "custom"),
                    "movies_path": explicit_movies_path,
                    "gain_ref_path": params.get("gain_ref_path", microscope_config.get("gain_ref_path")),
                    "gain_rot": microscope_config.get("gain_rot", 0),
                    "gain_flip": microscope_config.get("gain_flip", 0),
                    "pixel_size": microscope_config.get("pixel_size"),
                    "voltage": microscope_config.get("voltage"),
                    "cs_mm": microscope_config.get("cs_mm"),
                    "dose": microscope_config.get("dose"),
                }]
            else:
                sets_to_import = movie_sets

            wait_for_completion = self._parse_boolean_param(params.get("wait_for_completion"), default=False)
            timeout = int(params.get("timeout", self.config.job_management.default_timeout))
            check_interval = int(params.get("check_interval", self.config.job_management.status_check_interval))

            imported_jobs: List[Dict[str, Any]] = []
            job_uids: List[str] = []
            for movie_set in sets_to_import:
                gain_params = self._resolve_gain_reference_params(params, movie_set)
                tool_params = {
                    "project_uid": project_uid,
                    "workspace_uid": workspace_uid,
                    "movies_path": movie_set.get("movies_path"),
                    "gain_ref_path": movie_set.get("gain_ref_path"),
                    "pixel_size": self._parse_float_param(
                        params.get("pixel_size", movie_set.get("pixel_size", microscope_config.get("pixel_size", 0.6575))),
                        default=0.6575, param_name="pixel_size"
                    ),
                    "voltage": self._parse_float_param(
                        params.get("voltage", movie_set.get("voltage", microscope_config.get("voltage", 300.0))),
                        default=300.0, param_name="voltage"
                    ),
                    "cs_mm": self._parse_float_param(
                        params.get("cs_mm", movie_set.get("cs_mm", microscope_config.get("cs_mm", 2.7))),
                        default=2.7, param_name="cs_mm"
                    ),
                    "dose": self._parse_float_param(
                        params.get("dose", movie_set.get("dose", microscope_config.get("dose", 53.0))),
                        default=53.0, param_name="dose"
                    ),
                    "gainref_flip_x": gain_params["gainref_flip_x"],
                    "gainref_flip_y": gain_params["gainref_flip_y"],
                    "gainref_rotate_num": gain_params["gainref_rotate_num"],
                    "wait_for_completion": wait_for_completion,
                    "timeout": timeout,
                    "check_interval": check_interval,
                }

                result = self.cryosparc_tools.import_movies(**tool_params)
                job_uid = result["job_uid"]
                job_uids.append(job_uid)
                imported_jobs.append({
                    "set_name": movie_set.get("name"),
                    "job_uid": job_uid,
                    "movies_path": tool_params["movies_path"],
                    "gain_ref_path": tool_params.get("gain_ref_path"),
                })

            self.import_movies_job_uids = job_uids
            used_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "movie_sets": imported_jobs,
                "wait_for_completion": wait_for_completion,
                "timeout": timeout,
                "check_interval": check_interval,
            }
            combined_result = {
                "job_uid": job_uids[0],
                "job_uids": job_uids,
                "job_type": "import_movies",
                "imported_sets": imported_jobs,
            }
            self._record_tool_execution("import_movies", used_params, result=combined_result)

            if len(job_uids) == 1:
                return f"✅ Successfully queued import movies job: {job_uids[0]}"
            return (
                "✅ Successfully queued import movies jobs for "
                f"{len(job_uids)} sets: {', '.join(job_uids)}"
            )
            
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("import_movies", context, error=str(e))
            return f"❌ Error importing movies: {str(e)}"
    
    def _import_micrographs_tool(self, input_str: str) -> str:
        """Tool wrapper for importing micrographs directly."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            project_uid = self._sanitize_uid(
                params.get("project_uid", self.config.workflow.project_uid)
            )
            workspace_uid = self._sanitize_uid(
                params.get("workspace_uid", self.config.workflow.workspace_uid)
            )
            
            # Safely get microscope config values, handling case where it might not be set yet
            microscope_config = getattr(self, 'microscope_config', {})
            
            # Use micrographs_path if provided, otherwise fall back to movies_path
            micrographs_path = params.get("micrographs_path", 
                                         microscope_config.get("micrographs_path") or 
                                         microscope_config.get("movies_path", "/path/to/micrographs/*.mrc"))
            
            tool_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "micrographs_path": micrographs_path,
                "pixel_size": float(params.get("pixel_size", microscope_config.get("pixel_size", 0.6575))),
                "voltage": float(params.get("voltage", microscope_config.get("voltage", 300.0))),
                "cs_mm": float(params.get("cs_mm", microscope_config.get("cs_mm", 2.7))),
                "dose": float(params.get("dose", microscope_config.get("dose", 53.0))),
                "wait_for_completion": self._parse_boolean_param(params.get("wait_for_completion"), default=False),
                "timeout": int(params.get("timeout", self.config.job_management.default_timeout)),
                "check_interval": int(params.get("check_interval", self.config.job_management.status_check_interval))
            }

            result = self.cryosparc_tools.import_micrographs(**tool_params)
            used_params = dict(tool_params)
            # Track the import_micrographs job UID for use in CTF estimation
            if result.get('job_uid'):
                self.import_micrographs_job_uid = result['job_uid']
            self._record_tool_execution("import_micrographs", used_params, result=result)
            return f"✅ Successfully queued import micrographs job: {result['job_uid']}. Note: Motion correction is NOT needed - proceed directly to CTF estimation."
            
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("import_micrographs", context, error=str(e))
            return f"❌ Error importing micrographs: {str(e)}"
    
    def _motion_correction_tool(self, input_str: str) -> str:
        """Tool wrapper for motion correction."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            project_uid = self._sanitize_uid(
                params.get("project_uid", self.config.workflow.project_uid)
            )
            workspace_uid = self._sanitize_uid(
                params.get("workspace_uid", self.config.workflow.workspace_uid)
            )

            movies_job_uids = self._parse_job_uid_list(params, input_str)
            if not movies_job_uids:
                movies_job_uids = self._collect_import_movies_job_uids()
            if not movies_job_uids and self.import_movies_job_uids:
                movies_job_uids = list(self.import_movies_job_uids)

            if not movies_job_uids:
                return (
                    "❌ Error starting motion correction: Missing required parameter "
                    "'movies_job_uid' or 'movies_job_uids'. Please specify the job UID(s) "
                    "from the import_movies step (e.g., movies_job_uids=J16,J17)."
                )

            preprocessing_config = getattr(self, 'preprocessing_config', {}).get('workflow', {})
            motion_correction_config = preprocessing_config.get('motion_correction', {})

            used_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "movies_job_uids": movies_job_uids,
                "binning": self._parse_int_param(params.get("binning", motion_correction_config.get("binning", 1)), default=1, param_name="binning"),
                "patch_size": self._parse_int_param(params.get("patch_size", motion_correction_config.get("patch_size", 5)), default=5, param_name="patch_size"),
                "wait_for_completion": self._parse_boolean_param(params.get("wait_for_completion"), default=False),
                "timeout": self._parse_int_param(params.get("timeout", self.config.job_management.default_timeout), default=self.config.job_management.default_timeout, param_name="timeout"),
                "check_interval": self._parse_int_param(params.get("check_interval", self.config.job_management.status_check_interval), default=self.config.job_management.status_check_interval, param_name="check_interval")
            }

            result = self.cryosparc_tools.motion_correction(**used_params)
            self._record_tool_execution("motion_correction", used_params, result=result)
            if len(movies_job_uids) == 1:
                return f"✅ Successfully queued motion correction job: {result['job_uid']} (input: {movies_job_uids[0]})"
            return (
                f"✅ Successfully queued motion correction job: {result['job_uid']} "
                f"(inputs: {', '.join(movies_job_uids)})"
            )
            
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("motion_correction", context, error=str(e))
            return f"❌ Error starting motion correction: {str(e)}"
    
    def _ctf_estimation_tool(self, input_str: str) -> str:
        """Tool wrapper for CTF estimation."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            project_uid = self._sanitize_uid(
                params.get("project_uid", self.config.workflow.project_uid)
            )
            workspace_uid = self._sanitize_uid(
                params.get("workspace_uid", self.config.workflow.workspace_uid)
            )

            # Resolve micrographs_job_uid: params first, then extract from text (e.g. "from job J4"), then last motion_correction result
            micrographs_job_uid = params.get("micrographs_job_uid")
            if not micrographs_job_uid:
                job_uid_pattern = r'\bJ\d+\b'
                matches = re.findall(job_uid_pattern, input_str)
                if matches:
                    micrographs_job_uid = matches[0]
                    self.logger.info(f"Extracted micrographs_job_uid '{micrographs_job_uid}' from input text")
            if not micrographs_job_uid:
                for entry in reversed(self.tool_execution_log):
                    if entry.get("tool") == "motion_correction" and "result" in entry:
                        result = entry["result"]
                        if isinstance(result, dict) and result.get("job_uid"):
                            micrographs_job_uid = result["job_uid"]
                            self.logger.info(f"Using micrographs_job_uid '{micrographs_job_uid}' from last motion_correction result")
                            break
            if not micrographs_job_uid and self.import_micrographs_job_uid:
                micrographs_job_uid = self.import_micrographs_job_uid
                self.logger.info(f"Using micrographs_job_uid '{micrographs_job_uid}' from import_micrographs")
            if not micrographs_job_uid:
                return (
                    "❌ Error starting CTF estimation: Missing required parameter 'micrographs_job_uid'. "
                    "Please specify the job UID from the motion correction step (e.g., micrographs_job_uid=J4 or 'from job J4')."
                )

            # Safely get preprocessing config values, handling case where it might not be set yet
            preprocessing_config = getattr(self, 'preprocessing_config', {}).get('workflow', {})
            ctf_config = preprocessing_config.get('ctf_estimation', {})
            
            # Determine group_job_uid:
            # 1. If specified in params, use it
            # 2. If not specified and previous job was import_micrographs, use import_micrographs job UID
            # 3. Otherwise, default to "micrographs"
            group_job_uid = params.get("group_job_uid")
            if group_job_uid is None:
                if (self.import_micrographs_job_uid and
                    micrographs_job_uid == self.import_micrographs_job_uid):
                    # Previous job was import_micrographs, use its job UID as group_job_uid
                    group_job_uid = "imported_micrographs"
                else:
                    # Default to "micrographs"
                    group_job_uid = "micrographs"
            
            used_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "micrographs_job_uid": micrographs_job_uid,
                "group_job_uid": group_job_uid,
                "min_res": float(params.get("min_res", ctf_config.get("min_res", 30.0))),
                "max_res": float(params.get("max_res", ctf_config.get("max_res", 4.0))),
                "wait_for_completion": self._parse_boolean_param(params.get("wait_for_completion"), default=False),
                "timeout": self._parse_int_param(params.get("timeout", self.config.job_management.default_timeout), default=self.config.job_management.default_timeout, param_name="timeout"),
                "check_interval": self._parse_int_param(params.get("check_interval", self.config.job_management.status_check_interval), default=self.config.job_management.status_check_interval, param_name="check_interval")
            }

            result = self.cryosparc_tools.ctf_estimation(**used_params)
            self._record_tool_execution("ctf_estimation", used_params, result=result)
            return f"✅ Successfully queued CTF estimation job: {result['job_uid']}"
            
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("ctf_estimation", context, error=str(e))
            return f"❌ Error starting CTF estimation: {str(e)}"
    
    def _micrograph_selection_tool(self, input_str: str) -> str:
        """Tool wrapper for micrograph selection."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            project_uid = self._sanitize_uid(
                params.get("project_uid", self.config.workflow.project_uid)
            )
            workspace_uid = self._sanitize_uid(
                params.get("workspace_uid", self.config.workflow.workspace_uid)
            )

            # Safely get preprocessing config values, handling case where it might not be set yet
            preprocessing_config = getattr(self, 'preprocessing_config', {}).get('workflow', {})
            micrograph_selection_config = preprocessing_config.get('micrograph_selection', {})
            
            used_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "ctf_job_uid": params.get("ctf_job_uid"),
                "min_resolution": float(params.get("min_resolution", micrograph_selection_config.get("min_resolution", 5.0))),
                "wait_for_completion": self._parse_boolean_param(params.get("wait_for_completion"), default=False),
                "timeout": self._parse_int_param(params.get("timeout", self.config.job_management.default_timeout), default=self.config.job_management.default_timeout, param_name="timeout"),
                "check_interval": self._parse_int_param(params.get("check_interval", self.config.job_management.status_check_interval), default=self.config.job_management.status_check_interval, param_name="check_interval")
            }

            result = self.cryosparc_tools.micrograph_selection(**used_params)
            self._record_tool_execution("micrograph_selection", used_params, result=result)
            return f"✅ Successfully queued micrograph selection job: {result['job_uid']}"
            
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("micrograph_selection", context, error=str(e))
            return f"❌ Error starting micrograph selection: {str(e)}"
    
    def _reason_about_workflow_tool(self, input_str: str) -> str:
        """Tool for reasoning about preprocessing workflow state."""
        try:
            reasoning = f"""
🤔 **Preprocessing Workflow Analysis**:

**Current State**: {input_str}

**Workflow Dependencies**:
**Path 1 (Movies)**: Import Movies → Motion Correction → CTF Estimation → Micrograph Selection
**Path 2 (Micrographs)**: Import Micrographs → CTF Estimation → Micrograph Selection (SKIP motion correction)

**Next Steps Analysis**:
- If no jobs are running: 
  - Choose import_movies if you have raw movie files
  - Choose import_micrographs if you have already motion-corrected micrographs
- If import_movies job is running: Wait for completion, then start motion_correction
- If import_micrographs job is running: Wait for completion, then start ctf_estimation (SKIP motion correction)
- If motion correction is running: Wait for completion, then start ctf_estimation
- If CTF estimation is running: Wait for completion, then start micrograph_selection
- If micrograph selection is running: Wait for completion, preprocessing is done

**Recommended Actions**:
- Always check job status before proceeding
- Use wait_for_job for critical dependencies
- Verify each step completes successfully before moving to the next
- If you used import_micrographs, DO NOT run motion_correction
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
            results: List of preprocessing workflow results
            context: Workflow context with project/workspace info
            
        Returns:
            Dictionary of stage outputs
        """
        parser = CryoSPARCPreprocessingParser(self.cryosparc_tools, self.logger)
        return parser.process_workflow_results(results, context)
    
    def validate_results(self, stage_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that the preprocessing workflow completed successfully.
        
        Args:
            stage_outputs: Dictionary of stage outputs to validate
            
        Returns:
            Dictionary with 'success' boolean and 'error' message if failed
        """
        parser = CryoSPARCPreprocessingParser(self.cryosparc_tools, self.logger)
        return parser.validate_results(stage_outputs)
    
    def save_results(self, stage_outputs: Dict[str, Any], context: WorkflowContext, success: bool = True) -> str:
        """
        Save preprocessing results to a JSON file.
        
        Args:
            stage_outputs: Dictionary of stage outputs
            context: Workflow context
            success: Whether preprocessing was successful
            
        Returns:
            Path to the saved JSON file
        """
        parser = CryoSPARCPreprocessingParser(self.cryosparc_tools, self.logger)
        return parser.save_results(stage_outputs, context, success)

