"""Base ReAct (Reasoning + Acting) agent with common logic for all CryoEM agents."""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Sequence
from abc import ABC, abstractmethod
# Import AgentExecutor and create_tool_calling_agent
# Note: langchain 1.0+ removed AgentExecutor - this code requires langchain < 1.0.0
try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    # Fallback for different import paths in langchain 0.x
    try:
        from langchain.agents.agent import AgentExecutor
        from langchain.agents import create_tool_calling_agent
    
    except ImportError as e:
        raise ImportError(
            f"Failed to import AgentExecutor from langchain.agents. "
            f"This code requires langchain >= 0.3.0, < 1.0.0. "
            f"Please install a compatible version: pip install 'langchain>=0.3.0,<0.4.0'. "
            f"Original error: {e}"
        ) from e

from langchain.tools import Tool
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from ..tools.cryosparc_tools import CryoSPARCTools
from ..tools.cryosparc_forum_tools import (
    extract_search_queries_from_log,
    format_forum_search_response,
    search_cryosparc_forum,
    search_cryosparc_forum_multi,
)
from ..config.config_loader import CryoAgentConfig, resolve_movie_sets
from .llm_factory import LLMFactory
from ..utils.conversation_logger import ConversationLogger
from ..utils.realtime_conversation_logger import RealtimeConversationLogger
from ..utils.general_llm_logger import GeneralLLMLogger
from ..utils.log_resume import LogResumeParser
from ..prompts.prompt_loader import load_prompt


class BaseReActAgent(ABC):
    """Base class for ReAct-based CryoEM agents with common framework logic."""
    
    _ALLOWED_BOX_SIZES: Sequence[int] = (
        16, 20, 24, 28, 32, 36, 40, 48, 56, 60, 64, 72, 80, 84, 96, 100, 108, 112,
        120, 128, 140, 144, 160, 168, 180, 192, 196, 200, 216, 224, 240, 252, 256,
        280, 288, 300, 320, 324, 336, 360, 384, 392, 400, 420, 432, 448, 480, 500,
        504, 512, 540, 560, 576, 588, 600, 640, 648, 672, 700, 720, 756, 768, 784,
        800, 840, 864, 896, 900, 960, 972, 980, 1000, 1008, 1024, 1080, 1120, 1152,
        1176, 1200, 1260, 1280, 1296, 1344, 1372, 1400, 1440, 1500, 1512, 1536, 1568,
        1600, 1620, 1680, 1728, 1764, 1792, 1800, 1920, 1944, 1960, 2000,
    )
    
    def __init__(
        self,
        cryosparc_tools: CryoSPARCTools,
        config: CryoAgentConfig,
        llm: Optional[BaseLanguageModel] = None
    ):
        """
        Initialize the base ReAct agent.
        
        Args:
            cryosparc_tools: CryoSPARC tools instance
            config: Complete configuration object
            llm: Language model for the agent
        """
        self.cryosparc_tools = cryosparc_tools
        self.config = config
        
        # Auto-select provider if current one doesn't have valid API key
        self._ensure_valid_provider()
        
        self.llm = llm or self._create_default_llm()
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent_executor()
        self.reasoning_history: List[Dict[str, str]] = []
        self.tool_execution_log: List[Dict[str, Any]] = []
        self._microscope_config_cache: Optional[Dict[str, Any]] = None
        self._microscope_override_cache: Optional[Dict[str, Any]] = None
        self._microscope_override_enabled: Optional[bool] = None
        self._microscope_override_path: Optional[Path] = None
        
        # Memory control state
        self.conversation_count = 0
        self.last_conversation_id = None
        self.conversation_memory: List[Dict[str, Any]] = []
        
        # Conversation logging
        self.conversation_logger = ConversationLogger()
        self.realtime_logger = RealtimeConversationLogger()
        self.general_llm_logger = None  # Will be set by the workflow
        self.enable_conversation_logging = True
    
    def _ensure_valid_provider(self):
        """Ensure that the current provider has a valid API key, auto-select if not."""
        try:
            current_model_config = self.config.agent.get_current_model_config()
            if not self.config.agent._is_api_key_valid(current_model_config.api_key):
                selected_provider = self.config.agent.auto_select_provider()
                if self.config.agent.verbose:
                    print(f"🔄 Auto-selected provider: {selected_provider} (no valid API key for configured provider)")
        except ValueError as e:
            if self.config.agent.verbose:
                print(f"⚠️ Warning: {e}")
            raise
    
    def _create_default_llm(self) -> BaseLanguageModel:
        """Create default language model using the configured provider."""
        model_config = self.config.agent.get_current_model_config()
        return LLMFactory.create_llm(model_config, self.config.agent.provider)
    
    @abstractmethod
    def _create_tools(self) -> List[Tool]:
        """
        Create agent-specific tools. Must be implemented by subclasses.
        
        Returns:
            List of LangChain Tool objects
        """
        pass
    
    @abstractmethod
    def _get_react_system_prompt(self) -> str:
        """
        Get agent-specific ReAct system prompt. Must be implemented by subclasses.
        
        Returns:
            System prompt string
        """
        pass

    def _compose_stage_system_prompt(
        self,
        relative_path: str,
        variables: Optional[Dict[str, Any]] = None,
        *,
        include_failure_recovery: bool = True,
    ) -> str:
        """
        Load a stage system prompt and append shared forum-based failure recovery rules.

        Args:
            relative_path: Path under cryoagent/prompts/ (e.g. cryosparc/preprocessing/system.md).
            variables: Template variables for the stage prompt.
            include_failure_recovery: When True, append shared/failure-recovery-forum.md.
        """
        stage_prompt = load_prompt(relative_path, variables or {})
        if not include_failure_recovery:
            return stage_prompt
        recovery_prompt = load_prompt("shared/failure-recovery-forum.md")
        return f"{stage_prompt.rstrip()}\n\n{recovery_prompt.strip()}\n"

    def _get_microscope_parameter(self, key: str, default: Any = None) -> Any:
        """
        Fetch a single microscope parameter from stage config's microscope_parameters section.

        Args:
            key: Parameter name (e.g., 'pixel_size').
            default: Value returned when parameter is unavailable.
        """
        # Use cached microscope config from stage config (set by subclasses)
        if self._microscope_config_cache is not None:
            return self._microscope_config_cache.get(key, default)

        # If cache not populated yet, try to resolve from the stage configuration,
        # applying microscope_config.json overrides when requested.
        stage_config = getattr(self, "stage_config", None)
        if isinstance(stage_config, dict):
            stage_params = stage_config.get("microscope_parameters") or {}
            if stage_params:
                resolved = self._resolve_microscope_defaults(stage_params, update_cache=True)
                return resolved.get(key, default)
        
        return default

    def _ensure_microscope_overrides_loaded(self) -> None:
        """
        Lazily load microscope overrides from configs/microscope_config.json when available.
        """
        if self._microscope_override_enabled is not None:
            return

        config_path = Path(self.config.workflow.microscope_config_path)
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        self._microscope_override_path = config_path

        try:
            with open(config_path, "r", encoding="utf-8") as fp:
                microscope_data = json.load(fp) or {}
        except (FileNotFoundError, json.JSONDecodeError):
            self._microscope_override_enabled = False
            self._microscope_override_cache = {}
            return

        overwrite_flag = microscope_data.get("overwrite", False)
        self._microscope_override_enabled = bool(overwrite_flag)

        overrides = microscope_data.get("microscope_parameters")
        if not isinstance(overrides, dict):
            overrides = {}

        # Only cache overrides when overwrite flag is true
        self._microscope_override_cache = overrides if self._microscope_override_enabled else {}

    def _microscope_overrides_active(self) -> bool:
        """
        Determine whether microscope_config.json overrides should be applied.
        """
        self._ensure_microscope_overrides_loaded()
        return bool(self._microscope_override_enabled and self._microscope_override_cache)

    def _get_microscope_overrides(self) -> Dict[str, Any]:
        """
        Retrieve microscope override parameters when overwrite is enabled.
        """
        self._ensure_microscope_overrides_loaded()
        return dict(self._microscope_override_cache or {})

    def _resolve_microscope_defaults(
        self,
        stage_defaults: Optional[Dict[str, Any]] = None,
        *,
        update_cache: bool = False
    ) -> Dict[str, Any]:
        """
        Merge stage configuration microscope defaults with overrides from microscope_config.json.

        Args:
            stage_defaults: Baseline microscope parameters sourced from the stage config.
            update_cache: When True, update the shared microscope configuration cache so
                helper methods (e.g. _get_microscope_parameter) use the merged values.

        Returns:
            Dictionary containing the effective microscope parameters.
        """
        effective: Dict[str, Any] = {}
        if isinstance(stage_defaults, dict):
            effective.update(stage_defaults)

        overrides = self._get_microscope_overrides()
        if overrides:
            for key, value in overrides.items():
                if value is not None:
                    effective[key] = value

        if update_cache:
            self._microscope_config_cache = dict(effective)

        return effective

    def _get_movie_sets(self) -> List[Dict[str, Any]]:
        """Return resolved movie import sets from the effective microscope config."""
        microscope_config = getattr(self, "microscope_config", None) or {}
        return resolve_movie_sets(microscope_config)

    def _get_base_particle_diameter(self) -> Optional[float]:
        """
        Resolve the nominal particle diameter for the specimen.

        Preference order:
        1. microscope_config particle_diameter
        2. workflow-level particle_diameter (legacy support)
        """
        raw_value = self._get_microscope_parameter("particle_diameter")
        if raw_value is None:
            raw_value = getattr(self.config.workflow, "particle_diameter", None)

        try:
            if raw_value is None:
                return None
            diameter = float(raw_value)
            if diameter <= 0:
                return None
            return diameter
        except (TypeError, ValueError):
            return None

    def _get_scaled_particle_diameter(self, scale: float) -> Optional[float]:
        """
        Resolve a particle diameter scaled from microscope settings.

        Args:
            scale: Multiplier applied when microscope particle diameter is available.

        Returns:
            Scaled particle diameter if microscope configuration specifies a value.
            Returns None when unavailable so callers can fall back to config defaults.
        """
        raw_value = self._get_microscope_parameter("particle_diameter")
        if raw_value is None:
            return None
        try:
            diameter = float(raw_value)
            if diameter <= 0:
                return None
            return diameter * float(scale)
        except (TypeError, ValueError):
            return None

    def _normalize_box_size(self, value: Optional[float]) -> Optional[int]:
        """
        Map a computed box size to the closest allowed value.

        Args:
            value: Raw box size (pixels). If None, returns None.

        Returns:
            Closest allowed box size or None when input invalid.
        """
        if value is None:
            return None
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None
        if numeric_value <= 0:
            return None

        allowed = self._ALLOWED_BOX_SIZES
        closest = min(
            allowed,
            key=lambda candidate: (abs(candidate - numeric_value), candidate)
        )
        return int(closest)
    
    def _max_iterations(self) -> int:
        """ReAct iteration cap for this agent. Subclasses may raise it for
        multi-job loops (e.g. the improvement agent). Defaults to config."""
        return self.config.agent.max_iterations

    def _create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor with ReAct-style prompt."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self._get_react_system_prompt()),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.config.agent.verbose,
            max_iterations=self._max_iterations(),
            handle_parsing_errors=True
        )
    
    def _parse_bool_param(self, value: Any, default: bool = False) -> bool:
        """
        Robustly parse a boolean tool parameter.

        Tolerates the LLM passing a real JSON boolean (True/False), a string
        ("true"/"false"/"1"/"0"/"yes"/"no"/"on"/"off"), a number, or None.
        This avoids the fragile ``params.get(k, "true").lower()`` pattern which
        crashes with ``'bool' object has no attribute 'lower'`` when the value is
        already a Python bool.
        """
        if value is None or value == "":
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        return bool(value)

    def _parse_tool_input(self, input_str: str) -> Dict[str, Any]:
        """
        Parse tool input string into parameters.

        Supports multiple formats:
        - JSON format: {"key": "value"}
        - Job UID only: J123
        - Comma-separated: key=value,key2=value2
        
        Args:
            input_str: Input string to parse
            
        Returns:
            Dictionary of parsed parameters
        """
        import json

        input_str = input_str.strip()
        
        # Case 1: JSON format
        if input_str.startswith("{") and input_str.endswith("}"):
            try:
                return json.loads(input_str)
            except json.JSONDecodeError:
                pass
        
        # Case 2: Just a job UID (common case for single parameter tools)
        if input_str.startswith("J") and len(input_str) <= 10:
            return {"job_uid": input_str}
        
        # Case 3: Key/value pairs, accepting both comma and whitespace separators
        params = {}
        normalized = input_str.replace("\n", " ").strip()
        if normalized:
            import shlex
            segments = [seg for seg in (s.strip() for s in normalized.split(",")) if seg]
            tokens: list[str] = []
            for segment in segments:
                try:
                    tokens.extend(shlex.split(segment))
                except ValueError:
                    # Fallback to simple split on whitespace if shlex fails (e.g., unbalanced quotes)
                    tokens.extend(segment.split())
            for token in tokens:
                if "=" in token:
                    key, value = token.split("=", 1)
                    params[key.strip()] = value.strip()
        
        # Case 4: If no parameters found and input looks like a single value
        if not params and input_str:
            if input_str.startswith("J") or input_str.isdigit():
                params["job_uid"] = input_str
            else:
                params["input"] = input_str

        return params
    
    # Reserved keys that are handled explicitly by tool wrappers and must NOT
    # be forwarded as raw CryoSPARC job parameters via the passthrough dict.
    _RESERVED_TOOL_KEYS = frozenset({
        "project_uid", "workspace_uid", "job_uid", "input", "raw_input",
        "wait_for_completion", "timeout", "check_interval", "max_results",
        "query", "search_query", "params",
    })

    def _extract_passthrough_params(
        self,
        params: Dict[str, Any],
        *,
        consumed_keys: Optional[Sequence[str]] = None,
        allow_loose: bool = False,
    ) -> Dict[str, Any]:
        """
        Build the raw-CryoSPARC ``params`` dict to forward to the API layer.

        The unambiguous, default form is an explicit nested ``params`` object in
        the tool input (e.g. {"params": {"abinit_init_res": 30}}). Optionally,
        with ``allow_loose=True``, any leftover top-level keys the wrapper did
        not explicitly consume are also treated as raw CryoSPARC parameters;
        this is opt-in because wrappers may reuse the parsed-input dict for
        internal bookkeeping, which would otherwise leak non-parameter keys.

        Args:
            params: The parsed tool input.
            consumed_keys: Named arguments the wrapper already handles itself
                (only relevant when allow_loose=True).
            allow_loose: When True, also harvest unconsumed top-level keys.

        Returns:
            A dict of raw CryoSPARC parameter keys (empty if none supplied).
        """
        passthrough: Dict[str, Any] = {}

        # 1) Explicit nested params object (preferred, unambiguous form).
        explicit = params.get("params")
        if isinstance(explicit, str):
            explicit_str = explicit.strip()
            if explicit_str:
                try:
                    explicit = json.loads(explicit_str)
                except json.JSONDecodeError:
                    explicit = None
        if isinstance(explicit, dict):
            passthrough.update(explicit)

        # 2) Optional convenience: leftover top-level keys not otherwise consumed.
        if allow_loose:
            consumed = set(consumed_keys or ())
            for key, value in params.items():
                if key in self._RESERVED_TOOL_KEYS or key in consumed:
                    continue
                if key not in passthrough:
                    passthrough[key] = value

        return passthrough

    def _describe_job_params_tool(self, input_str: str) -> str:
        """Common tool wrapper exposing CryoSPARC job parameter specs to the LLM."""
        params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            job_type = params.get("job_type") or params.get("input")
            if not job_type:
                return (
                    "❌ Error: job_type parameter is required. Provide a job type "
                    "such as 'motion_correction', 'class_2d', or a raw CryoSPARC id "
                    "like 'patch_motion_correction_multi'."
                )
            include_hidden = self._parse_bool_param(params.get("include_hidden"), False)
            project_uid = params.get("project_uid", getattr(self.config.workflow, "project_uid", None))
            workspace_uid = params.get("workspace_uid", getattr(self.config.workflow, "workspace_uid", None))
            spec = self.cryosparc_tools.describe_job_params(
                job_type,
                include_hidden=include_hidden,
                project_uid=project_uid,
                workspace_uid=workspace_uid,
            )
            self._record_tool_execution("describe_job_params", {"job_type": job_type}, result=spec)

            lines = [
                f"📋 Parameters for job type '{spec['job_type']}' "
                f"({spec['param_count']} settable):"
            ]
            for key, detail in spec["params"].items():
                lines.append(
                    f"  - {key} [{detail.get('type')}] default={detail.get('default')!r}"
                    f"  — {detail.get('title')}"
                )
            lines.append(
                "\nPass any of these via the tool's 'params' dict, e.g. "
                'params={"' + (next(iter(spec["params"]), "key")) + '": <value>}.'
            )
            return "\n".join(lines)
        except ValueError as e:
            self._record_tool_execution("describe_job_params", params, error=str(e))
            return f"❌ {e}"
        except Exception as e:
            context = params or {"raw_input": input_str}
            self._record_tool_execution("describe_job_params", context, error=str(e))
            return f"❌ Error describing job params: {str(e)}"

    def _describe_job_results_tool(self, input_str: str) -> str:
        """Common tool wrapper: summarize a completed job's real result metrics."""
        params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            job_uid = params.get("job_uid") or params.get("input")
            if not job_uid:
                return "❌ Error: job_uid parameter is required."
            project_uid = params.get("project_uid", getattr(self.config.workflow, "project_uid", None))
            res = self.cryosparc_tools.describe_job_results(job_uid, project_uid=project_uid)
            self._record_tool_execution("describe_job_results", {"job_uid": job_uid}, result=res)
            if not res.get("success"):
                return f"❌ {res.get('error', 'Unknown error')}"

            lines = [f"📊 Results for {res['job_uid']} ({res.get('job_type')}, status={res.get('status')}):"]
            if res.get("resolution_angstroms") is not None:
                lines.append(f"  - resolution: {res['resolution_angstroms']:.2f} Å (lower is better)")
            if res.get("box_size") is not None:
                lines.append(f"  - box size: {res['box_size']} px")
            if res.get("symmetry"):
                lines.append(f"  - symmetry: {res['symmetry']}")
            if res.get("num_particles") is not None:
                lines.append(f"  - particles: {res['num_particles']}")
            if res.get("cfar") is not None:
                lines.append(f"  - cFAR: {res['cfar']:.3f} ({res.get('cfar_label')}) (higher is better)")
            elif res.get("cfar_note"):
                lines.append(f"  - cFAR: not computed — {res['cfar_note']}")
            if res.get("classes"):
                lines.append(f"  - {res.get('num_classes')} classes (per-class resolution / particle share / cFAR):")
                for c in res["classes"]:
                    frac = c.get("particle_fraction")
                    frac_s = f"{frac*100:.1f}%" if isinstance(frac, (int, float)) else "?"
                    rr = c.get("resolution_angstroms")
                    rr_s = f"{rr:.2f} Å" if isinstance(rr, (int, float)) else "?"
                    cf = c.get("cfar")
                    cf_s = f", cFAR {cf:.3f} ({c.get('cfar_label')})" if isinstance(cf, (int, float)) else ""
                    lines.append(
                        f"      class {c.get('class_id')}: {rr_s}, "
                        f"{c.get('num_particles')} particles ({frac_s}){cf_s}"
                    )
                if res.get("cfar_note"):
                    lines.append(f"  - note: {res['cfar_note']}")
            return "\n".join(lines)
        except Exception as e:
            context = params or {"raw_input": input_str}
            self._record_tool_execution("describe_job_results", context, error=str(e))
            return f"❌ Error describing job results: {str(e)}"

    def _get_orientation_diagnostics_tool(self, input_str: str) -> str:
        """Common tool wrapper: get/compute cFAR for a refinement via orientation diagnostics."""
        params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            refinement_job_uid = (
                params.get("refinement_job_uid") or params.get("job_uid") or params.get("input")
            )
            if not refinement_job_uid:
                return "❌ Error: refinement_job_uid parameter is required."
            project_uid = params.get("project_uid", getattr(self.config.workflow, "project_uid", None))
            workspace_uid = params.get("workspace_uid", getattr(self.config.workflow, "workspace_uid", None))
            run_if_missing = self._parse_bool_param(params.get("run_if_missing"), True)
            # Optional per-class targeting for heterogeneous refinements.
            volume_group_name = params.get("volume_group_name")
            particles_group_name = params.get("particles_group_name")
            res = self.cryosparc_tools.get_orientation_diagnostics(
                project_uid=project_uid,
                workspace_uid=workspace_uid,
                refinement_job_uid=refinement_job_uid,
                volume_group_name=volume_group_name,
                particles_group_name=particles_group_name,
                run_if_missing=run_if_missing,
            )
            self._record_tool_execution(
                "get_orientation_diagnostics",
                {"refinement_job_uid": refinement_job_uid, "volume_group_name": volume_group_name},
                result=res,
            )
            if not res.get("success"):
                return f"❌ {res.get('error', 'Unknown error')}"
            computed = " (newly computed)" if res.get("computed") else " (from existing job)"
            target = f" {res.get('volume_group_name')}" if res.get("volume_group_name") else ""
            return (
                f"🧭 cFAR for {refinement_job_uid}{target}: {res.get('cfar'):.3f} "
                f"({res.get('cfar_label')}){computed} [diagnostics job {res.get('job_uid')}]"
            )
        except Exception as e:
            context = params or {"raw_input": input_str}
            self._record_tool_execution("get_orientation_diagnostics", context, error=str(e))
            return f"❌ Error getting orientation diagnostics: {str(e)}"

    def _recent_history_context(self, max_actions: int = 8, max_chars: int = 1500) -> str:
        """A compact summary of what the agent has tried/observed so far, to ground
        the tutorial condensation in the actual problem (tool calls + results)."""
        parts: List[str] = []
        try:
            log = self.get_tool_execution_log() if hasattr(self, "get_tool_execution_log") else []
            for entry in log[-max_actions:]:
                tool = entry.get("tool")
                if not tool:
                    continue
                params = entry.get("params") or {}
                pbits = ", ".join(f"{k}={v}" for k, v in list(params.items())[:4]
                                  if k not in ("project_uid", "workspace_uid"))
                res = entry.get("result")
                if isinstance(res, dict):
                    rbits = res.get("job_uid") or res.get("resolution_angstroms") or res.get("message") or ""
                elif entry.get("error"):
                    rbits = f"ERROR {entry['error']}"
                else:
                    rbits = ""
                parts.append(f"- {tool}({pbits}) -> {str(rbits)[:120]}")
        except Exception:
            pass
        text = "\n".join(parts)
        return text[-max_chars:] if text else "(no prior actions recorded)"

    def _condense_tutorial(self, full_text: str, url: str, question: str) -> Optional[str]:
        """Read a WHOLE tutorial in a separate LLM call and distill it into
        actionable tool/parameter guidance for the current problem. Returns the
        digest, or None if condensation is unavailable/failed (caller falls back).
        The full text is NOT returned to the main agent — only this digest is."""
        llm = getattr(self, "llm", None)
        if llm is None or not full_text:
            return None
        system = (
            "You are condensing an official CryoSPARC tutorial/case-study for a "
            "cryo-EM processing agent that is trying to improve a 3D reconstruction. "
            "Read the FULL tutorial text and produce a TIGHT, actionable digest "
            "(~150-250 words, bullet style) focused on the agent's current problem. "
            "Cover: (1) which CryoSPARC job(s)/tool(s) the tutorial uses for this "
            "situation, (2) the key parameters/values it sets, (3) the order of "
            "steps, (4) the expected effect and how success is judged, (5) any "
            "caveats (e.g. model bias). Do NOT reproduce the page; output only the "
            "distilled guidance the agent can act on."
        )
        human = (
            f"CURRENT PROBLEM / QUESTION:\n{question or '(general improvement of the reconstruction)'}\n\n"
            f"WHAT THE AGENT HAS TRIED/OBSERVED SO FAR:\n{self._recent_history_context()}\n\n"
            f"TUTORIAL SOURCE: {url}\n\n"
            f"FULL TUTORIAL TEXT:\n{full_text}"
        )
        try:
            resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
            digest = getattr(resp, "content", None) or (resp if isinstance(resp, str) else None)
            digest = (digest or "").strip()
            return digest or None
        except Exception as e:
            print(f"⚠️ Tutorial condensation failed for {url}: {e}")
            return None

    def _consult_cryosparc_guide_tool(self, input_str: str) -> str:
        """
        Advisor tool: consult the official CryoSPARC guide AND tutorial/case-study
        library for a processing problem.

        Input (plain question, or JSON):
          - question: the processing problem (auto-matches job pages + tutorials).
          - slug: DEEP READ a specific tutorial/case-study — fetches the WHOLE page
            and returns a condensed, actionable digest (tools + params + steps),
            grounded in the current question + what the agent has tried.
          - list_tutorials: true -> browse the whole tutorial/case-study catalog.
        Advisory only.
        """
        try:
            from cryoagent.tools.cryosparc_guide_tools import consult_cryosparc_guide
            question = input_str if isinstance(input_str, str) else ""
            slug = None
            want_list = False
            params = self._parse_tool_input(input_str)
            if isinstance(params, dict):
                question = params.get("question", question if not params else "") or ""
                slug = params.get("slug")
                want_list = self._parse_bool_param(params.get("list_tutorials"), False) or \
                    self._parse_bool_param(params.get("list"), False)

            # DEEP-READ path: fetch the whole tutorial, condense to a digest, and
            # return ONLY the digest (the full page never enters the main context).
            if slug:
                res = consult_cryosparc_guide(question, slug=slug, full=True)
                if not res.get("success"):
                    self._record_tool_execution(
                        "consult_cryosparc_guide", {"slug": slug}, result={"success": False})
                    return f"ℹ️ {res.get('message')}"
                page = (res.get("pages") or [{}])[0]
                url = page.get("url", "")
                full_text = page.get("full_text") or page.get("excerpt") or ""
                digest = None
                # Only condense when the page is long enough to be worth it.
                if len(full_text) > 1500:
                    digest = self._condense_tutorial(full_text, url, question)
                self._record_tool_execution(
                    "consult_cryosparc_guide",
                    {"slug": slug, "condensed": digest is not None, "chars_in": len(full_text)},
                    result={"success": True, "url": url})
                if digest:
                    return (f"📖 CryoSPARC tutorial digest — {url}\n"
                            f"(condensed from the full page for: {question or 'general improvement'})\n\n{digest}")
                # Fallback: no/failed condensation -> return the excerpt directly.
                return f"📖 CryoSPARC guide — {url}\n{page.get('excerpt', full_text[:1200])}"

            res = consult_cryosparc_guide(question, list_tutorials=want_list)
            self._record_tool_execution(
                "consult_cryosparc_guide",
                {"question": question, "list_tutorials": want_list},
                result=res)

            # Browse-catalog response.
            if res.get("tutorials"):
                lines = [f"📚 CryoSPARC tutorial library ({res.get('message')}):"]
                for t in res["tutorials"]:
                    lines.append(f"- [{t['slug']}] {t['title']} — {t['when']}")
                return "\n".join(lines)

            if not res.get("success"):
                return f"ℹ️ {res.get('message')}"

            lines = [f"📖 CryoSPARC guide ({res.get('message')}):"]
            for p in res.get("pages", []):
                lines.append(f"\n— {p['url']}\n{p['excerpt']}")
            related = res.get("related_tutorials") or []
            if related:
                lines.append("\nRelated tutorials/case-studies (deep-read with slug=<slug>):")
                for t in related:
                    lines.append(f"- [{t['slug']}] {t['title']} — {t['when']}")
            return "\n".join(lines)
        except Exception as e:
            self._record_tool_execution("consult_cryosparc_guide", {"raw_input": input_str}, error=str(e))
            return f"❌ Error consulting CryoSPARC guide: {str(e)}"


    # ------------------------------------------------------------------
    # SPA resolution-improvement + deep-picking tool adapters (shared).
    # Thin LLM-facing wrappers over the CryoSPARCTools methods of the same
    # name. Defined on the base agent so every stage AND the improvement
    # agent can use them without per-stage duplication.
    # ------------------------------------------------------------------
    def _invoke_job_tool(self, wrapper_name: str, input_str: str, *,
                         required: Sequence[str],
                         optional: Optional[Sequence[str]] = None) -> str:
        """Generic adapter: parse input, validate, call the named CryoSPARCTools
        wrapper with required+optional args, forward raw `params`, record, return.

        `required`/`optional` entries are BOTH the tool-input key and the wrapper
        keyword argument (they share names by design).
        """
        params = self._parse_tool_input(input_str)
        missing = [k for k in required if not params.get(k)]
        if missing:
            return json.dumps({"success": False,
                               "error": f"Missing required parameters: {', '.join(missing)}"})
        call: Dict[str, Any] = {
            "project_uid": params.get("project_uid", self.config.workflow.project_uid),
            "workspace_uid": params.get("workspace_uid", self.config.workflow.workspace_uid),
        }
        for k in required:
            call[k] = params.get(k)
        for k in (optional or []):
            if params.get(k) is not None:
                call[k] = params.get(k)
        consumed = (["project_uid", "workspace_uid", "wait_for_completion",
                     "timeout", "check_interval"] + list(required) + list(optional or []))
        passthrough = self._extract_passthrough_params(params, consumed_keys=consumed)
        if passthrough:
            call["params"] = passthrough
        call["wait_for_completion"] = self._parse_bool_param(params.get("wait_for_completion"), False)
        if params.get("timeout") is not None:
            call["timeout"] = int(params.get("timeout"))
        if params.get("check_interval") is not None:
            call["check_interval"] = int(params.get("check_interval"))
        try:
            result = getattr(self.cryosparc_tools, wrapper_name)(**call)
        except Exception as e:
            self._record_tool_execution(wrapper_name, call, error=str(e))
            return json.dumps({"success": False, "error": str(e)})
        self._record_tool_execution(wrapper_name, call, result=result)
        return json.dumps(result)

    def _ctf_refine_global_tool(self, input_str: str) -> str:
        return self._invoke_job_tool("ctf_refine_global", input_str,
            required=["particles_job_uid", "volume_job_uid"], optional=["mask_job_uid"])

    def _ctf_refine_local_tool(self, input_str: str) -> str:
        return self._invoke_job_tool("ctf_refine_local", input_str,
            required=["particles_job_uid", "volume_job_uid"], optional=["mask_job_uid"])

    def _local_refinement_tool(self, input_str: str) -> str:
        return self._invoke_job_tool("local_refinement", input_str,
            required=["particles_job_uid", "volume_job_uid"], optional=["mask_job_uid"])

    def _particle_subtract_tool(self, input_str: str) -> str:
        return self._invoke_job_tool("particle_subtract", input_str,
            required=["particles_job_uid", "volume_job_uid"], optional=["mask_job_uid"])

    def _symmetry_expansion_tool(self, input_str: str) -> str:
        return self._invoke_job_tool("symmetry_expansion", input_str,
            required=["particles_job_uid"], optional=["symmetry"])

    def _sharpen_tool(self, input_str: str) -> str:
        return self._invoke_job_tool("sharpen", input_str,
            required=["volume_job_uid"], optional=["mask_job_uid", "bfactor"])

    def _deepemhancer_tool(self, input_str: str) -> str:
        return self._invoke_job_tool("deepemhancer", input_str,
            required=["volume_job_uid"], optional=["mask_job_uid"])

    def _local_resolution_tool(self, input_str: str) -> str:
        return self._invoke_job_tool("local_resolution", input_str,
            required=["volume_job_uid"], optional=["mask_job_uid"])

    def _class_3d_tool(self, input_str: str) -> str:
        return self._invoke_job_tool("class_3d", input_str,
            required=["particles_job_uid"],
            optional=["volume_job_uid", "mask_job_uid", "focus_mask_job_uid", "num_classes"])

    def _variability_3d_tool(self, input_str: str) -> str:
        return self._invoke_job_tool("variability_3d", input_str,
            required=["particles_job_uid"], optional=["mask_job_uid", "num_modes", "symmetry"])

    def _remove_duplicate_particles_tool(self, input_str: str) -> str:
        return self._invoke_job_tool("remove_duplicate_particles", input_str,
            required=["particles_job_uid"], optional=["micrographs_job_uid", "min_dist_A"])

    def _downsample_particles_tool(self, input_str: str) -> str:
        return self._invoke_job_tool("downsample_particles", input_str,
            required=["particles_job_uid"], optional=["box_size_pix", "bin_size_pix"])

    def _topaz_train_tool(self, input_str: str) -> str:
        return self._invoke_job_tool("topaz_train", input_str,
            required=["micrographs_job_uid", "particles_job_uid"])

    def _topaz_extract_tool(self, input_str: str) -> str:
        return self._invoke_job_tool("topaz_extract", input_str,
            required=["model_job_uid", "micrographs_job_uid"])

    def _topaz_denoise_tool(self, input_str: str) -> str:
        return self._invoke_job_tool("topaz_denoise", input_str,
            required=["micrographs_job_uid"], optional=["denoise_model_job_uid"])

    def _create_templates_tool(self, input_str: str) -> str:
        return self._invoke_job_tool("create_templates", input_str,
            required=["volume_job_uid"], optional=["num_projections"])

    def _class_2d_new_tool(self, input_str: str) -> str:
        return self._invoke_job_tool("class_2d_new", input_str,
            required=["particles_job_uid"], optional=["num_classes"])

    def _record_tool_execution(
        self,
        tool_name: str,
        params: Dict[str, Any],
        *,
        result: Optional[Any] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Record tool execution for analysis and debugging.
        
        Args:
            tool_name: Name of the tool executed
            params: Parameters passed to the tool
            result: Result returned by the tool (optional)
            error: Error message if tool failed (optional)
        """
        entry: Dict[str, Any] = {
            "tool": tool_name,
            "timestamp": time.time(),
            "params": dict(params) if params else {}
        }
        if result is not None:
            entry["result"] = result
        if error is not None:
            entry["error"] = error
        self.tool_execution_log.append(entry)
        
        # Also log to real-time conversation log
        if self.enable_conversation_logging and self.realtime_logger.current_log_file:
            result_str = str(result) if result is not None else "No result"
            if error is not None:
                result_str = f"ERROR: {error}"
            self.realtime_logger.log_tool_execution(tool_name, params, result_str)
    
    def get_tool_execution_log(self) -> List[Dict[str, Any]]:
        """Return a copy of the recorded tool executions."""
        return [entry.copy() for entry in self.tool_execution_log]
    
    def clear_tool_execution_log(self) -> None:
        """Clear the recorded tool execution history."""
        self.tool_execution_log = []
    
    def _check_and_resume_from_log(self, conversation_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Check for existing log file and return resume context if found.

        For most stages, completed logs are still returned as reference context.
        For the improvement stage, a completed log is ignored so a new --improve
        round starts a fresh conversation instead of appending to the last one.

        Args:
            conversation_id: Optional conversation identifier

        Returns:
            Resume context dictionary if a resumable log file is found, None otherwise
        """
        try:
            stage_name = getattr(self, 'stage_name', 'unknown')
            
            # Get outputs directory from realtime logger
            if hasattr(self.realtime_logger, 'outputs_dir'):
                outputs_dir = self.realtime_logger.outputs_dir
            else:
                outputs_dir = Path('outputs')
            
            # Initialize log resume parser
            log_parser = LogResumeParser(outputs_dir=str(outputs_dir))
            
            # Find log file for this stage
            log_file = log_parser.find_log_file(stage_name, conversation_id)
            
            if not log_file:
                return None
            
            # Always get context from log file, even if completed
            # This allows the agent to recognize existing work
            parsed = log_parser.parse_log_file(log_file)
            is_completed = parsed.get("completed", False)

            # A finished --improve run must start a fresh conversation, not
            # append to (or re-read as resume context) the completed log.
            if stage_name == "improvement" and is_completed:
                return None
            
            # Get resume context
            resume_context = log_parser.get_resume_context(log_file)
            
            # Mark if this is a completed log
            resume_context["is_completed"] = is_completed
            
            # If completed, update the resume message to indicate it's for reference
            if is_completed:
                resume_context["resume_message"] = (
                    f"Found completed log file from previous execution. "
                    f"This shows what work was already done. "
                    f"Total tools executed: {resume_context.get('num_tools_executed', 0)}. "
                    f"Review the completed work summary below and continue with any remaining work."
                )
            
            if self.config.agent.verbose:
                print(f"📋 Found existing log file: {log_file}")
                if is_completed:
                    print(f"✅ Previous execution completed. Providing context for reference.")
                else:
                    print(f"🔄 {resume_context.get('resume_message', 'Resuming from previous execution')}")
            
            return resume_context
            
        except Exception as e:
            if self.config.agent.verbose:
                print(f"⚠️ Warning: Failed to check for resume log: {e}")
            return None
    
    def run_react_workflow(self, workflow_input: str, conversation_id: Optional[str] = None) -> str:
        """
        Run a workflow using ReAct approach.
        
        Args:
            workflow_input: Description of the workflow to run
            conversation_id: Optional conversation identifier for memory control
            
        Returns:
            Result of the workflow execution
        """
        try:
            # Check for existing log file and resume context
            resume_context = self._check_and_resume_from_log(conversation_id)
            
            # If resuming or found existing log, use the existing log file and add resume message
            if resume_context:
                is_completed = resume_context.get("is_completed", False)
                
                # Use the existing conversation ID from the log
                if resume_context.get("conversation_id"):
                    conversation_id = resume_context["conversation_id"]
                
                # Restore the log file path to the realtime logger
                log_file = resume_context.get("log_file")
                if log_file and hasattr(self.realtime_logger, 'current_log_file'):
                    self.realtime_logger.current_log_file = log_file
                    self.realtime_logger.conversation_id = resume_context.get("conversation_id")
                    self.realtime_logger.stage_name = resume_context.get("stage_name")
                    self.realtime_logger.workflow_type = resume_context.get("workflow_type")
                    self.realtime_logger.start_time = resume_context.get("start_time")
                
                # Add resume information to workflow input
                resume_message = resume_context.get("resume_message", "")
                last_tool = resume_context.get("last_tool_execution")
                completed_work = resume_context.get("completed_work")
                all_tool_executions = resume_context.get("all_tool_executions", [])
                
                if last_tool or all_tool_executions:
                    # Add context about what was already done
                    workflow_input_parts = [workflow_input, "", resume_message]
                    
                    # Add completed work summary if available (especially important for heterogeneity)
                    if completed_work:
                        workflow_input_parts.append("")
                        workflow_input_parts.append("=== COMPLETED WORK SUMMARY ===")
                        workflow_input_parts.append(completed_work)
                    
                    # Add summary of all tool executions
                    if all_tool_executions:
                        workflow_input_parts.append("")
                        workflow_input_parts.append("=== ALL PREVIOUS TOOL EXECUTIONS ===")
                        workflow_input_parts.append(f"Total tools executed: {len(all_tool_executions)}")
                        for i, tool_exec in enumerate(all_tool_executions[-10:], 1):  # Show last 10 tools
                            tool_name = tool_exec.get("tool_name", "unknown")
                            tool_result = tool_exec.get("result")
                            if isinstance(tool_result, dict) and tool_result.get("success"):
                                result_summary = f"✅ {tool_name}"
                                if tool_result.get("job_uid"):
                                    result_summary += f" → {tool_result.get('job_uid')}"
                                workflow_input_parts.append(f"  {i}. {result_summary}")
                            elif isinstance(tool_result, dict) and not tool_result.get("success"):
                                workflow_input_parts.append(f"  {i}. ❌ {tool_name} (failed)")
                            else:
                                workflow_input_parts.append(f"  {i}. {tool_name}")
                        if len(all_tool_executions) > 10:
                            workflow_input_parts.append(f"  ... and {len(all_tool_executions) - 10} more tools")
                    
                    if last_tool:
                        workflow_input_parts.append("")
                        workflow_input_parts.append("Previous execution context:")
                        workflow_input_parts.append(f"- Last tool executed: {last_tool['tool_name']}")
                        workflow_input_parts.append(f"- Last tool arguments: {json.dumps(last_tool['arguments'], indent=2)}")
                        
                        # Show last tool result (truncated if too long)
                        last_result = last_tool.get('result')
                        if isinstance(last_result, dict):
                            result_str = json.dumps(last_result, indent=2)
                        else:
                            result_str = str(last_result)[:500] if last_result else "No result"
                        workflow_input_parts.append(f"- Last tool result: {result_str}")
                    
                    workflow_input_parts.append("")
                    if is_completed:
                        workflow_input_parts.append("NOTE: The previous execution completed successfully.")
                        workflow_input_parts.append("Review the completed work above. If you need to continue or redo any work, proceed accordingly.")
                        workflow_input_parts.append("IMPORTANT: Do NOT re-run work that is already completed (as shown in the summary above).")
                    else:
                        workflow_input_parts.append("Please continue from where the previous execution left off.")
                        if completed_work:
                            workflow_input_parts.append("IMPORTANT: Do NOT re-run work that is already completed (as shown in the summary above).")
                            workflow_input_parts.append("Continue with the next steps that have NOT been completed yet.")
                        else:
                            workflow_input_parts.append("Review the last tool execution and proceed with the next steps in the workflow.")
                    
                    workflow_input = "\n".join(workflow_input_parts)
                else:
                    workflow_input = f"""{workflow_input}

{resume_message}

Please continue with the workflow execution."""
                
                if self.config.agent.verbose:
                    if is_completed:
                        print(f"📋 Found completed log file: {log_file}")
                    else:
                        print(f"🔄 Resuming from log file: {log_file}")
            else:
                # Start fresh conversation logging if enabled
                if self.enable_conversation_logging:
                    self._start_realtime_conversation_log(workflow_input, conversation_id)
            
            # Reset execution log for this run (unless resuming)
            if not resume_context:
                self.clear_tool_execution_log()
            else:
                # If resuming (not completed), restore tool execution log from previous run
                # This helps maintain context about what was already done
                # For completed logs, we don't restore the log (context is provided in workflow_input)
                is_completed = resume_context.get("is_completed", False)
                if not is_completed and resume_context.get("all_tool_executions"):
                    # Restore previous tool executions to memory only (don't log to file again)
                    # Directly append to tool_execution_log to avoid duplicate log entries
                    for tool_exec in resume_context["all_tool_executions"]:
                        entry: Dict[str, Any] = {
                            "tool": tool_exec.get("tool_name", "unknown"),  # Map tool_name to tool
                            "timestamp": tool_exec.get("timestamp", time.time()),
                            "params": tool_exec.get("arguments", {})  # Map arguments to params
                        }
                        if "result" in tool_exec:
                            entry["result"] = tool_exec["result"]
                        if "error" in tool_exec:
                            entry["error"] = tool_exec["error"]
                        self.tool_execution_log.append(entry)
                    if self.config.agent.verbose:
                        print(f"📋 Restored {len(resume_context['all_tool_executions'])} previous tool executions (in memory only, not logged to file)")
                elif is_completed:
                    if self.config.agent.verbose:
                        print(f"📋 Found completed log with {len(resume_context.get('all_tool_executions', []))} tool executions. Context provided in workflow input.")

            # Check if memory should be cleared
            if self._should_clear_memory(conversation_id):
                self._clear_agent_memory()
                if self.config.agent.verbose:
                    print("🧠 Memory cleared - starting fresh conversation")
            
            # Update conversation state
            self._update_conversation_state(conversation_id)
            
            # Force clear memory if configuration requires it (but not when resuming)
            if not resume_context and (self.config.memory_control.clear_memory_on_new_conversation and 
                not self.config.memory_control.maintain_context_between_interactions):
                self._clear_agent_memory()
                if self.config.agent.verbose:
                    print("🧠 Memory forcefully cleared for fresh start")
            
            # Add ReAct-specific instructions to the input
            react_input = self._format_react_input(workflow_input)
            
            # Log the formatted input in real-time (only if not resuming or if it's a new input)
            if self.enable_conversation_logging and not resume_context:
                self.realtime_logger.log_user_input(react_input)
            elif self.enable_conversation_logging and resume_context:
                # Log resume message
                self.realtime_logger.log_system_message(
                    f"Resuming workflow execution",
                    metadata={"resume_context": resume_context.get("resume_message")}
                )
                self.realtime_logger.log_user_input(react_input)
            
            # Attach a callback that captures the LLM's intermediate reasoning
            # (the "responded:" text before each tool call) and its final answer
            # into the realtime conversation log — not just the final output.
            invoke_config = None
            if self.enable_conversation_logging:
                try:
                    from ..utils.reasoning_log_callback import ReasoningLogCallbackHandler
                    invoke_config = {"callbacks": [ReasoningLogCallbackHandler(self.realtime_logger)]}
                except Exception:
                    invoke_config = None

            if invoke_config:
                result = self.agent_executor.invoke({"input": react_input}, config=invoke_config)
            else:
                result = self.agent_executor.invoke({"input": react_input})

            # The final answer is already captured by the callback (on_agent_finish);
            # only close the log here to avoid duplicating the final output.
            if self.enable_conversation_logging:
                self._end_realtime_conversation_log(success=True)

            # Also log to general LLM logger if available
            if self.general_llm_logger:
                stage_name = getattr(self, 'stage_name', 'unknown')
                self.general_llm_logger.log_llm_response(stage_name, result["output"])

            return result["output"]
            
        except Exception as e:
            error_msg = f"❌ ReAct workflow execution failed: {str(e)}"
            
            # Log the error in real-time
            if self.enable_conversation_logging:
                self.realtime_logger.log_error(str(e))
                self._end_realtime_conversation_log(success=False, error=str(e))
            
            return error_msg
    
    def _format_react_input(self, workflow_input: str) -> str:
        """
        Format workflow input with ReAct instructions.
        
        Args:
            workflow_input: Original workflow input
            
        Returns:
            Formatted input with ReAct instructions
        """
        return load_prompt(
            "shared/react-wrapper.md",
            {"workflow_input": workflow_input},
        )
    
    def run_single_step(self, step_description: str, conversation_id: Optional[str] = None) -> str:
        """
        Run a single processing step using ReAct approach.
        
        Args:
            step_description: Description of the step to run
            conversation_id: Optional conversation identifier
            
        Returns:
            Result of the step execution
        """
        return self.run_react_workflow(step_description, conversation_id)
    
    def get_reasoning_history(self) -> List[Dict[str, str]]:
        """Get the history of reasoning steps."""
        return self.reasoning_history.copy()
    
    def clear_reasoning_history(self):
        """Clear the reasoning history."""
        self.reasoning_history = []
    
    def _should_clear_memory(self, conversation_id: Optional[str] = None) -> bool:
        """
        Determine if memory should be cleared based on configuration.
        
        Args:
            conversation_id: Optional conversation identifier
            
        Returns:
            True if memory should be cleared
        """
        if self.config.memory_control.clear_memory_on_new_conversation:
            if conversation_id is not None and conversation_id != self.last_conversation_id:
                return True
            elif conversation_id is None and self.conversation_count > 0:
                return True
            elif self.conversation_count == 0 and not self.config.memory_control.maintain_context_between_interactions:
                return True
        
        if not self.config.memory_control.maintain_context_between_interactions:
            return True
            
        return False
    
    def _clear_agent_memory(self):
        """Clear all agent memory and reset conversation state."""
        self.reasoning_history = []
        self.conversation_memory = []
        self.agent_executor = self._create_agent_executor()
        self.llm = self._create_default_llm()
        self.clear_tool_execution_log()
        self.conversation_count = 0
        self.last_conversation_id = None
        
        if self.config.agent.verbose:
            print("🧠 Completely resetting agent state for fresh start")
    
    def _update_conversation_state(self, conversation_id: Optional[str] = None):
        """Update conversation state tracking."""
        if conversation_id is not None:
            self.last_conversation_id = conversation_id
        self.conversation_count += 1
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status and configuration."""
        return {
            "conversation_count": self.conversation_count,
            "last_conversation_id": self.last_conversation_id,
            "reasoning_history_length": len(self.reasoning_history),
            "conversation_memory_length": len(self.conversation_memory),
            "memory_control_config": {
                "clear_memory_on_new_conversation": self.config.memory_control.clear_memory_on_new_conversation,
                "maintain_context_between_interactions": self.config.memory_control.maintain_context_between_interactions
            }
        }
    
    def force_clear_memory(self):
        """Force clear all memory regardless of configuration."""
        self._clear_agent_memory()
        if self.config.agent.verbose:
            print("🧠 Memory forcefully cleared")
    
    def switch_model_provider(self, provider: str):
        """
        Switch the LLM provider dynamically.
        
        Args:
            provider: New provider name (openai, deepseek, panshi)
            
        Raises:
            ValueError: If provider is not supported or not configured
        """
        provider = provider.lower()
        supported_providers = LLMFactory.get_supported_providers()
        
        if provider not in supported_providers:
            raise ValueError(f"Unsupported provider: {provider}. Supported: {supported_providers}")
        
        if provider not in self.config.agent.models:
            raise ValueError(f"Provider '{provider}' not configured in config file")
        
        old_provider = self.config.agent.provider
        self.config.agent.provider = provider
        
        self.llm = self._create_default_llm()
        self.agent_executor = self._create_agent_executor()
        
        if self.config.agent.verbose:
            model_config = self.config.agent.get_current_model_config()
            print(f"🔄 Model provider switched from '{old_provider}' to '{provider}'")
            print(f"   Model: {model_config.model_name}")
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about the currently configured model."""
        model_config = self.config.agent.get_current_model_config()
        return {
            "provider": self.config.agent.provider,
            "model_name": model_config.model_name,
            "base_url": model_config.base_url,
            "temperature": model_config.temperature,
            "timeout": model_config.timeout,
            "available_providers": list(self.config.agent.models.keys())
        }
    
    # Common tool wrappers that can be used by all agents
    def _get_job_status_tool(self, input_str: str) -> str:
        """Common tool wrapper for getting job status."""
        params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            job_uid = params.get("job_uid")
            
            if not job_uid:
                return f"❌ Error: job_uid parameter is required. Input was: '{input_str}'"
            
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)

            status = self.cryosparc_tools.get_job_status(
                job_uid,
                project_uid=project_uid,
                workspace_uid=workspace_uid
            )
            progress_display = f"{status['progress']}%" if status.get("progress") is not None else "N/A"
            self._record_tool_execution("get_job_status", {
                "job_uid": job_uid,
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }, result=status)
            return f"📊 Job {job_uid} status: {status['status']} ({progress_display})"
            
        except Exception as e:
            context = params or {"raw_input": input_str}
            self._record_tool_execution("get_job_status", context, error=str(e))
            return f"❌ Error getting job status: {str(e)}"
    
    def _wait_for_job_tool(self, input_str: str) -> str:
        """Common tool wrapper for waiting for job completion."""
        params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            job_uid = params.get("job_uid")
            timeout = int(params.get("timeout", self.config.job_management.default_timeout))
            check_interval = int(params.get("check_interval", self.config.job_management.status_check_interval))
            
            if not job_uid:
                return f"❌ Error: job_uid parameter is required. Input was: '{input_str}'"
            
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)

            # First, check the current job status
            print(f"🔍 Checking current status of job {job_uid}...")
            current_status = self.cryosparc_tools.get_job_status(
                job_uid,
                project_uid=project_uid,
                workspace_uid=workspace_uid
            )
            
            # If job is already completed, return immediately without starting retry
            if current_status["status"] in ["completed", "failed", "cancelled"]:
                print(f"ℹ️ Job {job_uid} is already {current_status['status']}. No retry needed.")
                self._record_tool_execution(
                    "wait_for_job",
                    {
                        "job_uid": job_uid,
                        "project_uid": project_uid,
                        "workspace_uid": workspace_uid,
                        "timeout": timeout,
                        "check_interval": check_interval
                    },
                    result=current_status
                )
                return f"✅ Job {job_uid} is already {current_status['status']}. No adaptive retry needed."
            
            # Only wait for completion if job is still running
            print(f"🛰️ Job {job_uid} is {current_status['status']}. Waiting for completion (checking every {check_interval}s)...")
            status = self.cryosparc_tools.wait_for_job_completion(
                project_uid,
                job_uid,
                workspace_uid,
                timeout,
                check_interval
            )
            self._record_tool_execution(
                "wait_for_job",
                {
                    "job_uid": job_uid,
                    "project_uid": project_uid,
                    "workspace_uid": workspace_uid,
                    "timeout": timeout,
                    "check_interval": check_interval
                },
                result=status
            )
            return f"✅ Job {job_uid} completed with status: {status['status']}"
            
        except Exception as e:
            context = params or {"raw_input": input_str}
            self._record_tool_execution("wait_for_job", context, error=str(e))
            return f"❌ Error waiting for job: {str(e)}"
    
    def _get_job_log_tool(self, input_str: str) -> str:
        """Common tool wrapper for reading job logs."""
        params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            job_uid = params.get("job_uid")
            
            if not job_uid:
                return f"❌ Error: job_uid parameter is required. Input was: '{input_str}'"
            
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)

            log_result = self.cryosparc_tools.get_job_log(
                job_uid,
                project_uid=project_uid,
                workspace_uid=workspace_uid
            )
            
            self._record_tool_execution("get_job_log", {
                "job_uid": job_uid,
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }, result=log_result)
            
            if log_result.get("success", False):
                analysis = log_result.get("error_analysis", {})
                summary = analysis.get("summary", "No analysis available")

                response = f"📋 Job {job_uid} log analysis:\n{summary}\n"

                if analysis.get("critical_errors"):
                    response += f"\n🚨 Error lines from log:\n" + "\n".join(f"  - {e}" for e in analysis["critical_errors"][:5])

                if analysis.get("warnings"):
                    response += f"\n⚠️ Warning lines from log:\n" + "\n".join(f"  - {w}" for w in analysis["warnings"][:5])

                return response
            else:
                return f"❌ Error reading job log: {log_result.get('error', 'Unknown error')}"
            
        except Exception as e:
            context = params or {"raw_input": input_str}
            self._record_tool_execution("get_job_log", context, error=str(e))
            return f"❌ Error reading job log: {str(e)}"

    def _search_cryosparc_forum_tool(self, input_str: str) -> str:
        """Search CryoSPARC Discuss for troubleshooting advice related to a job failure."""
        params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            query = params.get("query") or params.get("search_query")
            job_uid = params.get("job_uid")
            max_results = int(params.get("max_results", 5))

            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)

            queries: List[str] = []
            if query:
                queries.append(str(query))

            if job_uid:
                log_result = self.cryosparc_tools.get_job_log(
                    job_uid,
                    project_uid=project_uid,
                    workspace_uid=workspace_uid,
                )
                if log_result.get("success"):
                    log_queries = extract_search_queries_from_log(
                        log_result.get("log_content", ""),
                        log_result.get("error_analysis"),
                    )
                    for candidate in log_queries:
                        if candidate not in queries:
                            queries.append(candidate)
                elif not queries:
                    return (
                        f"❌ Could not read job log for {job_uid}: "
                        f"{log_result.get('error', 'unknown error')}. "
                        "Provide a explicit query parameter to search the forum."
                    )

            if not queries:
                return (
                    "❌ Error: provide query (error keywords) and/or job_uid "
                    f"(failed job whose log should be searched). Input was: '{input_str}'"
                )

            if len(queries) == 1:
                search_payload = search_cryosparc_forum(queries[0], max_results=max_results)
            else:
                search_payload = search_cryosparc_forum_multi(
                    queries,
                    max_results_per_query=3,
                    max_total_results=max_results,
                )

            self._record_tool_execution(
                "search_cryosparc_forum",
                {
                    "query": query,
                    "job_uid": job_uid,
                    "queries": queries,
                    "project_uid": project_uid,
                    "workspace_uid": workspace_uid,
                    "max_results": max_results,
                },
                result=search_payload,
            )
            return format_forum_search_response(search_payload)

        except Exception as e:
            context = params or {"raw_input": input_str}
            self._record_tool_execution("search_cryosparc_forum", context, error=str(e))
            return f"❌ Error searching CryoSPARC forum: {str(e)}"
    
    def _start_realtime_conversation_log(self, workflow_input: str, conversation_id: Optional[str] = None):
        """Start real-time logging of a conversation."""
        if not conversation_id:
            conversation_id = f"conversation_{int(time.time())}"
        
        # Determine stage name from the agent type
        stage_name = getattr(self, 'stage_name', 'unknown')
        workflow_type = getattr(self, 'workflow_type', 'cryoem')
        
        # Start real-time logging
        log_file = self.realtime_logger.start_conversation(
            conversation_id=conversation_id,
            workflow_type=workflow_type,
            stage_name=stage_name
        )
        
        # Log the initial workflow input
        self.realtime_logger.log_system_message(
            f"Starting {stage_name} workflow",
            metadata={"workflow_input": workflow_input}
        )
        
        if self.config.agent.verbose:
            print(f"💬 Real-time LLM conversation log: {log_file}")
    
    def _end_realtime_conversation_log(self, success: bool, error: Optional[str] = None):
        """End the real-time conversation log."""
        summary = "Workflow completed successfully" if success else f"Workflow failed: {error}"
        self.realtime_logger.end_conversation(success=success, summary=summary)
        
        if self.config.agent.verbose:
            log_file = self.realtime_logger.get_log_file_path()
            if log_file:
                print(f"💬 Real-time LLM conversation log completed: {log_file}")
    
    def set_general_llm_logger(self, logger: GeneralLLMLogger):
        """Set the general LLM logger for this agent."""
        self.general_llm_logger = logger
    
    def get_conversation_log_file(self) -> Optional[str]:
        """Get the path to the most recent conversation log file."""
        # Return the real-time log file if available
        realtime_log = self.realtime_logger.get_log_file_path()
        if realtime_log:
            return realtime_log
        
        # Fallback to the JSON conversation log
        if hasattr(self.conversation_logger, 'current_log') and self.conversation_logger.current_log:
            try:
                return self.conversation_logger.save_conversation_log()
            except:
                return None
        return None

