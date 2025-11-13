"""Base ReAct (Reasoning + Acting) agent with common logic for all CryoEM agents."""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Sequence
from abc import ABC, abstractmethod
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

from ..tools.cryosparc_tools import CryoSPARCTools
from ..config.config_loader import CryoAgentConfig
from .llm_factory import LLMFactory
from ..utils.conversation_logger import ConversationLogger
from ..utils.realtime_conversation_logger import RealtimeConversationLogger
from ..utils.general_llm_logger import GeneralLLMLogger


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

        config_path = Path("configs/microscope_config.json")
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
            max_iterations=self.config.agent.max_iterations,
            handle_parsing_errors=True
        )
    
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
            # Start real-time conversation logging if enabled
            if self.enable_conversation_logging:
                self._start_realtime_conversation_log(workflow_input, conversation_id)
            
            # Reset execution log for this run
            self.clear_tool_execution_log()

            # Check if memory should be cleared
            if self._should_clear_memory(conversation_id):
                self._clear_agent_memory()
                if self.config.agent.verbose:
                    print("🧠 Memory cleared - starting fresh conversation")
            
            # Update conversation state
            self._update_conversation_state(conversation_id)
            
            # Force clear memory if configuration requires it
            if (self.config.memory_control.clear_memory_on_new_conversation and 
                not self.config.memory_control.maintain_context_between_interactions):
                self._clear_agent_memory()
                if self.config.agent.verbose:
                    print("🧠 Memory forcefully cleared for fresh start")
            
            # Add ReAct-specific instructions to the input
            react_input = self._format_react_input(workflow_input)
            
            # Log the formatted input in real-time
            if self.enable_conversation_logging:
                self.realtime_logger.log_user_input(react_input)
            
            result = self.agent_executor.invoke({"input": react_input})
            
            # Log the result in real-time
            if self.enable_conversation_logging:
                self.realtime_logger.log_assistant_response(result["output"])
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
        return f"""
Execute the following workflow using the ReAct framework:

{workflow_input}

Remember to:
1. Start with reasoning about the workflow state
2. Follow the Thought → Action → Observation pattern
3. Check dependencies before each step
4. Wait for jobs to complete before proceeding
5. Provide clear status updates

Begin with reasoning about the current workflow state.
"""
    
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
                suggestions = analysis.get("suggestions", [])
                
                response = f"📋 Job {job_uid} log analysis:\n{summary}\n"
                if suggestions:
                    response += f"\n💡 Suggestions:\n" + "\n".join(f"  - {s}" for s in suggestions)
                
                if analysis.get("critical_errors"):
                    response += f"\n🚨 Critical errors found:\n" + "\n".join(f"  - {e}" for e in analysis["critical_errors"][:5])
                
                return response
            else:
                return f"❌ Error reading job log: {log_result.get('error', 'Unknown error')}"
            
        except Exception as e:
            context = params or {"raw_input": input_str}
            self._record_tool_execution("get_job_log", context, error=str(e))
            return f"❌ Error reading job log: {str(e)}"
    
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

