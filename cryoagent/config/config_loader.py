"""Configuration loader for CryoAgent."""

import json
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def resolve_env_vars(data: Any) -> Any:
    """
    Resolve environment variables in configuration data.
    Supports ${VARIABLE_NAME} pattern for API keys and other variables.
    Returns empty string for missing environment variables instead of raising error.
    """
    if isinstance(data, dict):
        return {key: resolve_env_vars(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [resolve_env_vars(item) for item in data]
    elif isinstance(data, str) and data.startswith('${') and data.endswith('}'):
        var_name = data[2:-1]  # Remove ${ and }
        env_value = os.environ.get(var_name)
        if env_value is None:
            # Return empty string for missing environment variables
            # This allows the system to work with only some API keys configured
            return ""
        return env_value
    else:
        return data


class CryoSPARCSettings(BaseModel):
    """Settings for CryoSPARC connection."""
    
    host: str = Field(default="localhost", description="CryoSPARC host address")
    base_port: int = Field(default=61000, description="CryoSPARC base port")
    username: Optional[str] = Field(default="your-username", description="CryoSPARC username")
    password: Optional[str] = Field(default="your-password", description="CryoSPARC password")
    license_id: Optional[str] = Field(default="your-cryosparc-license-id-here", description="CryoSPARC license ID")
    
    @classmethod
    def from_env(cls) -> "CryoSPARCSettings":
        """Create settings from environment variables."""
        return cls(
            host=os.getenv("CRYOSPARC_HOST", "localhost"),
            base_port=int(os.getenv("CRYOSPARC_BASE_PORT", "61000")),
            username=os.getenv("CRYOSPARC_USERNAME"),
            password=os.getenv("CRYOSPARC_PASSWORD"),
            license_id=os.getenv("CRYOSPARC_LICENSE_ID"),
        )

class BackendExecutionSettings(BaseModel):
    """Settings for RELION backend execution."""
    
    enabled: bool = Field(default=False, description="Enable backend execution")
    default_timeout: int = Field(default=3600, description="Default timeout in seconds")
    check_interval: int = Field(default=30, description="Status check interval in seconds")
    max_concurrent_jobs: int = Field(default=3, description="Maximum concurrent backend jobs")
    auto_cleanup: bool = Field(default=True, description="Automatically cleanup completed jobs")


class RELIONSettings(BaseModel):
    """Settings for RELION connection and processing."""
    
    relion_exe: str = Field(default="/usr/local/bin/relion", description="RELION executable path")
    relion_dir: str = Field(default="/home/daoyi/relion/relion_test", description="RELION working directory")
    continue_job: bool = Field(default=True, description="Continue existing jobs")
    backend_execution: Optional[BackendExecutionSettings] = Field(default=None, description="Backend execution settings")


class WorkflowSettings(BaseModel):
    """Settings for cryoEM workflow parameters."""
    
    # Essential parameters only
    microscope_config_path: str = Field(default="configs/microscope_config.json", description="Path to microscope configuration file")
    project_uid: str = Field(default="P1", description="CryoSPARC project UID")
    workspace_uid: str = Field(default="W1", description="CryoSPARC workspace UID")
    particle_diameter: Optional[float] = Field(default=None, description="Global particle diameter override (Å)")
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "CRYOEM_"


class ModelConfig(BaseModel):
    """Configuration for a specific LLM model."""
    api_key: str = Field(description="API key for the model")
    base_url: str = Field(description="Base URL for the API")
    model_name: str = Field(description="Model name")
    temperature: float = Field(default=0.1, description="Model temperature")
    timeout: int = Field(default=60, description="Request timeout in seconds")


class AgentSettings(BaseModel):
    """Settings for LangChain agent configuration."""
    
    provider: str = Field(default="deepseek", description="LLM provider (deepseek, openai, panshi)")
    model_name: str = Field(default="deepseek-chat", description="LLM model name (legacy)")
    temperature: float = Field(default=0.1, description="LLM temperature (legacy)")
    max_iterations: int = Field(default=10, description="Maximum agent iterations")
    verbose: bool = Field(default=True, description="Enable verbose logging")
    timeout: int = Field(default=60, description="Request timeout in seconds")
    api_key: str = Field(default="", description="API key (legacy)")
    base_url: str = Field(default="", description="API base URL (legacy)")
    models: Dict[str, ModelConfig] = Field(default_factory=dict, description="Available model configurations")
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "AGENT_"
    
    def get_current_model_config(self) -> ModelConfig:
        """Get the configuration for the currently selected model provider."""
        if self.provider in self.models:
            return self.models[self.provider]
        else:
            # Fallback to legacy configuration
            return ModelConfig(
                api_key=self.api_key,
                base_url=self.base_url,
                model_name=self.model_name,
                temperature=self.temperature,
                timeout=self.timeout
            )
    
    def get_available_providers(self) -> list[str]:
        """Get list of providers that have valid API keys configured."""
        available = []
        for provider, model_config in self.models.items():
            if self._is_api_key_valid(model_config.api_key):
                available.append(provider)
        return available
    
    def auto_select_provider(self) -> str:
        """
        Automatically select the first available provider with a valid API key.
        
        Returns:
            Provider name that has a valid API key
            
        Raises:
            ValueError: If no providers have valid API keys
        """
        available_providers = self.get_available_providers()
        
        if not available_providers:
            # Check if legacy configuration has valid API key
            if self._is_api_key_valid(self.api_key):
                return "legacy"
            else:
                raise ValueError(
                    "No valid API keys found for any provider. "
                    "Please set one of: DEEPSEEK_API_KEY, OPENAI_API_KEY, or PANSHI_API_KEY"
                )
        
        # Return the first available provider
        selected = available_providers[0]
        self.provider = selected  # Update the current provider
        return selected
    
    def _is_api_key_valid(self, api_key: str) -> bool:
        """
        Check if an API key is valid (not empty, not placeholder).
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if API key appears valid, False otherwise
        """
        if not api_key or not api_key.strip():
            return False
        
        # Check for common placeholder patterns (be more specific)
        placeholder_patterns = [
            "your-api-key",
            "your-key",
            "example-key",
            "placeholder-key",
            "replace-with-your",
            "set-your-key"
        ]
        
        api_key_lower = api_key.lower()
        for pattern in placeholder_patterns:
            if pattern in api_key_lower:
                return False
        
        # Special handling for sk- keys (OpenAI format)
        if api_key.startswith("sk-"):
            # Valid OpenAI keys are typically longer and have specific format
            if len(api_key.strip()) < 20:
                return False
            return True
        
        # For other API keys (DeepSeek, Panshi, etc.), be more lenient
        # Just check that it's not obviously a placeholder and has reasonable length
        if len(api_key.strip()) < 10:
            return False
        
        return True


class MemoryControlSettings(BaseModel):
    """Memory control settings for the LLM agent."""
    clear_memory_on_new_conversation: bool = Field(default=True, description="Clear conversation history on new conversation")
    maintain_context_between_interactions: bool = Field(default=False, description="Maintain context between different interactions")


class ReActSettings(BaseModel):
    """ReAct-specific agent settings."""
    max_reasoning_steps: int = Field(default=10, description="Maximum reasoning steps per cycle")
    enable_self_reflection: bool = Field(default=True, description="Enable self-reflection in reasoning")
    require_explicit_reasoning: bool = Field(default=True, description="Require explicit reasoning before actions")


class JobManagementSettings(BaseModel):
    """Job management settings."""
    default_timeout: int = Field(default=3600, description="Default job timeout in seconds")
    status_check_interval: int = Field(default=30, description="Status check interval in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: int = Field(default=60, description="Delay between retries in seconds")


class LoggingSettings(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    file: str = Field(default="cryoagent.log", description="Log file path")
    max_file_size: str = Field(default="10MB", description="Maximum log file size")
    backup_count: int = Field(default=5, description="Number of backup log files")


class WorkflowStepConfig(BaseModel):
    """Configuration for a single workflow step."""
    name: str = Field(description="Step name")
    description: str = Field(description="Step description")
    required_params: list[str] = Field(description="Required parameters")
    optional_params: list[str] = Field(default_factory=list, description="Optional parameters")
    depends_on: list[str] = Field(default_factory=list, description="Dependencies")


class ReActWorkflowSettings(BaseModel):
    """ReAct workflow configuration."""
    steps: list[WorkflowStepConfig] = Field(description="Workflow steps configuration")


class ErrorHandlingSettings(BaseModel):
    """Error handling configuration."""
    max_consecutive_failures: int = Field(default=3, description="Maximum consecutive failures")
    fallback_strategies: list[str] = Field(default_factory=list, description="Fallback strategies")


class PerformanceSettings(BaseModel):
    """Performance configuration."""
    parallel_jobs: bool = Field(default=False, description="Allow parallel job execution")
    max_concurrent_jobs: int = Field(default=1, description="Maximum concurrent jobs")
    resource_monitoring: bool = Field(default=True, description="Enable resource monitoring")
    memory_limit: str = Field(default="8GB", description="Memory limit")


class CryoAgentConfig(BaseModel):
    """Complete CryoAgent configuration."""
    cryosparc: CryoSPARCSettings
    relion: RELIONSettings
    agent: AgentSettings
    workflow: WorkflowSettings
    job_management: JobManagementSettings
    logging: LoggingSettings
    react_workflow: ReActWorkflowSettings
    error_handling: ErrorHandlingSettings
    performance: PerformanceSettings
    
    # ReAct specific settings
    react: ReActSettings
    
    # Memory control settings
    memory_control: MemoryControlSettings


class ConfigLoader:
    """Configuration loader for CryoAgent."""
    
    def __init__(self, config_path: Optional[str] = None, master_config_path: Optional[str] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Path to the configuration file. If None, uses default config.json
            master_config_path: Path to the master configuration file for merging
        """
        self.config_path = config_path or "config.json"
        self.master_config_path = master_config_path
        self._config: Optional[CryoAgentConfig] = None
    
    def load_config(self) -> CryoAgentConfig:
        """
        Load configuration from JSON file.
        
        Returns:
            Loaded configuration object
        """
        if self._config is None:
            self._config = self._load_from_file()
        return self._config
    
    def _load_from_file(self) -> CryoAgentConfig:
        """Load configuration from JSON file."""
        config_path = Path(self.config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # If master config is provided, merge it with stage config
        if self.master_config_path:
            master_config_path = Path(self.master_config_path)
            if master_config_path.exists():
                with open(master_config_path, 'r') as f:
                    master_config_data = json.load(f)
                
                # Merge master config with stage config (stage config takes precedence for overlapping keys)
                config_data = self._merge_configs(master_config_data, config_data)
        
        # Resolve environment variables in the configuration data
        config_data = resolve_env_vars(config_data)
        
        return self._parse_config(config_data)

    def _merge_configs(self, master_config: Dict[str, Any], stage_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge master configuration with stage-specific configuration."""
        merged = master_config.copy()
        
        # Merge stage-specific sections
        for key, value in stage_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                merged[key] = self._merge_configs(merged[key], value)
            else:
                # Stage config takes precedence
                merged[key] = value
        
        return merged
    
    def _parse_config(self, config_data: Dict[str, Any]) -> CryoAgentConfig:
        """Parse configuration data into structured objects."""
        # Parse CryoSPARC settings
        cryosparc_settings = CryoSPARCSettings(**config_data.get("cryosparc", {}))
        
        # Parse RELION settings
        relion_settings = RELIONSettings(**config_data.get("relion", {}))
        
        # Parse agent settings
        agent_data = config_data.get("agent", {})
        react_data = agent_data.pop("react", {})
        memory_control_data = agent_data.pop("memory_control", {})
        agent_settings = AgentSettings(**agent_data)
        
        # Parse workflow settings - keep it simple, just pass the raw data
        workflow_data = config_data.get("workflow", {})
        
        # Only extract the essential parameters that are actually used
        workflow_params = {
            "microscope_config_path": workflow_data.get("import_movies", {}).get("microscope_config_path", "configs/microscope_config.json"),
            "project_uid": workflow_data.get("project_uid", "P1"),
            "workspace_uid": workflow_data.get("workspace_uid", "W1"),
            "particle_diameter": workflow_data.get("particle_diameter"),
        }
        
        workflow_settings = WorkflowSettings(**workflow_params)
        
        # Parse other settings
        job_management = JobManagementSettings(**config_data.get("job_management", {}))
        logging = LoggingSettings(**config_data.get("logging", {}))
        error_handling = ErrorHandlingSettings(**config_data.get("error_handling", {}))
        performance = PerformanceSettings(**config_data.get("performance", {}))
        # Parse ReAct workflow settings
        react_workflow_data = config_data.get("react_workflow", {})
        steps = [WorkflowStepConfig(**step) for step in react_workflow_data.get("steps", [])]
        react_workflow = ReActWorkflowSettings(steps=steps)
        
        # Parse ReAct settings
        react_settings = ReActSettings(**react_data)
        
        # Parse memory control settings
        memory_control_settings = MemoryControlSettings(**memory_control_data)
        
        return CryoAgentConfig(
            cryosparc=cryosparc_settings,
            relion=relion_settings,
            agent=agent_settings,
            workflow=workflow_settings,
            job_management=job_management,
            logging=logging,
            react_workflow=react_workflow,
            error_handling=error_handling,
            performance=performance,
            react=react_settings,
            memory_control=memory_control_settings
        )
    
    def get_cryosparc_settings(self) -> CryoSPARCSettings:
        """Get CryoSPARC settings."""
        return self.load_config().cryosparc
    
    def get_relion_settings(self) -> RELIONSettings:
        """Get RELION settings."""
        return self.load_config().relion
    
    def get_agent_settings(self) -> AgentSettings:
        """Get agent settings."""
        return self.load_config().agent
    
    def get_workflow_settings(self) -> WorkflowSettings:
        """Get workflow settings."""
        return self.load_config().workflow
    
    def get_react_settings(self) -> ReActSettings:
        """Get ReAct settings."""
        return self.load_config().react
    
    def get_react_workflow_settings(self) -> ReActWorkflowSettings:
        """Get ReAct workflow settings."""
        return self.load_config().react_workflow
    
    def get_job_management_settings(self) -> JobManagementSettings:
        """Get job management settings."""
        return self.load_config().job_management
    
    def get_logging_settings(self) -> LoggingSettings:
        """Get logging settings."""
        return self.load_config().logging
    
    def get_error_handling_settings(self) -> ErrorHandlingSettings:
        """Get error handling settings."""
        return self.load_config().error_handling
    
    def get_performance_settings(self) -> PerformanceSettings:
        """Get performance settings."""
        return self.load_config().performance
    
    def get_memory_control_settings(self) -> MemoryControlSettings:
        """Get memory control settings."""
        return self.load_config().memory_control
