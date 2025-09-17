"""Configuration loader for CryoAgent."""

import json
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


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


class WorkflowSettings(BaseModel):
    """Settings for cryoEM workflow parameters."""
    
    # Import movies parameters
    movies_path: str = Field(default="/path/to/your/movies", description="Path to movie files")
    gain_ref_path: Optional[str] = Field(default="/path/to/gain_ref.mrc", description="Path to gain reference file")
    pixel_size: float = Field(default=1.0, description="Pixel size in Angstroms")
    voltage: float = Field(default=300.0, description="Acceleration voltage in kV")
    cs_mm: float = Field(default=2.7, description="Spherical aberration in mm")
    dose: float = Field(default=1.0, description="Electron dose per frame in e-/Å²")
    
    # Motion correction parameters
    motion_correction_binning: int = Field(default=1, description="Binning for motion correction")
    motion_correction_patch_size: int = Field(default=5, description="Patch size for motion correction")
    
    # CTF estimation parameters
    ctf_min_res: float = Field(default=30.0, description="Minimum resolution for CTF estimation")
    ctf_max_res: float = Field(default=4.0, description="Maximum resolution for CTF estimation")
    
    # Project and workspace settings
    project_uid: str = Field(default="P1", description="CryoSPARC project UID")
    workspace_uid: str = Field(default="W1", description="CryoSPARC workspace UID")
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "CRYOEM_"


class AgentSettings(BaseModel):
    """Settings for LangChain agent configuration."""
    
    model_name: str = Field(default="deepseek-chat", description="LLM model name")
    temperature: float = Field(default=0.1, description="LLM temperature")
    max_iterations: int = Field(default=10, description="Maximum agent iterations")
    verbose: bool = Field(default=True, description="Enable verbose logging")
    api_key: str = Field(default="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", description="DeepSeek API key")
    base_url: str = Field(default="https://api.deepseek.com", description="API base URL")
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "AGENT_"


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
    agent: AgentSettings
    workflow: WorkflowSettings
    job_management: JobManagementSettings
    logging: LoggingSettings
    react_workflow: ReActWorkflowSettings
    error_handling: ErrorHandlingSettings
    performance: PerformanceSettings
    
    # ReAct specific settings
    react: ReActSettings


class ConfigLoader:
    """Configuration loader for CryoAgent."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Path to the configuration file. If None, uses default config.json
        """
        self.config_path = config_path or "config.json"
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
        
        return self._parse_config(config_data)
    
    def _parse_config(self, config_data: Dict[str, Any]) -> CryoAgentConfig:
        """Parse configuration data into structured objects."""
        # Parse CryoSPARC settings
        cryosparc_settings = CryoSPARCSettings(**config_data.get("cryosparc", {}))
        
        # Parse agent settings
        agent_data = config_data.get("agent", {})
        react_data = agent_data.pop("react", {})
        agent_settings = AgentSettings(**agent_data)
        
        # Parse workflow settings
        workflow_data = config_data.get("workflow", {})
        
        # Extract import movies parameters
        import_movies = workflow_data.pop("import_movies", {})
        motion_correction = workflow_data.pop("motion_correction", {})
        ctf_estimation = workflow_data.pop("ctf_estimation", {})
        
        # Merge all workflow parameters
        workflow_params = {
            **workflow_data,
            **import_movies,
            "motion_correction_binning": motion_correction.get("binning", 1),
            "motion_correction_patch_size": motion_correction.get("patch_size", 5),
            "ctf_min_res": ctf_estimation.get("min_res", 30.0),
            "ctf_max_res": ctf_estimation.get("max_res", 4.0),
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
        
        return CryoAgentConfig(
            cryosparc=cryosparc_settings,
            agent=agent_settings,
            workflow=workflow_settings,
            job_management=job_management,
            logging=logging,
            react_workflow=react_workflow,
            error_handling=error_handling,
            performance=performance,
            react=react_settings
        )
    
    def get_cryosparc_settings(self) -> CryoSPARCSettings:
        """Get CryoSPARC settings."""
        return self.load_config().cryosparc
    
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
