"""ReAct (Reasoning + Acting) CryoEM agent implementation."""

from typing import Dict, Any, List, Optional, Tuple
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from ..tools.cryosparc_tools import CryoSPARCTools
from ..config.config_loader import ConfigLoader, CryoAgentConfig


class ReActCryoEMAgent:
    """ReAct-based CryoEM agent that uses explicit reasoning and acting cycles."""
    
    def __init__(
        self,
        cryosparc_tools: CryoSPARCTools,
        config: CryoAgentConfig,
        llm: Optional[BaseLanguageModel] = None
    ):
        """
        Initialize the ReAct CryoEM agent.
        
        Args:
            cryosparc_tools: CryoSPARC tools instance
            config: Complete configuration object
            llm: Language model for the agent
        """
        self.cryosparc_tools = cryosparc_tools
        self.config = config
        self.llm = llm or self._create_default_llm()
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent_executor()
        self.reasoning_history: List[Dict[str, str]] = []
    
    def _create_default_llm(self) -> BaseLanguageModel:
        """Create default language model."""
        return ChatOpenAI(
            model=self.config.agent.model_name,
            temperature=self.config.agent.temperature,
            api_key=self.config.agent.api_key,
            base_url=self.config.agent.base_url
        )
    
    def _create_tools(self) -> List[Tool]:
        """Create LangChain tools for CryoSPARC operations."""
        return [
            Tool(
                name="import_movies",
                description="Import movie files into CryoSPARC for processing. "
                           "Required parameters: movies_path, pixel_size, voltage, cs_mm, dose. "
                           "Optional parameters: gain_ref_path, project_uid, workspace_uid.",
                func=self._import_movies_tool
            ),
            Tool(
                name="motion_correction",
                description="Perform motion correction on imported movies. "
                           "Required parameters: movies_job_uid. "
                           "Optional parameters: binning, patch_size, max_shift, project_uid, workspace_uid.",
                func=self._motion_correction_tool
            ),
            Tool(
                name="ctf_estimation",
                description="Estimate CTF parameters for micrographs. "
                           "Required parameters: micrographs_job_uid. "
                           "Optional parameters: min_res, max_res, defocus_range, project_uid, workspace_uid.",
                func=self._ctf_estimation_tool
            ),
            Tool(
                name="get_job_status",
                description="Check the status of a CryoSPARC job. "
                           "Required parameters: job_uid.",
                func=self._get_job_status_tool
            ),
            Tool(
                name="wait_for_job",
                description="Wait for a job to complete and return final status. "
                           "Required parameters: job_uid. "
                           "Optional parameters: timeout.",
                func=self._wait_for_job_tool
            ),
            Tool(
                name="reason_about_workflow",
                description="Analyze the current workflow state and determine next steps. "
                           "Use this to think through the workflow progression and identify dependencies.",
                func=self._reason_about_workflow_tool
            )
        ]
    
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
    
    def _get_react_system_prompt(self) -> str:
        """Get the ReAct-style system prompt."""
        return f"""You are a CryoEM processing assistant using the ReAct (Reasoning + Acting) framework. 
You must follow a structured approach of Reasoning, Acting, and Observing.

## ReAct Framework Rules:
1. **REASONING**: Always think through the problem step by step before taking action
2. **ACTING**: Execute specific tools based on your reasoning
3. **OBSERVING**: Analyze the results and update your understanding

## Workflow Steps (in order):
{self._format_workflow_steps()}

## ReAct Process:
For each step, you MUST follow this pattern:

**Thought**: [Your reasoning about what needs to be done, why, and how]
**Action**: [The specific tool to use with exact parameters]
**Observation**: [What happened as a result of the action]

## CRITICAL: Job Monitoring and Waiting
- After starting any job (import_movies, motion_correction, ctf_estimation), you MUST wait for it to complete
- Use wait_for_job tool with the job UID to wait for completion
- Do NOT proceed to the next step until the current job is completed
- If a job fails, report the error and stop the workflow

## Tool Usage Guidelines:
- import_movies: Start the import, then wait for completion
- motion_correction: Requires movies_job_uid from completed import_movies job
- ctf_estimation: Requires micrographs_job_uid from completed motion_correction job
- get_job_status: Check status of a specific job (use job UID only, e.g., "J81")
- wait_for_job: Wait for job completion (use job UID only, e.g., "J81")
- reason_about_workflow: Analyze current state and dependencies

## Job UID Format:
- Job UIDs are strings like "J81", "J82", etc.
- When calling get_job_status or wait_for_job, pass ONLY the job UID (e.g., "J81")
- Do NOT use JSON format or complex parameters for these tools

## Workflow Dependencies:
1. Import movies → Wait for completion → Motion correction → Wait for completion → CTF estimation
2. Each step must complete successfully before the next can begin
3. Always verify job completion before proceeding

## Current Configuration:
- Project UID: {self.config.workflow.project_uid}
- Workspace UID: {self.config.workflow.workspace_uid}
- Movies Path: {self.config.workflow.movies_path}
- Pixel Size: {self.config.workflow.pixel_size}
- Voltage: {self.config.workflow.voltage}

## Example Workflow:
1. **Thought**: I need to start by importing movies
2. **Action**: import_movies
3. **Observation**: Job J81 started
4. **Thought**: Now I need to wait for the import job to complete
5. **Action**: wait_for_job with "J81"
6. **Observation**: Job J81 completed successfully
7. **Thought**: Now I can start motion correction using the movies from J81
8. **Action**: motion_correction with movies_job_uid=J81
9. **Observation**: Job J82 started
10. **Thought**: I need to wait for motion correction to complete
11. **Action**: wait_for_job with "J82"
12. **Observation**: Job J82 completed successfully
13. Continue with CTF estimation...

Remember: Always follow the Thought → Action → Observation pattern and WAIT for each job to complete!"""
    
    def _format_workflow_steps(self) -> str:
        """Format workflow steps for the prompt."""
        steps_text = ""
        for i, step in enumerate(self.config.react_workflow.steps, 1):
            steps_text += f"{i}. **{step.name}**: {step.description}\n"
            if step.required_params:
                steps_text += f"   Required: {', '.join(step.required_params)}\n"
            if step.optional_params:
                steps_text += f"   Optional: {', '.join(step.optional_params)}\n"
            if step.depends_on:
                steps_text += f"   Depends on: {', '.join(step.depends_on)}\n"
            steps_text += "\n"
        return steps_text
    
    def _import_movies_tool(self, input_str: str) -> str:
        """Tool wrapper for importing movies."""
        try:
            params = self._parse_tool_input(input_str)
            
            # Use config defaults if not provided
            result = self.cryosparc_tools.import_movies(
                project_uid=params.get("project_uid", self.config.workflow.project_uid),
                workspace_uid=params.get("workspace_uid", self.config.workflow.workspace_uid),
                movies_path=params.get("movies_path", self.config.workflow.movies_path),
                gain_ref_path=params.get("gain_ref_path", self.config.workflow.gain_ref_path),
                pixel_size=float(params.get("pixel_size", self.config.workflow.pixel_size)),
                voltage=float(params.get("voltage", self.config.workflow.voltage)),
                cs_mm=float(params.get("cs_mm", self.config.workflow.cs_mm)),
                dose=float(params.get("dose", self.config.workflow.dose))
            )
            
            return f"✅ Successfully queued import movies job: {result['job_uid']}"
            
        except Exception as e:
            return f"❌ Error importing movies: {str(e)}"
    
    def _motion_correction_tool(self, input_str: str) -> str:
        """Tool wrapper for motion correction."""
        try:
            params = self._parse_tool_input(input_str)
            
            result = self.cryosparc_tools.motion_correction(
                project_uid=params.get("project_uid", self.config.workflow.project_uid),
                workspace_uid=params.get("workspace_uid", self.config.workflow.workspace_uid),
                movies_job_uid=params.get("movies_job_uid"),
                binning=int(params.get("binning", self.config.workflow.motion_correction_binning)),
                patch_size=int(params.get("patch_size", self.config.workflow.motion_correction_patch_size))
            )
            
            return f"✅ Successfully queued motion correction job: {result['job_uid']}"
            
        except Exception as e:
            return f"❌ Error starting motion correction: {str(e)}"
    
    def _ctf_estimation_tool(self, input_str: str) -> str:
        """Tool wrapper for CTF estimation."""
        try:
            params = self._parse_tool_input(input_str)
            
            result = self.cryosparc_tools.ctf_estimation(
                project_uid=params.get("project_uid", self.config.workflow.project_uid),
                workspace_uid=params.get("workspace_uid", self.config.workflow.workspace_uid),
                micrographs_job_uid=params.get("micrographs_job_uid"),
                min_res=float(params.get("min_res", self.config.workflow.ctf_min_res)),
                max_res=float(params.get("max_res", self.config.workflow.ctf_max_res))
            )
            
            return f"✅ Successfully queued CTF estimation job: {result['job_uid']}"
            
        except Exception as e:
            return f"❌ Error starting CTF estimation: {str(e)}"
    
    def _get_job_status_tool(self, input_str: str) -> str:
        """Tool wrapper for getting job status."""
        try:
            params = self._parse_tool_input(input_str)
            job_uid = params.get("job_uid")
            
            if not job_uid:
                return f"❌ Error: job_uid parameter is required. Input was: '{input_str}', parsed params: {params}"
            
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)

            status = self.cryosparc_tools.get_job_status(
                job_uid,
                project_uid=project_uid,
                workspace_uid=workspace_uid
            )
            progress_display = f"{status['progress']}%" if status.get("progress") is not None else "N/A"
            return f"📊 Job {job_uid} status: {status['status']} ({progress_display})"
            
        except Exception as e:
            return f"❌ Error getting job status: {str(e)}"
    
    def _wait_for_job_tool(self, input_str: str) -> str:
        """Tool wrapper for waiting for job completion."""
        try:
            params = self._parse_tool_input(input_str)
            job_uid = params.get("job_uid")
            timeout = int(params.get("timeout", self.config.job_management.default_timeout))
            
            if not job_uid:
                return f"❌ Error: job_uid parameter is required. Input was: '{input_str}', parsed params: {params}"
            
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)

            print(f"🛰️ Waiting for job {job_uid} to complete...")
            status = self.cryosparc_tools.wait_for_job_completion(
                project_uid,
                job_uid,
                workspace_uid,
                timeout
            )
            return f"✅ Job {job_uid} completed with status: {status['status']}"
            
        except Exception as e:
            return f"❌ Error waiting for job: {str(e)}"
    
    def _reason_about_workflow_tool(self, input_str: str) -> str:
        """Tool for reasoning about workflow state and dependencies."""
        try:
            # This tool helps the agent think through workflow dependencies
            reasoning = f"""
🤔 **Workflow Analysis**:

**Current State**: {input_str}

**Workflow Dependencies**:
1. Import Movies → Motion Correction → CTF Estimation
2. Each step must complete before the next can begin

**Next Steps Analysis**:
- If no jobs are running: Start with import_movies
- If import job is running: Wait for completion, then start motion_correction
- If motion correction is running: Wait for completion, then start ctf_estimation
- If CTF estimation is running: Wait for completion, workflow is done

**Recommended Actions**:
- Always check job status before proceeding
- Use wait_for_job for critical dependencies
- Verify each step completes successfully before moving to the next
"""
            return reasoning
            
        except Exception as e:
            return f"❌ Error in workflow reasoning: {str(e)}"
    
    def _parse_tool_input(self, input_str: str) -> Dict[str, Any]:
        """Parse tool input string into parameters."""
        import json
        
        # Handle different input formats
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
        
        # Case 3: Comma-separated key=value pairs
        params = {}
        for pair in input_str.split(","):
            if "=" in pair:
                key, value = pair.strip().split("=", 1)
                params[key.strip()] = value.strip()
        
        # Case 4: If no parameters found and input looks like a single value, treat as job_uid
        if not params and input_str:
            # Check if it looks like a job UID or other single parameter
            if input_str.startswith("J") or input_str.isdigit():
                params["job_uid"] = input_str
            else:
                # For other tools, try to extract meaningful parameters
                params["input"] = input_str
        
        return params
    
    def run_react_workflow(self, workflow_input: str) -> str:
        """
        Run a cryoEM processing workflow using ReAct approach.
        
        Args:
            workflow_input: Description of the workflow to run
            
        Returns:
            Result of the workflow execution
        """
        try:
            # Add ReAct-specific instructions to the input
            react_input = f"""
Execute the following cryoEM workflow using the ReAct framework:

{workflow_input}

Remember to:
1. Start with reasoning about the workflow state
2. Follow the Thought → Action → Observation pattern
3. Check dependencies before each step
4. Wait for jobs to complete before proceeding
5. Provide clear status updates

Begin with reasoning about the current workflow state.
"""
            
            result = self.agent_executor.invoke({"input": react_input})
            return result["output"]
            
        except Exception as e:
            return f"❌ ReAct workflow execution failed: {str(e)}"
    
    def run_single_step(self, step_description: str) -> str:
        """
        Run a single processing step using ReAct approach.
        
        Args:
            step_description: Description of the step to run
            
        Returns:
            Result of the step execution
        """
        return self.run_react_workflow(step_description)
    
    def get_reasoning_history(self) -> List[Dict[str, str]]:
        """Get the history of reasoning steps."""
        return self.reasoning_history.copy()
    
    def clear_reasoning_history(self):
        """Clear the reasoning history."""
        self.reasoning_history = []
