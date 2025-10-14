"""General LLM conversation logger for capturing all LLM responses during workflow execution."""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional


class GeneralLLMLogger:
    """Simple logger for capturing all LLM responses during workflow execution."""
    
    def __init__(self, outputs_dir: str = "outputs"):
        """
        Initialize the general LLM logger.
        
        Args:
            outputs_dir: Directory to save the log file
        """
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(exist_ok=True)
        self.log_file: Optional[str] = None
        self.workflow_id: Optional[str] = None
    
    def start_workflow_log(self, workflow_id: Optional[str] = None) -> str:
        """
        Start logging for a workflow execution.
        
        Args:
            workflow_id: Optional workflow identifier
            
        Returns:
            Path to the log file
        """
        if not workflow_id:
            workflow_id = f"workflow_{int(time.time())}"
        
        self.workflow_id = workflow_id
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"llm_conversation_{timestamp}.log"
        self.log_file = str(self.outputs_dir / log_filename)
        
        # Write initial header
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== LLM Conversation Log ===\n")
            f.write(f"Workflow ID: {workflow_id}\n")
            f.write(f"Start Time: {datetime.now().isoformat()}\n")
            f.write("=" * 50 + "\n\n")
        
        return self.log_file
    
    def log_llm_response(self, stage: str, response: str, metadata: Optional[dict] = None):
        """
        Log an LLM response to the conversation file.
        
        Args:
            stage: The workflow stage (e.g., 'preprocessing', 'particle_picking', 'reconstruction')
            response: The LLM response text
            metadata: Optional metadata about the response
        """
        if not self.log_file:
            return
        
        timestamp = datetime.now().isoformat()
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {stage.upper()} - LLM RESPONSE:\n")
            f.write(f"{response}\n")
            if metadata:
                f.write(f"Metadata: {metadata}\n")
            f.write("-" * 50 + "\n\n")
    
    def log_llm_reasoning(self, stage: str, reasoning: str):
        """
        Log LLM reasoning/thought process.
        
        Args:
            stage: The workflow stage
            reasoning: The reasoning text
        """
        if not self.log_file:
            return
        
        timestamp = datetime.now().isoformat()
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {stage.upper()} - LLM REASONING:\n")
            f.write(f"{reasoning}\n")
            f.write("-" * 30 + "\n\n")
    
    def log_llm_action(self, stage: str, action: str, tool_calls: Optional[list] = None):
        """
        Log LLM actions and tool calls.
        
        Args:
            stage: The workflow stage
            action: The action description
            tool_calls: Optional list of tool calls
        """
        if not self.log_file:
            return
        
        timestamp = datetime.now().isoformat()
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {stage.upper()} - LLM ACTION:\n")
            f.write(f"{action}\n")
            if tool_calls:
                f.write(f"Tool Calls: {len(tool_calls)} tools\n")
                for i, tool in enumerate(tool_calls, 1):
                    f.write(f"  {i}. {tool.get('name', 'Unknown')}\n")
            f.write("-" * 30 + "\n\n")
    
    def log_llm_observation(self, stage: str, observation: str):
        """
        Log LLM observations from tool results.
        
        Args:
            stage: The workflow stage
            observation: The observation text
        """
        if not self.log_file:
            return
        
        timestamp = datetime.now().isoformat()
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {stage.upper()} - LLM OBSERVATION:\n")
            f.write(f"{observation}\n")
            f.write("-" * 30 + "\n\n")
    
    def end_workflow_log(self, success: bool, summary: Optional[str] = None):
        """End the workflow log with final summary."""
        if not self.log_file:
            return
        
        end_time = datetime.now().isoformat()
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{end_time}] WORKFLOW ENDED:\n")
            f.write(f"Success: {success}\n")
            f.write(f"End Time: {end_time}\n")
            if summary:
                f.write(f"Summary: {summary}\n")
            f.write("=" * 50 + "\n")
    
    def get_log_file_path(self) -> Optional[str]:
        """Get the path to the current log file."""
        return self.log_file
