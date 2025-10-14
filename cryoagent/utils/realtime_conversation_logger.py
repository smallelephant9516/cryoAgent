"""Real-time conversation logging utility for capturing LLM interactions as they happen."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


class RealtimeConversationLogger:
    """Real-time logger for capturing LLM conversations and writing them immediately to log files."""
    
    def __init__(self, outputs_dir: str = "outputs"):
        """
        Initialize the real-time conversation logger.
        
        Args:
            outputs_dir: Directory to save conversation logs
        """
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(exist_ok=True)
        self.current_log_file: Optional[str] = None
        self.conversation_id: Optional[str] = None
        self.stage_name: Optional[str] = None
        self.workflow_type: Optional[str] = None
        self.start_time: Optional[str] = None
    
    def start_conversation(
        self, 
        conversation_id: str, 
        workflow_type: str, 
        stage_name: str
    ) -> str:
        """
        Start a new conversation log file.
        
        Args:
            conversation_id: Unique identifier for the conversation
            workflow_type: Type of workflow being executed
            stage_name: Name of the current stage
            
        Returns:
            Path to the log file
        """
        self.conversation_id = conversation_id
        self.workflow_type = workflow_type
        self.stage_name = stage_name
        self.start_time = datetime.now().isoformat()
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"llm_conversation_{stage_name}_{timestamp}.log"
        self.current_log_file = str(self.outputs_dir / log_filename)
        
        # Write initial header to log file
        with open(self.current_log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== LLM Conversation Log ===\n")
            f.write(f"Conversation ID: {conversation_id}\n")
            f.write(f"Workflow Type: {workflow_type}\n")
            f.write(f"Stage: {stage_name}\n")
            f.write(f"Start Time: {self.start_time}\n")
            f.write("=" * 50 + "\n\n")
        
        return self.current_log_file
    
    def log_user_input(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Log user input to the conversation file."""
        if not self.current_log_file:
            return
        
        timestamp = datetime.now().isoformat()
        with open(self.current_log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] USER INPUT:\n")
            f.write(f"{content}\n")
            if metadata:
                f.write(f"Metadata: {json.dumps(metadata, indent=2)}\n")
            f.write("-" * 50 + "\n\n")
    
    def log_assistant_response(
        self, 
        content: str, 
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log assistant response to the conversation file."""
        if not self.current_log_file:
            return
        
        timestamp = datetime.now().isoformat()
        with open(self.current_log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] ASSISTANT RESPONSE:\n")
            f.write(f"{content}\n")
            
            if tool_calls:
                f.write(f"\nTool Calls:\n")
                for i, tool_call in enumerate(tool_calls, 1):
                    f.write(f"  {i}. {tool_call.get('name', 'Unknown')}\n")
                    f.write(f"     Arguments: {json.dumps(tool_call.get('arguments', {}), indent=6)}\n")
            
            if metadata:
                f.write(f"\nMetadata: {json.dumps(metadata, indent=2)}\n")
            
            f.write("-" * 50 + "\n\n")
    
    def log_tool_execution(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any], 
        result: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log tool execution to the conversation file."""
        if not self.current_log_file:
            return
        
        timestamp = datetime.now().isoformat()
        with open(self.current_log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] TOOL EXECUTION: {tool_name}\n")
            f.write(f"Arguments: {json.dumps(arguments, indent=2)}\n")
            f.write(f"Result: {result}\n")
            
            if metadata:
                f.write(f"Metadata: {json.dumps(metadata, indent=2)}\n")
            
            f.write("-" * 50 + "\n\n")
    
    def log_system_message(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Log system message to the conversation file."""
        if not self.current_log_file:
            return
        
        timestamp = datetime.now().isoformat()
        with open(self.current_log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] SYSTEM MESSAGE:\n")
            f.write(f"{content}\n")
            if metadata:
                f.write(f"Metadata: {json.dumps(metadata, indent=2)}\n")
            f.write("-" * 50 + "\n\n")
    
    def log_error(self, error: str, context: Optional[str] = None):
        """Log error to the conversation file."""
        if not self.current_log_file:
            return
        
        timestamp = datetime.now().isoformat()
        with open(self.current_log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] ERROR:\n")
            f.write(f"{error}\n")
            if context:
                f.write(f"Context: {context}\n")
            f.write("-" * 50 + "\n\n")
    
    def end_conversation(self, success: bool, summary: Optional[str] = None):
        """End the conversation and write final summary."""
        if not self.current_log_file:
            return
        
        end_time = datetime.now().isoformat()
        with open(self.current_log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{end_time}] CONVERSATION ENDED:\n")
            f.write(f"Success: {success}\n")
            f.write(f"End Time: {end_time}\n")
            if summary:
                f.write(f"Summary: {summary}\n")
            f.write("=" * 50 + "\n")
    
    def get_log_file_path(self) -> Optional[str]:
        """Get the path to the current log file."""
        return self.current_log_file
    
    def clear_current_log(self):
        """Clear the current conversation log."""
        self.current_log_file = None
        self.conversation_id = None
        self.stage_name = None
        self.workflow_type = None
        self.start_time = None
