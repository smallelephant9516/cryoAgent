"""Conversation logging utility for capturing LLM interactions."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class ConversationEntry:
    """Single entry in a conversation log."""
    timestamp: str
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ConversationLog:
    """Complete conversation log for a workflow execution."""
    conversation_id: str
    workflow_type: str
    stage_name: str
    start_time: str
    end_time: Optional[str] = None
    entries: List[ConversationEntry] = None
    summary: Optional[str] = None
    success: Optional[bool] = None
    
    def __post_init__(self):
        if self.entries is None:
            self.entries = []


class ConversationLogger:
    """Logger for capturing and saving LLM conversations."""
    
    def __init__(self, outputs_dir: str = "outputs"):
        """
        Initialize the conversation logger.
        
        Args:
            outputs_dir: Directory to save conversation logs
        """
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(exist_ok=True)
        self.current_log: Optional[ConversationLog] = None
    
    def start_conversation(
        self, 
        conversation_id: str, 
        workflow_type: str, 
        stage_name: str
    ) -> ConversationLog:
        """
        Start a new conversation log.
        
        Args:
            conversation_id: Unique identifier for the conversation
            workflow_type: Type of workflow being executed
            stage_name: Name of the current stage
            
        Returns:
            ConversationLog object
        """
        self.current_log = ConversationLog(
            conversation_id=conversation_id,
            workflow_type=workflow_type,
            stage_name=stage_name,
            start_time=datetime.now().isoformat(),
            entries=[]
        )
        return self.current_log
    
    def add_entry(
        self,
        role: str,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add an entry to the current conversation log.
        
        Args:
            role: Role of the speaker (user, assistant, system, tool)
            content: Content of the message
            tool_calls: Tool calls made by the assistant
            tool_results: Results from tool executions
            metadata: Additional metadata
        """
        if not self.current_log:
            raise ValueError("No active conversation. Call start_conversation() first.")
        
        entry = ConversationEntry(
            timestamp=datetime.now().isoformat(),
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_results=tool_results,
            metadata=metadata
        )
        
        self.current_log.entries.append(entry)
    
    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a user message to the conversation."""
        self.add_entry("user", content, metadata=metadata)
    
    def add_assistant_message(
        self, 
        content: str, 
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add an assistant message to the conversation."""
        self.add_entry("assistant", content, tool_calls=tool_calls, metadata=metadata)
    
    def add_system_message(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a system message to the conversation."""
        self.add_entry("system", content, metadata=metadata)
    
    def add_tool_result(
        self, 
        tool_name: str, 
        result: str, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a tool execution result to the conversation."""
        self.add_entry(
            "tool", 
            f"Tool {tool_name} executed", 
            tool_results={tool_name: result},
            metadata=metadata
        )
    
    def end_conversation(self, success: bool, summary: Optional[str] = None):
        """
        End the current conversation log.
        
        Args:
            success: Whether the conversation was successful
            summary: Optional summary of the conversation
        """
        if not self.current_log:
            raise ValueError("No active conversation to end.")
        
        self.current_log.end_time = datetime.now().isoformat()
        self.current_log.success = success
        self.current_log.summary = summary
    
    def save_conversation_log(self, filename: Optional[str] = None) -> str:
        """
        Save the current conversation log to a file.
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path to the saved log file
        """
        if not self.current_log:
            raise ValueError("No conversation log to save.")
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"llm_conversation_{self.current_log.stage_name}_{timestamp}.json"
        
        log_file = self.outputs_dir / filename
        
        # Convert to dictionary for JSON serialization
        log_dict = asdict(self.current_log)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_dict, f, indent=2, ensure_ascii=False)
        
        return str(log_file)
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation."""
        if not self.current_log:
            return {"error": "No active conversation"}
        
        return {
            "conversation_id": self.current_log.conversation_id,
            "workflow_type": self.current_log.workflow_type,
            "stage_name": self.current_log.stage_name,
            "start_time": self.current_log.start_time,
            "end_time": self.current_log.end_time,
            "total_entries": len(self.current_log.entries),
            "success": self.current_log.success,
            "summary": self.current_log.summary
        }
    
    def clear_current_log(self):
        """Clear the current conversation log."""
        self.current_log = None
