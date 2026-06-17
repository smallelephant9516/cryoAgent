"""Utility for parsing conversation log files and resuming from the last tool execution."""

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime


class LogResumeParser:
    """Parser for conversation log files to extract resume information."""
    
    def __init__(self, outputs_dir: str = "outputs"):
        """
        Initialize the log resume parser.
        
        Args:
            outputs_dir: Directory where log files are stored
        """
        self.outputs_dir = Path(outputs_dir)
    
    def find_log_file(self, stage_name: str, conversation_id: Optional[str] = None) -> Optional[str]:
        """
        Find the most recent log file for a given stage.
        
        Args:
            stage_name: Name of the workflow stage
            conversation_id: Optional conversation ID to match
            
        Returns:
            Path to the log file if found, None otherwise
        """
        # Pattern: llm_conversation_{stage_name}_{timestamp}.log
        # NOTE: a plain glob of "llm_conversation_{stage}_*.log" would also match
        # longer stage names that share a prefix (e.g. stage "optimization" would
        # wrongly match "optimization_2d" logs). The token after the stage name is
        # always the timestamp (YYYYMMDD_HHMMSS), so require it to start with digits.
        import re
        pattern = f"llm_conversation_{stage_name}_*.log"
        exact_re = re.compile(
            r"^llm_conversation_" + re.escape(stage_name) + r"_\d{8}_\d{6}\.log$"
        )
        matching_files = [
            f for f in self.outputs_dir.glob(pattern)
            if exact_re.match(f.name)
        ]

        if not matching_files:
            return None
        
        # Sort by modification time (most recent first)
        matching_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # If conversation_id is provided, try to match it
        if conversation_id:
            for log_file in matching_files:
                if self._log_file_matches_conversation_id(log_file, conversation_id):
                    return str(log_file)
        
        # Return the most recent file
        return str(matching_files[0])
    
    def _log_file_matches_conversation_id(self, log_file: Path, conversation_id: str) -> bool:
        """Check if a log file matches the given conversation ID."""
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Check if conversation ID appears in the header
                if f"Conversation ID: {conversation_id}" in content:
                    return True
        except Exception:
            pass
        return False
    
    def parse_log_file(self, log_file_path: str) -> Dict[str, Any]:
        """
        Parse a conversation log file and extract resume information.
        
        Args:
            log_file_path: Path to the log file
            
        Returns:
            Dictionary with parsed log information including:
            - conversation_id: Conversation ID from the log
            - stage_name: Stage name from the log
            - last_tool_execution: Last tool execution details (if any)
            - all_tool_executions: List of all tool executions
            - user_input: Original user input
            - workflow_input: Workflow input from system message
        """
        result = {
            "conversation_id": None,
            "stage_name": None,
            "workflow_type": None,
            "start_time": None,
            "last_tool_execution": None,
            "all_tool_executions": [],
            "user_input": None,
            "workflow_input": None,
            "completed": False
        }
        
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse header
            header_match = re.search(r'Conversation ID: (.+)', content)
            if header_match:
                result["conversation_id"] = header_match.group(1).strip()
            
            stage_match = re.search(r'Stage: (.+)', content)
            if stage_match:
                result["stage_name"] = stage_match.group(1).strip()
            
            workflow_match = re.search(r'Workflow Type: (.+)', content)
            if workflow_match:
                result["workflow_type"] = workflow_match.group(1).strip()
            
            start_time_match = re.search(r'Start Time: (.+)', content)
            if start_time_match:
                result["start_time"] = start_time_match.group(1).strip()
            
            # Check if conversation ended
            if "CONVERSATION ENDED:" in content or "WORKFLOW ENDED:" in content:
                result["completed"] = True
            
            # Parse tool executions - handle multi-line results and missing results
            # Pattern: [timestamp] TOOL EXECUTION: tool_name\nArguments: {...}\nResult: ...\n---
            # Split by tool execution blocks
            tool_blocks = re.split(r'\[([^\]]+)\] TOOL EXECUTION: (.+?)\n', content)
            
            # Process blocks in pairs: timestamp, tool_name, content
            for i in range(1, len(tool_blocks) - 1, 3):
                if i + 2 >= len(tool_blocks):
                    break
                    
                timestamp = tool_blocks[i].strip()
                tool_name = tool_blocks[i + 1].strip()
                block_content = tool_blocks[i + 2] if i + 2 < len(tool_blocks) else ""
                
                # Extract arguments and result from block content
                arguments_str = "{}"
                result_str = "No result"
                
                # Look for Arguments: section
                args_match = re.search(r'Arguments:\s*(.+?)(?=\nResult:|\n-{50}|\Z)', block_content, re.DOTALL)
                if args_match:
                    arguments_str = args_match.group(1).strip()
                
                # Look for Result: section
                result_match = re.search(r'Result:\s*(.+?)(?=\n\[|\n-{50}|\Z)', block_content, re.DOTALL)
                if result_match:
                    result_str = result_match.group(1).strip()
                
                # Parse arguments JSON - try to extract JSON from the string
                arguments = {}
                try:
                    # Try to find JSON object in arguments string
                    json_match = re.search(r'\{.*\}', arguments_str, re.DOTALL)
                    if json_match:
                        arguments = json.loads(json_match.group(0))
                    else:
                        arguments = json.loads(arguments_str)
                except (json.JSONDecodeError, AttributeError):
                    # If JSON parsing fails, try to extract key-value pairs
                    try:
                        # Try to parse as JSON one more time with the full string
                        arguments = json.loads(arguments_str)
                    except json.JSONDecodeError:
                        arguments = {"raw": arguments_str}
                
                # Parse result (might be JSON, Python dict string, or plain text)
                result_data = None
                try:
                    # Try to find JSON/dict object in result string
                    json_match = re.search(r'\{.*\}', result_str, re.DOTALL)
                    if json_match:
                        dict_str = json_match.group(0)
                        # First try JSON (double quotes)
                        try:
                            result_data = json.loads(dict_str)
                        except json.JSONDecodeError:
                            # Try Python dict (single quotes) using ast.literal_eval
                            try:
                                import ast
                                result_data = ast.literal_eval(dict_str)
                            except (ValueError, SyntaxError):
                                # If both fail, try converting single quotes to double quotes
                                try:
                                    dict_str_fixed = dict_str.replace("'", '"')
                                    result_data = json.loads(dict_str_fixed)
                                except json.JSONDecodeError:
                                    result_data = result_str[:1000] if len(result_str) > 1000 else result_str
                    else:
                        # Try parsing the whole string as JSON
                        try:
                            result_data = json.loads(result_str)
                        except json.JSONDecodeError:
                            # Try Python dict syntax
                            try:
                                import ast
                                result_data = ast.literal_eval(result_str)
                            except (ValueError, SyntaxError):
                                result_data = result_str[:1000] if len(result_str) > 1000 else result_str
                except (json.JSONDecodeError, AttributeError, ValueError, SyntaxError):
                    # If it's not JSON or Python dict, keep as string (truncate if too long)
                    result_data = result_str[:1000] if len(result_str) > 1000 else result_str
                
                tool_exec = {
                    "timestamp": timestamp,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": result_data
                }
                
                result["all_tool_executions"].append(tool_exec)
            
            # Get the last tool execution
            if result["all_tool_executions"]:
                result["last_tool_execution"] = result["all_tool_executions"][-1]
            
            # Parse user input
            user_input_match = re.search(r'\[([^\]]+)\] USER INPUT:\n(.+?)(?=\n\[|\n-{50}|\Z)', content, re.DOTALL)
            if user_input_match:
                result["user_input"] = user_input_match.group(2).strip()
            
            # Parse workflow input from system message
            system_message_match = re.search(r'\[([^\]]+)\] SYSTEM MESSAGE:\n(.+?)(?=\n\[|\n-{50}|\Z)', content, re.DOTALL)
            if system_message_match:
                system_content = system_message_match.group(2).strip()
                # Try to extract workflow_input from metadata
                metadata_match = re.search(r'Metadata: \{.*?"workflow_input":\s*"(.+?)"', system_content, re.DOTALL)
                if metadata_match:
                    result["workflow_input"] = metadata_match.group(1).strip()
                else:
                    result["workflow_input"] = system_content
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def should_resume(self, log_file_path: str) -> bool:
        """
        Determine if we should resume from a log file.
        
        Args:
            log_file_path: Path to the log file
            
        Returns:
            True if we should resume (log exists and is not completed), False otherwise
        """
        parsed = self.parse_log_file(log_file_path)
        
        # Don't resume if conversation is already completed
        if parsed.get("completed", False):
            return False
        
        # Resume if there are tool executions
        if parsed.get("all_tool_executions"):
            return True
        
        return False
    
    def get_resume_context(self, log_file_path: str) -> Dict[str, Any]:
        """
        Get context information for resuming from a log file.
        
        Args:
            log_file_path: Path to the log file
            
        Returns:
            Dictionary with resume context including:
            - last_tool_name: Name of the last tool executed
            - last_tool_arguments: Arguments of the last tool
            - last_tool_result: Result of the last tool
            - conversation_history: Summary of what was done
            - completed_work: Summary of completed work (for heterogeneity: completed K values)
        """
        parsed = self.parse_log_file(log_file_path)
        
        context = {
            "log_file": log_file_path,
            "conversation_id": parsed.get("conversation_id"),
            "stage_name": parsed.get("stage_name"),
            "workflow_input": parsed.get("workflow_input"),
            "user_input": parsed.get("user_input"),
            "last_tool_execution": parsed.get("last_tool_execution"),
            "all_tool_executions": parsed.get("all_tool_executions", []),
            "num_tools_executed": len(parsed.get("all_tool_executions", [])),
            "resume_message": None,
            "completed_work": None,
            "work_summary": None
        }
        
        # Analyze tool executions to create a work summary
        # For heterogeneity analysis, track completed K values
        if parsed.get("stage_name") == "heterogeneity":
            context["completed_work"] = self._analyze_heterogeneity_progress(parsed.get("all_tool_executions", []))
        
        # For heterogeneity_depth analysis, track completed clusters and branches
        elif parsed.get("stage_name") == "heterogeneity_depth":
            context["completed_work"] = self._analyze_heterogeneity_depth_progress(parsed.get("all_tool_executions", []))
        
        # For optimization stage, track completed multi-round 3D classification, heterogeneous refinement, and box size optimization
        elif parsed.get("stage_name") == "optimization":
            context["completed_work"] = self._analyze_optimization_progress(parsed.get("all_tool_executions", []))
        
        # Create a resume message
        if context["last_tool_execution"]:
            last_tool = context["last_tool_execution"]
            resume_msg = (
                f"Resuming from previous execution. Last tool executed: {last_tool['tool_name']} "
                f"at {last_tool['timestamp']}. Total tools executed: {context['num_tools_executed']}."
            )
            
            # Add work summary if available
            if context.get("completed_work"):
                resume_msg += f"\n\n{context['completed_work']}"
            
            context["resume_message"] = resume_msg
        else:
            context["resume_message"] = (
                f"Resuming from previous execution. No tools executed yet. "
                f"Starting fresh with workflow input."
            )
        
        return context
    
    def _analyze_heterogeneity_progress(self, tool_executions: List[Dict[str, Any]]) -> str:
        """
        Analyze heterogeneity analysis progress from tool executions.
        
        Args:
            tool_executions: List of tool execution dictionaries
            
        Returns:
            String summary of completed work
        """
        completed_k_values = set()
        in_progress_k_values = set()
        hetero_jobs = {}  # k -> hetero_job_uid
        ab_initio_jobs = {}  # k -> ab_initio_job_uid
        k_has_density_extraction = set()  # K values that have density maps extracted
        k_has_comparison = set()  # K values that have been compared
        
        # Track tool executions by K value
        # Process in reverse order to get the most recent successful results
        for tool_exec in reversed(tool_executions):
            tool_name = tool_exec.get("tool_name", "")
            args = tool_exec.get("arguments", {})
            result = tool_exec.get("result")
            
            # Parse result - it might be a string (JSON) or a dict
            result_dict = result
            if isinstance(result, str):
                # Try to parse as JSON
                if result.strip().startswith("{") or result.strip().startswith("'"):
                    try:
                        import json
                        # Remove single quotes if present and try to parse
                        result_str = result.strip().strip("'\"")
                        result_dict = json.loads(result_str)
                    except (json.JSONDecodeError, ValueError):
                        result_dict = None
                elif result == "No result" or "No result" in result:
                    result_dict = None
                else:
                    result_dict = None
            
            # Check if result is successful
            is_successful = isinstance(result_dict, dict) and result_dict.get("success")
            is_no_result = (result == "No result" or 
                          (isinstance(result, str) and "No result" in str(result)) or
                          result is None or result_dict is None)
            
            # Track ab_initio_reconstruction by num_classes (K value)
            # Only store if we haven't seen a successful result for this K yet (process in reverse, so first = most recent)
            if tool_name == "ab_initio_reconstruction":
                k = args.get("num_classes")
                if k and k not in ab_initio_jobs:
                    # Check if result indicates completion
                    if is_successful and result_dict.get("job_uid"):
                        ab_initio_jobs[k] = result_dict.get("job_uid")
                    elif is_no_result:
                        # Mark as in progress if we haven't seen a successful result for this K
                        in_progress_k_values.add(k)
            
            # Track heterogeneous_refinement
            # Only store if we haven't seen a successful result for this K yet
            elif tool_name == "heterogeneous_refinement":
                # Try to find which K this belongs to by looking at volume_groups
                volume_groups = args.get("volume_groups", [])
                if volume_groups:
                    k = len(volume_groups)
                    if k not in hetero_jobs and is_successful and result_dict.get("job_uid"):
                        hetero_jobs[k] = result_dict.get("job_uid")
            
            # Track extract_density_maps - indicates K value analysis is in progress
            elif tool_name == "extract_density_maps":
                hetero_job_uid = args.get("hetero_job_uid")
                # Also check result for hetero_job_uid (parse if string)
                result_dict_extract = result
                if isinstance(result, str) and result.strip().startswith("{"):
                    try:
                        import json
                        result_dict_extract = json.loads(result.strip().strip("'\""))
                    except (json.JSONDecodeError, ValueError):
                        result_dict_extract = None
                
                if isinstance(result_dict_extract, dict):
                    hetero_job_uid = hetero_job_uid or result_dict_extract.get("hetero_job_uid")
                
                if hetero_job_uid and isinstance(result_dict_extract, dict) and result_dict_extract.get("success"):
                    # Extract just the job number (e.g., "J78" from "J78" or "/path/to/J78/...")
                    job_num = str(hetero_job_uid).replace("J", "").split("/")[0]
                    if job_num.isdigit():
                        job_uid_str = f"J{job_num}"
                    else:
                        job_uid_str = str(hetero_job_uid)
                    
                    # Find which K this belongs to by matching hetero_job_uid
                    for k, job_uid in hetero_jobs.items():
                        job_uid_str_stored = str(job_uid).replace("J", "").split("/")[0]
                        if job_uid_str_stored.isdigit():
                            job_uid_stored = f"J{job_uid_str_stored}"
                        else:
                            job_uid_stored = str(job_uid)
                        
                        if job_uid_str == job_uid_stored or job_uid_str in str(job_uid) or str(job_uid) in job_uid_str:
                            k_has_density_extraction.add(k)
                            break
            
            # Track compare_all_densities - indicates K value analysis is complete
            elif tool_name == "compare_all_densities":
                folder = args.get("folder", "")
                # Find which K this belongs to by matching folder path with hetero job UIDs
                for k, job_uid in hetero_jobs.items():
                    job_uid_str = str(job_uid)
                    # Extract job number from folder path (e.g., "/path/to/J78/..." -> "J78")
                    if job_uid_str in folder or f"/{job_uid_str}/" in folder:
                        k_has_comparison.add(k)
                        break
        
        # Determine completed K values
        # A K value is complete if it has:
        # 1. Successful ab_initio job
        # 2. Successful hetero job
        # 3. Density maps extracted
        # 4. Comparison done (optional but indicates full analysis)
        for k in set(list(ab_initio_jobs.keys()) + list(hetero_jobs.keys())):
            has_ab_initio = k in ab_initio_jobs
            has_hetero = k in hetero_jobs
            has_extraction = k in k_has_density_extraction
            
            # K is completed if it has ab_initio, hetero, and extraction
            if has_ab_initio and has_hetero and has_extraction:
                completed_k_values.add(k)
                # Remove from in_progress if it was there
                in_progress_k_values.discard(k)
            elif has_ab_initio and not has_hetero:
                # Ab initio done but hetero not started
                if k not in completed_k_values:
                    in_progress_k_values.add(k)
        
        # Build summary message
        summary_parts = []
        
        if completed_k_values:
            summary_parts.append(f"✅ Completed K values: {sorted(completed_k_values)}")
            for k in sorted(completed_k_values):
                details = []
                if k in ab_initio_jobs:
                    details.append(f"ab_initio: {ab_initio_jobs[k]}")
                if k in hetero_jobs:
                    details.append(f"hetero: {hetero_jobs[k]}")
                if k in k_has_comparison:
                    details.append("comparison: done")
                if details:
                    summary_parts.append(f"   - K={k}: {'; '.join(details)}")
        
        if in_progress_k_values:
            summary_parts.append(f"\n⏳ In progress K values: {sorted(in_progress_k_values)}")
            summary_parts.append(f"   → Continue with these K values, do NOT restart completed ones")
        
        # Find the highest completed K and provide guidance
        if completed_k_values:
            max_completed_k = max(completed_k_values)
            summary_parts.append(f"\n📊 Status: K={max_completed_k} is the highest completed value.")
            # Check if K=5 is in progress or needs to be started
            if 5 in completed_k_values:
                summary_parts.append(f"   → Both K=3 and K=5 are completed. Check convergence and proceed to refinement.")
            elif 5 in in_progress_k_values:
                summary_parts.append(f"   → K=5 is in progress. Continue with K=5, do NOT restart K=3.")
            else:
                summary_parts.append(f"   → Next step: Start or continue with K=5.")
            summary_parts.append(f"   → Do NOT re-run K={max_completed_k} or lower values.")
        
        return "\n".join(summary_parts) if summary_parts else "No K values completed yet."
    
    def _analyze_heterogeneity_depth_progress(self, tool_executions: List[Dict[str, Any]]) -> str:
        """
        Analyze heterogeneity depth analysis progress from tool executions.
        
        Args:
            tool_executions: List of tool execution dictionaries
            
        Returns:
            String summary of completed work
        """
        hetero_jobs = []  # List of all heterogeneous refinement jobs
        homo_jobs = []  # List of all homogeneous refinement jobs
        completed_clusters = set()  # Track which starting clusters have been processed
        branch_structure = {}  # Track branch structure: parent_job -> [child_jobs]
        
        # Track tool executions in order
        for tool_exec in tool_executions:
            tool_name = tool_exec.get("tool_name", "")
            args = tool_exec.get("arguments", {})
            result = tool_exec.get("result")
            
            # Parse result - it might be a string (JSON) or a dict
            result_dict = result
            if isinstance(result, str):
                if result.strip().startswith("{") or result.strip().startswith("'"):
                    try:
                        import json
                        result_str = result.strip().strip("'\"")
                        result_dict = json.loads(result_str)
                    except (json.JSONDecodeError, ValueError):
                        result_dict = None
                elif result == "No result" or "No result" in result:
                    result_dict = None
                else:
                    result_dict = None
            
            # Check if result is successful
            is_successful = isinstance(result_dict, dict) and result_dict.get("success")
            
            # Track heterogeneous_refinement jobs
            if tool_name == "heterogeneous_refinement" and is_successful:
                job_uid = result_dict.get("job_uid") or result_dict.get("hetero_job_uid")
                if job_uid:
                    hetero_jobs.append({
                        "job_uid": job_uid,
                        "parent_job": args.get("particles_job_uid") or args.get("volume_job_uid"),
                        "k": args.get("num_classes", 4)
                    })
            
            # Track homogeneous_refinement jobs (final refinements)
            elif tool_name == "run_homogeneous_refinement" and is_successful:
                job_uid = result_dict.get("job_uid")
                if job_uid:
                    parent_job = args.get("particles_job_uid") or args.get("volume_job_uid")
                    homo_jobs.append({
                        "job_uid": job_uid,
                        "parent_job": parent_job
                    })
            
            # Track read_input_json to identify starting clusters
            elif tool_name == "read_input_json" and is_successful:
                if isinstance(result_dict, dict):
                    clusters = result_dict.get("clusters", [])
                    for cluster in clusters:
                        refinement_job_uid = cluster.get("refinement_job_uid")
                        if refinement_job_uid:
                            completed_clusters.add(refinement_job_uid)
        
        # Build summary
        summary_parts = []
        
        if completed_clusters:
            summary_parts.append(f"✅ Starting clusters processed: {sorted(completed_clusters)}")
        
        if hetero_jobs:
            summary_parts.append(f"\n📊 Heterogeneous refinement jobs: {len(hetero_jobs)}")
            for i, hetero_job in enumerate(hetero_jobs[-5:], 1):  # Show last 5
                summary_parts.append(f"   {i}. {hetero_job['job_uid']} (K={hetero_job['k']}, parent: {hetero_job['parent_job']})")
            if len(hetero_jobs) > 5:
                summary_parts.append(f"   ... and {len(hetero_jobs) - 5} more")
        
        if homo_jobs:
            summary_parts.append(f"\n✅ Final homogeneous refinement jobs: {len(homo_jobs)}")
            for i, homo_job in enumerate(homo_jobs[-5:], 1):  # Show last 5
                summary_parts.append(f"   {i}. {homo_job['job_uid']} (parent: {homo_job['parent_job']})")
            if len(homo_jobs) > 5:
                summary_parts.append(f"   ... and {len(homo_jobs) - 5} more")
        
        if not summary_parts:
            summary_parts.append("No heterogeneity depth analysis work completed yet.")
        
        return "\n".join(summary_parts)
    
    def _analyze_optimization_progress(self, tool_executions: List[Dict[str, Any]]) -> str:
        """
        Summarize optimization work already done, factually, from the actual tool
        executions in the log.

        The optimization stage is now driven by atomic tools (extract_particles,
        ab_initio_reconstruction, heterogeneous_refinement, regroup_classes,
        nonuniform_refinement, homogeneous_refinement, get_fsc_info, ...). This
        reports the jobs that were successfully created and any FSC resolutions
        measured, so the resuming agent can see what already ran. It does NOT
        prescribe next steps — the agent decides.

        Args:
            tool_executions: List of tool execution dictionaries

        Returns:
            String summary of completed optimization work
        """
        import json

        created_jobs = []      # (tool_name, job_uid)
        fsc_measurements = []   # (refinement_job_uid, resolution_angstroms)

        for tool_exec in tool_executions:
            tool_name = tool_exec.get("tool_name", "")
            result = tool_exec.get("result")

            # Parse result into a dict when possible.
            result_dict = result
            if isinstance(result, str):
                s = result.strip().strip("'\"")
                if s.startswith("{"):
                    try:
                        result_dict = json.loads(s)
                    except (json.JSONDecodeError, ValueError):
                        result_dict = None
                else:
                    result_dict = None
            if not isinstance(result_dict, dict):
                continue

            # Record any job that was created (job-submitting tools return job_uid).
            job_uid = result_dict.get("job_uid")
            if result_dict.get("success") and job_uid:
                created_jobs.append((tool_name, job_uid))

            # Record FSC measurements (get_fsc_info reports refinement_job_uid + resolution).
            resolution = result_dict.get("resolution_angstroms")
            ref_uid = result_dict.get("refinement_job_uid") or job_uid
            if resolution is not None and ref_uid:
                fsc_measurements.append((ref_uid, resolution))

        summary_parts = []
        if created_jobs:
            summary_parts.append("Jobs already created in this optimization run:")
            for tool_name, job_uid in created_jobs:
                summary_parts.append(f"   - {job_uid} (from {tool_name})")
        if fsc_measurements:
            summary_parts.append("\nFSC resolutions measured (job → resolution):")
            for ref_uid, resolution in fsc_measurements:
                try:
                    summary_parts.append(f"   - {ref_uid}: {float(resolution):.3f} Å")
                except (TypeError, ValueError):
                    summary_parts.append(f"   - {ref_uid}: {resolution} Å")
            best = min(
                (m for m in fsc_measurements if isinstance(m[1], (int, float))),
                key=lambda m: m[1],
                default=None,
            )
            if best:
                summary_parts.append(f"   Best so far: {best[0]} at {best[1]:.3f} Å")

        if not summary_parts:
            return "No optimization jobs completed yet."
        summary_parts.append(
            "\nThese jobs already ran — reuse their results rather than recreating them."
        )
        return "\n".join(summary_parts)

