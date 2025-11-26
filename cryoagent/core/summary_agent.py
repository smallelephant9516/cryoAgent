"""
Summary Agent for CryoAgent Workflow

This module provides functionality to collect summaries from each stage agent
and generate a comprehensive final report of the entire workflow execution.
"""

import json
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .master_orchestrator import WorkflowStage, StageResult, WorkflowContext


@dataclass
class StageSummary:
    """Summary information for a single workflow stage."""
    stage_name: str
    stage_description: str
    success: bool
    execution_time: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None
    key_outputs: Dict[str, Any] = None
    job_uids: Dict[str, str] = None
    metrics: Dict[str, Any] = None
    notes: Optional[str] = None
    result_file: Optional[str] = None
    
    def __post_init__(self):
        if self.key_outputs is None:
            self.key_outputs = {}
        if self.job_uids is None:
            self.job_uids = {}
        if self.metrics is None:
            self.metrics = {}


class SummaryAgent:
    """Agent responsible for collecting stage summaries and generating final reports."""
    
    def __init__(self, outputs_dir: str = "outputs"):
        """
        Initialize the summary agent.
        
        Args:
            outputs_dir: Directory where summary reports will be saved
        """
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger("SummaryAgent")
        self.stage_summaries: List[StageSummary] = []
        self.workflow_start_time: Optional[float] = None
        self.workflow_end_time: Optional[float] = None
        self.workflow_context: Optional['WorkflowContext'] = None
        
    def set_workflow_start_time(self, start_time: float):
        """Set the workflow start time."""
        self.workflow_start_time = start_time
    
    def set_workflow_end_time(self, end_time: float):
        """Set the workflow end time."""
        self.workflow_end_time = end_time
    
    def set_workflow_context(self, context: 'WorkflowContext'):
        """Set the workflow context."""
        self.workflow_context = context
    
    def add_stage_summary(self, stage_result: 'StageResult', stage_agent: Any) -> StageSummary:
        """
        Add a summary for a completed stage.
        
        Args:
            stage_result: The StageResult from the stage execution
            stage_agent: The stage agent that executed the stage
            
        Returns:
            StageSummary object containing the stage summary
        """
        # Import at runtime to avoid circular import
        from .master_orchestrator import WorkflowStage
        
        # Extract key information from stage result
        stage_outputs = stage_result.stage_outputs or {}
        
        # Get stage description from agent
        stage_description = ""
        if hasattr(stage_agent, 'get_stage_description'):
            stage_description = stage_agent.get_stage_description()
        
        # Extract key outputs and job UIDs
        key_outputs = {}
        job_uids = {}
        metrics = {}
        
        # Extract job UIDs from stage_outputs
        for key, value in stage_outputs.items():
            if key.endswith("_job_uid") and value:
                job_name = key.replace("_job_uid", "")
                job_uids[job_name] = value
            elif key in ["result_file", "output_file", "final_volume_job_uid", 
                         "final_particles_job_uid", "selected_micrographs_star",
                         "final_star_file", "volume_location", "final_volume_absolute_path"]:
                key_outputs[key] = value
            elif key in ["final_good_particles_count", "final_good_particles_percentage",
                        "total_rounds", "best_box_size", "best_resolution_angstroms",
                        "final_resolution", "iterations"]:
                metrics[key] = value
        
        # Extract additional metrics from nested structures, but filter out technical fields
        if "workflow_summary" in stage_outputs:
            workflow_summary = stage_outputs["workflow_summary"]
            if isinstance(workflow_summary, dict):
                # Filter out technical/internal fields
                technical_fields = {'steps', 'workflow_state', 'reasoning_history', 
                                  'function1_enabled', 'function2_enabled', 'max_rounds', 
                                  'threshold_percentage', 'steps_executed'}
                for k, v in workflow_summary.items():
                    if k not in technical_fields and not isinstance(v, (list, dict)):
                        metrics[k] = v
        
        # Create stage summary
        stage_summary = StageSummary(
            stage_name=stage_result.stage.value,
            stage_description=stage_description,
            success=stage_result.success,
            execution_time=stage_result.execution_time,
            error=stage_result.error,
            key_outputs=key_outputs,
            job_uids=job_uids,
            metrics=metrics,
            result_file=stage_outputs.get("result_file") or stage_outputs.get("output_file")
        )
        
        self.stage_summaries.append(stage_summary)
        self.logger.info(f"Added summary for stage: {stage_result.stage.value}")
        
        return stage_summary
    
    def generate_final_report(self, conversation_id: Optional[str] = None) -> str:
        """
        Generate a comprehensive final report based on all stage summaries.
        
        Args:
            conversation_id: Optional conversation ID for tracking
            
        Returns:
            Path to the generated report file
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate total execution time
        total_execution_time = 0.0
        if self.workflow_start_time and self.workflow_end_time:
            total_execution_time = self.workflow_end_time - self.workflow_start_time
        else:
            # Fallback: sum all stage execution times
            total_execution_time = sum(s.execution_time for s in self.stage_summaries)
        
        # Count successful and failed stages
        successful_stages = [s for s in self.stage_summaries if s.success]
        failed_stages = [s for s in self.stage_summaries if not s.success]
        
        # Extract project and workspace information
        project_uid = "Unknown"
        workspace_uid = "Unknown"
        if self.workflow_context:
            project_uid = self.workflow_context.project_uid
            workspace_uid = self.workflow_context.workspace_uid
        
        # Build comprehensive report
        report = {
            "report_type": "cryoagent_workflow_summary",
            "timestamp": timestamp,
            "conversation_id": conversation_id,
            "workflow_metadata": {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "total_stages": len(self.stage_summaries),
                "successful_stages": len(successful_stages),
                "failed_stages": len(failed_stages),
                "total_execution_time_seconds": total_execution_time,
                "workflow_start_time": self.workflow_start_time,
                "workflow_end_time": self.workflow_end_time
            },
            "executive_summary": {
                "overall_status": "success" if len(failed_stages) == 0 else "partial_failure" if len(successful_stages) > 0 else "failure",
                "completion_rate": f"{len(successful_stages)}/{len(self.stage_summaries)} stages completed successfully",
                "total_time": f"{total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)",
                "key_achievements": self._extract_key_achievements(),
                "issues_encountered": [s.error for s in failed_stages if s.error]
            },
            "stage_summaries": [
                {
                    "stage_name": summary.stage_name,
                    "stage_description": summary.stage_description,
                    "status": "success" if summary.success else "failed",
                    "execution_time_seconds": summary.execution_time,
                    "error": summary.error,
                    "key_outputs": summary.key_outputs,
                    "job_uids": summary.job_uids,
                    "metrics": summary.metrics,
                    "result_file": summary.result_file,
                    "notes": summary.notes
                }
                for summary in self.stage_summaries
            ],
            "workflow_timeline": self._generate_timeline(),
            "output_files": self._collect_output_files(),
            "next_steps": self._suggest_next_steps()
        }
        
        # Save report to file
        report_file = self.outputs_dir / f"workflow_summary_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Also generate a human-readable markdown report
        markdown_report = self._generate_markdown_report(report)
        markdown_file = self.outputs_dir / f"workflow_summary_report_{timestamp}.md"
        with open(markdown_file, 'w') as f:
            f.write(markdown_report)
        
        self.logger.info(f"Final workflow report generated: {report_file}")
        self.logger.info(f"Markdown report generated: {markdown_file}")
        
        return str(report_file)
    
    def _extract_key_achievements(self) -> List[str]:
        """Extract key achievements from stage summaries."""
        achievements = []
        
        for summary in self.stage_summaries:
            if not summary.success:
                continue
                
            stage_name = summary.stage_name.replace('_', ' ').title()
            
            # Preprocessing achievements
            if summary.stage_name == "preprocessing":
                if summary.job_uids.get("micrograph_selection"):
                    achievements.append(f"✅ {stage_name}: Successfully selected micrographs")
                if summary.key_outputs.get("selected_micrographs_star"):
                    achievements.append(f"✅ {stage_name}: Generated RELION micrograph selection file")
            
            # Particle picking achievements
            elif summary.stage_name == "particle_picking":
                if summary.job_uids.get("final_selection") or summary.job_uids.get("selected_particles"):
                    achievements.append(f"✅ {stage_name}: Successfully picked and selected particles")
                if summary.metrics.get("final_good_particles_count"):
                    count = summary.metrics.get("final_good_particles_count")
                    achievements.append(f"✅ {stage_name}: Selected {count:,} good particles")
            
            # 2D optimization achievements
            elif summary.stage_name == "optimization_2d":
                if summary.metrics.get("final_good_particles_percentage"):
                    pct = summary.metrics.get("final_good_particles_percentage")
                    achievements.append(f"✅ {stage_name}: Achieved {pct:.1f}% good particles")
                if summary.metrics.get("total_rounds"):
                    rounds = summary.metrics.get("total_rounds")
                    achievements.append(f"✅ {stage_name}: Completed {rounds} optimization rounds")
            
            # Reconstruction achievements
            elif summary.stage_name == "reconstruction":
                if summary.job_uids.get("final_volume") or summary.key_outputs.get("final_volume_job_uid"):
                    achievements.append(f"✅ {stage_name}: Successfully generated 3D reconstruction")
                if summary.key_outputs.get("final_volume_absolute_path"):
                    achievements.append(f"✅ {stage_name}: Final volume saved")
            
            # Optimization achievements
            elif summary.stage_name == "optimization":
                if summary.metrics.get("best_resolution_angstroms"):
                    res = summary.metrics.get("best_resolution_angstroms")
                    achievements.append(f"✅ {stage_name}: Achieved {res:.2f} Å resolution")
                if summary.metrics.get("best_box_size"):
                    box = summary.metrics.get("best_box_size")
                    achievements.append(f"✅ {stage_name}: Optimized box size to {box} pixels")
            
            # Polish achievements
            elif summary.stage_name == "polish":
                if summary.metrics.get("final_resolution"):
                    res = summary.metrics.get("final_resolution")
                    achievements.append(f"✅ {stage_name}: Final resolution: {res:.2f} Å")
        
        return achievements
    
    def _generate_timeline(self) -> List[Dict[str, Any]]:
        """Generate a timeline of stage executions."""
        timeline = []
        current_time = self.workflow_start_time or 0.0
        
        for summary in self.stage_summaries:
            timeline.append({
                "stage": summary.stage_name,
                "start_time_offset": current_time - (self.workflow_start_time or 0.0),
                "duration_seconds": summary.execution_time,
                "status": "success" if summary.success else "failed"
            })
            current_time += summary.execution_time
        
        return timeline
    
    def _collect_output_files(self) -> List[Dict[str, str]]:
        """Collect all output files from stage summaries."""
        output_files = []
        
        for summary in self.stage_summaries:
            if summary.result_file:
                output_files.append({
                    "stage": summary.stage_name,
                    "file_path": summary.result_file,
                    "file_type": "json"
                })
        
        return output_files
    
    def _suggest_next_steps(self) -> List[str]:
        """Suggest next steps based on completed stages."""
        next_steps = []
        completed_stages = {s.stage_name for s in self.stage_summaries if s.success}
        
        if "preprocessing" not in completed_stages:
            next_steps.append("Run preprocessing stage to import and process micrographs")
        elif "particle_picking" not in completed_stages:
            next_steps.append("Run particle picking stage to detect and extract particles")
        elif "optimization_2d" not in completed_stages:
            next_steps.append("Consider running 2D optimization to improve particle quality")
        elif "reconstruction" not in completed_stages:
            next_steps.append("Run 3D reconstruction to generate initial model")
        elif "optimization" not in completed_stages:
            next_steps.append("Run box size optimization to improve resolution")
        elif "polish" not in completed_stages:
            next_steps.append("Run polish refinement for final resolution improvement")
        else:
            next_steps.append("All stages completed! Consider further refinement or analysis")
            next_steps.append("Review the final volume and metrics for quality assessment")
        
        return next_steps
    
    def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """Generate a human-readable markdown report."""
        md = []
        md.append("# CryoAgent Workflow Summary Report\n")
        md.append(f"**Generated:** {report['timestamp']}\n")
        md.append(f"**Conversation ID:** {report.get('conversation_id', 'N/A')}\n")
        md.append("\n---\n")
        
        # Executive Summary with better formatting
        md.append("## 📊 Executive Summary\n")
        exec_summary = report['executive_summary']
        
        # Status with emoji
        status_emoji = {
            'success': '✅',
            'partial_failure': '⚠️',
            'failure': '❌'
        }
        status = exec_summary['overall_status']
        emoji = status_emoji.get(status, '📋')
        md.append(f"\n{emoji} **Overall Status:** {status.replace('_', ' ').upper()}\n")
        md.append(f"📈 **Completion Rate:** {exec_summary['completion_rate']}\n")
        md.append(f"⏱️ **Total Execution Time:** {exec_summary['total_time']}\n")
        
        if exec_summary['key_achievements']:
            md.append("\n### Key Achievements\n")
            for achievement in exec_summary['key_achievements']:
                md.append(f"- {achievement}\n")
        
        if exec_summary['issues_encountered']:
            md.append("\n### Issues Encountered\n")
            for issue in exec_summary['issues_encountered']:
                md.append(f"- ⚠️ {issue}\n")
        
        # Workflow Metadata (simplified)
        md.append("\n---\n")
        md.append("## ℹ️ Workflow Information\n")
        metadata = report['workflow_metadata']
        md.append(f"- **Project:** {metadata['project_uid']}\n")
        md.append(f"- **Workspace:** {metadata['workspace_uid']}\n")
        md.append(f"- **Stages Completed:** {metadata['successful_stages']} of {metadata['total_stages']}\n")
        if metadata['failed_stages'] > 0:
            md.append(f"- **Failed Stages:** {metadata['failed_stages']}\n")
        
        # Stage Summaries
        md.append("\n---\n")
        md.append("## Stage Summaries\n")
        
        for i, stage_summary in enumerate(report['stage_summaries'], 1):
            # Stage header with status indicator
            status_icon = "✅" if stage_summary['status'] == 'success' else "❌"
            stage_title = stage_summary['stage_name'].replace('_', ' ').title()
            md.append(f"\n### {i}. {status_icon} {stage_title}\n")
            
            # Description
            md.append(f"*{stage_summary['stage_description']}*\n")
            
            # Status and timing
            if stage_summary['execution_time_seconds'] > 0:
                minutes = stage_summary['execution_time_seconds'] / 60
                if minutes >= 1:
                    md.append(f"\n⏱️ **Execution Time:** {minutes:.1f} minutes ({stage_summary['execution_time_seconds']:.0f} seconds)\n")
                else:
                    md.append(f"\n⏱️ **Execution Time:** {stage_summary['execution_time_seconds']:.1f} seconds\n")
            else:
                md.append(f"\n⏱️ **Execution Time:** < 1 second (loaded from cache)\n")
            
            # Error message if failed
            if stage_summary['error']:
                md.append(f"\n⚠️ **Error:** {stage_summary['error']}\n")
            
            # Job UIDs in a cleaner format
            if stage_summary['job_uids']:
                md.append("\n**CryoSPARC Jobs:**\n")
                for job_name, job_uid in stage_summary['job_uids'].items():
                    # Format job name nicely
                    formatted_name = job_name.replace('_', ' ').title()
                    md.append(f"- {formatted_name}: **{job_uid}**\n")
            
            if stage_summary['metrics']:
                # Filter out technical/internal metrics and format human-friendly ones
                human_metrics = self._format_human_friendly_metrics(stage_summary['metrics'], stage_summary['stage_name'])
                if human_metrics:
                    md.append("\n**Key Results:**\n")
                    for metric_text in human_metrics:
                        md.append(f"- {metric_text}\n")
            
            # Filter out technical/internal outputs
            key_outputs = self._filter_human_friendly_outputs(stage_summary.get('key_outputs', {}))
            if key_outputs:
                md.append("\n**Output Files:**\n")
                for output_name, output_value in key_outputs.items():
                    if isinstance(output_value, str) and len(output_value) > 100:
                        md.append(f"- {output_name.replace('_', ' ').title()}: `{output_value[:100]}...`\n")
                    else:
                        md.append(f"- {output_name.replace('_', ' ').title()}: `{output_value}`\n")
            
            # Result file (if not already shown in outputs)
            if stage_summary['result_file'] and 'result_file' not in stage_summary.get('key_outputs', {}):
                md.append(f"\n📄 **Detailed Results:** `{stage_summary['result_file']}`\n")
        
        # Timeline (simplified)
        md.append("\n---\n")
        md.append("## ⏱️ Workflow Timeline\n")
        for event in report['workflow_timeline']:
            status_icon = "✅" if event['status'] == 'success' else "❌"
            stage_name = event['stage'].replace('_', ' ').title()
            duration = event['duration_seconds']
            if duration >= 60:
                duration_str = f"{duration/60:.1f} min"
            elif duration > 0:
                duration_str = f"{duration:.0f}s"
            else:
                duration_str = "< 1s"
            md.append(f"- {status_icon} **{stage_name}**: {duration_str}\n")
        
        # Output Files
        if report['output_files']:
            md.append("\n---\n")
            md.append("## Output Files\n")
            for output_file in report['output_files']:
                md.append(f"- **{output_file['stage']}**: `{output_file['file_path']}`\n")
        
        # Next Steps
        if report['next_steps']:
            md.append("\n---\n")
            md.append("## Suggested Next Steps\n")
            for step in report['next_steps']:
                md.append(f"- {step}\n")
        
        return "".join(md)
    
    def _format_human_friendly_metrics(self, metrics: Dict[str, Any], stage_name: str) -> List[str]:
        """Format metrics in a human-friendly way, filtering out technical details."""
        human_metrics = []
        
        # Fields to exclude (technical/internal)
        exclude_fields = {
            'steps', 'workflow_state', 'reasoning_history', 'function1_enabled', 
            'function2_enabled', 'max_rounds', 'threshold_percentage', 'steps_executed',
            'total_steps', 'successful_steps', 'failed_steps'
        }
        
        for metric_name, metric_value in metrics.items():
            # Skip technical/internal fields
            if metric_name in exclude_fields:
                continue
            
            # Skip complex nested structures
            if isinstance(metric_value, (list, dict)):
                continue
            
            # Format based on metric type
            if metric_name == 'final_good_particles_count':
                human_metrics.append(f"**{metric_value:,}** good particles selected")
            elif metric_name == 'final_good_particles_percentage':
                human_metrics.append(f"**{metric_value:.1f}%** of particles are high quality")
            elif metric_name == 'total_rounds':
                human_metrics.append(f"Completed **{int(metric_value)}** optimization rounds")
            elif metric_name == 'best_box_size':
                human_metrics.append(f"Optimal box size: **{metric_value}** pixels")
            elif metric_name == 'best_resolution_angstroms':
                human_metrics.append(f"Best resolution achieved: **{metric_value:.2f} Å**")
            elif metric_name == 'final_resolution':
                human_metrics.append(f"Final resolution: **{metric_value:.2f} Å**")
            elif metric_name == 'iterations':
                human_metrics.append(f"Tested **{int(metric_value)}** different configurations")
            elif metric_name == 'final_particles_job_uid':
                # Skip job UIDs as they're shown separately
                continue
            else:
                # Generic formatting
                if isinstance(metric_value, (int, float)):
                    if metric_value >= 1000:
                        human_metrics.append(f"{metric_name.replace('_', ' ').title()}: **{metric_value:,}**")
                    else:
                        human_metrics.append(f"{metric_name.replace('_', ' ').title()}: **{metric_value}**")
                else:
                    human_metrics.append(f"{metric_name.replace('_', ' ').title()}: **{metric_value}**")
        
        return human_metrics
    
    def _filter_human_friendly_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Filter outputs to show only human-relevant information."""
        filtered = {}
        
        # Fields to include (user-relevant)
        include_fields = {
            'result_file', 'output_file', 'final_volume_absolute_path', 
            'selected_micrographs_star', 'final_star_file', 'volume_location'
        }
        
        for key, value in outputs.items():
            # Include if it's in the include list or if it's a file path
            if key in include_fields or (isinstance(value, str) and ('/' in value or value.endswith(('.json', '.star', '.mrc')))):
                filtered[key] = value
        
        return filtered
    
    def clear_summaries(self):
        """Clear all collected summaries."""
        self.stage_summaries = []
        self.workflow_start_time = None
        self.workflow_end_time = None
        self.workflow_context = None

