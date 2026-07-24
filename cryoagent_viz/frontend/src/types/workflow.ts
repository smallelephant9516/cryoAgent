/**
 * TypeScript interfaces for CryoAgent workflow data structures.
 */

export interface WorkflowMetadata {
  project_uid?: string;
  workspace_uid?: string;
  total_execution_time?: number;
  total_stages?: number;
  successful_stages?: number;
  failed_stages?: number;
}

export interface StageMetrics {
  resolution_angstroms?: number;
  num_particles?: number;
  box_size?: number;
  symmetry?: string;
  cfar?: number;
  cfar_label?: string;
  num_classes?: number;
  [key: string]: any;
}

export interface TestedCombination {
  box_size?: number;
  resolution?: number;
  job_uid: string;
  iteration?: number;
  phase?: string;
  type?: string;
  k?: number;
  note?: string;
}

export interface StageRecord {
  stage: string;
  success: boolean;
  primary_job_uid?: string;
  metrics: StageMetrics;
  goal?: string;
  decisions: string[];
  stage_outputs: Record<string, any>;
  timestamp: number;
  execution_time?: number;
  reasoning_summary?: string;
  assessment?: string;
  detailed_results?: {
    stage?: string;
    status?: string;
    tested_combinations?: TestedCombination[];
    [key: string]: any;
  };
  llm_log?: string;
}

export interface WorkflowState {
  stages: StageRecord[];
  metadata: WorkflowMetadata;
}

export interface WorkflowTimelineItem {
  stage: string;
  start_offset: number;
  duration: number;
}

export interface WorkflowSummary {
  report_type?: string;
  timestamp?: string;
  conversation_id?: string;
  workflow_metadata: WorkflowMetadata;
  executive_summary?: string;
  stage_summaries?: Array<{
    stage: string;
    status: string;
    execution_time: number;
    [key: string]: any;
  }>;
  workflow_timeline: WorkflowTimelineItem[];
  output_files?: string[];
  next_steps?: string[];
}

export interface WorkflowData {
  workflow_state: WorkflowState;
  summary: WorkflowSummary | null;
  output_dir: string;
  last_updated: string;
}

export interface WorkflowListItem {
  path: string;
  name: string;
  project: string;
  trial: string;
}

export interface WorkflowListResponse {
  workflows: WorkflowListItem[];
  count: number;
  registered_dirs?: string[];
}
