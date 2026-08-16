import { Handle, Position } from 'reactflow';
import type { StageRecord } from '../types/workflow';

interface StageNodeProps {
  data: {
    stage: string;
    success: boolean;
    metrics: StageRecord['metrics'];
    executionTime?: number;
    primaryJobUid?: string;
    onClick: () => void;
    isFirst?: boolean;
    isLast?: boolean;
    isRunning?: boolean;
  };
}

export default function StageNode({ data }: StageNodeProps) {
  const { stage, success, metrics, executionTime, onClick, isFirst, isLast, isRunning } = data;

  const statusColor = isRunning ? '#2563EB' : success ? '#C4612F' : '#DC2626';
  const bgColor = isRunning ? '#EFF6FF' : success ? '#FFFFFF' : '#FEF2F2';

  // Format stage name
  const stageName = stage
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');

  return (
    <div
      className={`px-6 py-4 rounded-lg shadow-lg border-2 hover:shadow-xl transition-all ${
        isRunning ? 'stage-node-running' : ''
      }`}
      style={{
        borderColor: statusColor,
        backgroundColor: bgColor,
        minWidth: '220px',
        maxWidth: '280px',
        cursor: 'grab',
      }}
      onDoubleClick={onClick}
    >
      {!isFirst && <Handle type="target" position={Position.Left} style={{ background: statusColor }} />}

      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="text-xs font-medium text-muted uppercase tracking-wide">
          {stageName}
        </div>
        {isRunning && (
          <span className="text-[10px] font-semibold uppercase tracking-wide text-blue-700 bg-blue-100 px-2 py-0.5 rounded">
            Running
          </span>
        )}
      </div>

      <div className="space-y-1">
        {isRunning && !metrics?.resolution_angstroms && !metrics?.num_particles && !metrics?.num_micrographs && (
          <div className="text-sm text-blue-700">In progress…</div>
        )}
        {metrics?.resolution_angstroms && (
          <div className="text-xl font-semibold text-ink">
            {metrics.resolution_angstroms.toFixed(2)} Å
          </div>
        )}
        {metrics?.num_particles && (
          <div className="text-sm text-muted">
            {metrics.num_particles.toLocaleString()} particles
          </div>
        )}
        {metrics?.num_micrographs && (
          <div className="text-sm text-muted">
            {metrics.num_micrographs} micrographs
          </div>
        )}
        {metrics?.box_size && (
          <div className="text-xs text-muted">
            Box: {metrics.box_size}px
          </div>
        )}
        {metrics?.symmetry && (
          <div className="text-xs text-muted">
            Symmetry: {metrics.symmetry}
          </div>
        )}
        {(executionTime && executionTime > 0) ? (
          <div className="text-xs text-muted mt-2 pt-2 border-t border-border">
            ⏱ {(executionTime / 60).toFixed(1)} min
          </div>
        ) : null}
      </div>

      {!isLast && <Handle type="source" position={Position.Right} style={{ background: statusColor }} />}
    </div>
  );
}
