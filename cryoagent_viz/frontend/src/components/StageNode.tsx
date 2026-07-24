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
  };
}

export default function StageNode({ data }: StageNodeProps) {
  const { stage, success, metrics, executionTime, onClick, isFirst, isLast } = data;

  const statusColor = success ? '#C4612F' : '#DC2626'; // terracotta or red
  const bgColor = success ? '#FFFFFF' : '#FEF2F2';

  // Format stage name
  const stageName = stage
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');

  return (
    <div
      className="px-6 py-4 rounded-lg shadow-lg border-2 hover:shadow-xl transition-all"
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

      <div className="text-xs font-medium text-muted uppercase tracking-wide mb-2">
        {stageName}
      </div>

      <div className="space-y-1">
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
