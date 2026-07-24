import type { WorkflowTimelineItem } from '../types/workflow';

interface TimelineBarProps {
  timeline: WorkflowTimelineItem[];
}

export default function TimelineBar({ timeline }: TimelineBarProps) {
  if (!timeline || timeline.length === 0) return null;

  // Calculate total duration
  const totalDuration = Math.max(
    ...timeline.map(item => item.start_offset + item.duration)
  );

  return (
    <div className="bg-white border-b-2 border-border px-6 py-4">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-xs font-semibold text-muted uppercase tracking-wide">
          Workflow Timeline
        </span>
        <span className="text-xs text-muted">
          ({(totalDuration / 60).toFixed(1)} min total)
        </span>
      </div>

      <div className="relative h-8 bg-cream rounded overflow-hidden border border-border">
        {timeline.map((item, idx) => {
          const leftPercent = (item.start_offset / totalDuration) * 100;
          const widthPercent = (item.duration / totalDuration) * 100;

          // Format stage name
          const stageName = item.stage
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');

          return (
            <div
              key={idx}
              className="absolute h-full bg-terracotta hover:bg-terracotta-hover transition-colors cursor-pointer group"
              style={{
                left: `${leftPercent}%`,
                width: `${widthPercent}%`,
              }}
              title={`${stageName}: ${(item.duration / 60).toFixed(1)} min`}
            >
              <div className="h-full flex items-center justify-center px-2">
                <span className="text-xs font-medium text-white truncate">
                  {widthPercent > 10 ? stageName : ''}
                </span>
              </div>

              {/* Tooltip on hover */}
              <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-1 bg-charcoal text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
                {stageName}: {(item.duration / 60).toFixed(1)} min
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
