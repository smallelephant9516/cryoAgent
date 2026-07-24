import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import type { TestedCombination } from '../types/workflow';

interface MetricChartProps {
  data: TestedCombination[];
}

export default function MetricChart({ data }: MetricChartProps) {
  // Transform data for chart - convert resolution to reciprocal (1/Å)
  const chartData = data.map((item, idx) => ({
    iteration: item.iteration !== undefined ? item.iteration : idx,
    resolution_angstrom: item.resolution,
    resolution_reciprocal: item.resolution ? 1 / item.resolution : 0,
    box_size: item.box_size,
    job_uid: item.job_uid,
    phase: item.phase || 'unknown',
  }));

  // Define phase colors
  const phaseColors: Record<string, string> = {
    reconstruction_baseline: '#9E9E9E',  // gray for reconstruction baseline
    '3d_classification': '#2196F3',  // blue for 3D classification
    heterogeneous_refinement: '#9C27B0',  // purple for heterogeneous refinement
    box_size_optimization: '#C4612F',  // terracotta for box size optimization
    unknown: '#757575',
  };

  // Phase labels for legend
  const phaseLabels: Record<string, string> = {
    reconstruction_baseline: 'Reconstruction Baseline (Iteration 0)',
    '3d_classification': '3D Classification',
    heterogeneous_refinement: 'Heterogeneous Refinement',
    box_size_optimization: 'Box Size Optimization',
  };

  // Custom tick formatter to show "1/{round(resolution,2)}" format
  const formatReciprocalTick = (value: number) => {
    if (value === 0) return '0';
    const angstrom = 1 / value;
    return `1/${angstrom.toFixed(2)}`;
  };

  // Custom dot component that colors based on phase
  const CustomDot = (props: any) => {
    const { cx, cy, payload } = props;
    const phase = payload.phase || 'unknown';
    const color = phaseColors[phase] || phaseColors.unknown;

    return (
      <circle
        cx={cx}
        cy={cy}
        r={5}
        fill={color}
        stroke={color}
        strokeWidth={2}
      />
    );
  };

  // Get unique phases in order for legend
  const uniquePhases = Array.from(new Set(chartData.map(d => d.phase)));

  return (
    <div className="bg-white p-4 rounded border border-border">
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData} margin={{ top: 20, right: 20, left: 20, bottom: 30 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#E7E1D7" />
          <XAxis
            dataKey="iteration"
            label={{ value: 'Iteration', position: 'insideBottom', offset: -20 }}
            stroke="#5C635D"
          />
          <YAxis
            label={{
              value: 'Resolution (1/Å)',
              angle: -90,
              position: 'insideLeft',
              style: { textAnchor: 'middle' },
              offset: -10
            }}
            stroke="#5C635D"
            domain={['auto', 'auto']}
            tickFormatter={formatReciprocalTick}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#FBF9F5',
              border: '1px solid #E7E1D7',
              borderRadius: '4px',
              padding: '8px 12px',
            }}
            labelStyle={{ color: '#1F2421', fontWeight: 600 }}
            labelFormatter={(iteration) => `Iteration ${iteration}`}
            formatter={(value: number, _name: string, props: any) => {
              const angstrom = props.payload.resolution_angstrom;
              const jobUid = props.payload.job_uid;
              const phaseName = phaseLabels[props.payload.phase as keyof typeof phaseLabels] || 'Unknown';
              const boxSize = props.payload.box_size;

              return [
                <div key="tooltip" className="space-y-1">
                  <div className="font-semibold text-primary">{angstrom ? `${angstrom.toFixed(3)} Å` : 'N/A'}</div>
                  <div className="text-sm text-muted">{phaseName}</div>
                  <div className="text-xs text-muted">Job: {jobUid}</div>
                  {boxSize && <div className="text-xs text-muted">Box size: {boxSize}px</div>}
                </div>
              ];
            }}
          />

          {/* Single continuous line with phase-colored dots */}
          <Line
            type="monotone"
            dataKey="resolution_reciprocal"
            stroke="#5C635D"
            strokeWidth={2}
            dot={<CustomDot />}
            activeDot={{ r: 7 }}
            connectNulls={false}
          />
        </LineChart>
      </ResponsiveContainer>

      {/* Custom legend showing phases */}
      <div className="flex flex-wrap gap-4 justify-center mt-4">
        {uniquePhases.map((phase) => (
          <div key={phase} className="flex items-center gap-2">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: phaseColors[phase] || phaseColors.unknown }}
            />
            <span className="text-sm text-muted">
              {phaseLabels[phase as keyof typeof phaseLabels] || 'Unknown'}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
