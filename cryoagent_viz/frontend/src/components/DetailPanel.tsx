import type { StageRecord } from '../types/workflow';
import MetricChart from './MetricChart';

interface DetailPanelProps {
  stageData: StageRecord | null;
  onClose: () => void;
}

export default function DetailPanel({ stageData, onClose }: DetailPanelProps) {
  if (!stageData) return null;

  // Format stage name
  const stageName = stageData.stage
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');

  return (
    <div className="fixed right-0 top-0 h-screen w-[600px] bg-cream-light shadow-2xl overflow-y-auto border-l-2 border-border z-50 animate-slide-in">
      <div className="p-6">
        {/* Header */}
        <div className="flex justify-between items-start mb-6">
          <h2 className="text-2xl font-serif text-ink">
            {stageName}
          </h2>
          <button
            onClick={onClose}
            className="text-muted hover:text-ink transition-colors text-2xl leading-none"
            aria-label="Close panel"
          >
            ×
          </button>
        </div>

        {/* Status Badge */}
        <div
          className={`inline-block px-3 py-1 rounded-full text-sm font-medium mb-6 ${
            stageData.success
              ? 'bg-terracotta-tint text-terracotta'
              : 'bg-red-100 text-red-700'
          }`}
        >
          {stageData.success ? '✓ Completed' : '✗ Failed'}
        </div>

        {/* Execution Time */}
        {(stageData.execution_time && stageData.execution_time > 0) ? (
          <div className="mb-6">
            <div className="text-xs font-semibold text-muted uppercase tracking-wide mb-1">
              Execution Time
            </div>
            <div className="text-lg text-ink">
              {(stageData.execution_time / 60).toFixed(1)} minutes
            </div>
          </div>
        ) : null}

        {/* Primary Job UID */}
        {stageData.primary_job_uid && (
          <div className="mb-6">
            <div className="text-xs font-semibold text-muted uppercase tracking-wide mb-1">
              Primary Job
            </div>
            <code className="text-sm text-ink bg-white px-2 py-1 rounded border border-border">
              {stageData.primary_job_uid}
            </code>
          </div>
        )}

        {/* Goal */}
        {stageData.goal && (
          <div className="mb-6">
            <h3 className="text-xs font-semibold text-muted uppercase tracking-wide mb-2">
              Goal
            </h3>
            <p className="text-ink leading-relaxed">{stageData.goal}</p>
          </div>
        )}

        {/* Key Metrics */}
        <div className="mb-6">
          <h3 className="text-xs font-semibold text-muted uppercase tracking-wide mb-3">
            Key Metrics
          </h3>
          <div className="grid grid-cols-2 gap-3">
            {Object.entries(stageData.metrics || {}).map(([key, value]) => {
              if (value === null || value === undefined) return null;

              // Format key
              const label = key
                .replace(/_/g, ' ')
                .replace(/\b\w/g, c => c.toUpperCase());

              // Format value
              let displayValue = value;
              if (typeof value === 'number') {
                if (key.includes('resolution')) {
                  displayValue = `${value.toFixed(2)} Å`;
                } else if (key.includes('particles') || key === 'num_classes') {
                  displayValue = value.toLocaleString();
                } else {
                  displayValue = value.toString();
                }
              }

              return (
                <div key={key} className="bg-white p-3 rounded border border-border">
                  <div className="text-xs text-muted">{label}</div>
                  <div className="text-base font-semibold text-ink mt-1">
                    {displayValue}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Iterative Improvements Chart */}
        {stageData.detailed_results?.tested_combinations &&
          stageData.detailed_results.tested_combinations.length > 0 && (
            <div className="mb-6">
              <h3 className="text-xs font-semibold text-muted uppercase tracking-wide mb-3">
                Optimization Progress
              </h3>
              <MetricChart data={stageData.detailed_results.tested_combinations} />
            </div>
          )}

        {/* Decisions */}
        {stageData.decisions && stageData.decisions.length > 0 && (
          <div className="mb-6">
            <h3 className="text-xs font-semibold text-muted uppercase tracking-wide mb-2">
              Decisions ({stageData.decisions.length})
            </h3>
            <ul className="space-y-2">
              {stageData.decisions.map((decision, idx) => (
                <li
                  key={idx}
                  className="text-xs text-ink font-mono bg-white p-2 rounded border border-border"
                >
                  {decision}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Reasoning Summary */}
        {stageData.reasoning_summary && (
          <div className="mb-6">
            <h3 className="text-xs font-semibold text-muted uppercase tracking-wide mb-2">
              Reasoning Summary
            </h3>
            <p className="text-sm text-ink bg-white p-3 rounded border border-border">
              {stageData.reasoning_summary}
            </p>
          </div>
        )}

        {/* Assessment */}
        {stageData.assessment && (
          <div className="mb-6">
            <h3 className="text-xs font-semibold text-muted uppercase tracking-wide mb-2">
              Assessment
            </h3>
            <p className="text-sm text-ink bg-white p-3 rounded border border-border">
              {stageData.assessment}
            </p>
          </div>
        )}

        {/* LLM Reasoning Log */}
        {stageData.llm_log && (
          <details className="mb-6">
            <summary className="text-xs font-semibold text-muted uppercase tracking-wide cursor-pointer mb-2 hover:text-ink">
              LLM Reasoning Log ▾
            </summary>
            <pre className="text-xs text-ink bg-white p-4 rounded border border-border overflow-x-auto whitespace-pre-wrap font-mono max-h-96 overflow-y-auto">
              {stageData.llm_log}
            </pre>
          </details>
        )}
      </div>
    </div>
  );
}
