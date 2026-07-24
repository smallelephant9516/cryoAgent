import { useState } from 'react';
import WorkflowGraph from './components/WorkflowGraph';
import DetailPanel from './components/DetailPanel';
import TimelineBar from './components/TimelineBar';
import { useWebSocket } from './hooks/useWebSocket';
import { useWorkflowData } from './hooks/useWorkflowData';
import { useWorkflowList } from './hooks/useWorkflowList';
import type { StageRecord } from './types/workflow';

export default function App() {
  const [selectedWorkflowPath, setSelectedWorkflowPath] = useState<string>('');
  const [realtimeMode, setRealtimeMode] = useState(false);
  const [selectedStage, setSelectedStage] = useState<StageRecord | null>(null);
  const [folderInput, setFolderInput] = useState('');
  const [showAddFolder, setShowAddFolder] = useState(false);
  const [scanMessage, setScanMessage] = useState<string | null>(null);

  const { workflows, scanning, error: listError, scanFolder } = useWorkflowList();

  const { data: staticData, loading } = useWorkflowData(selectedWorkflowPath, !realtimeMode);
  const { data: realtimeData, connected } = useWebSocket(selectedWorkflowPath, realtimeMode);

  const workflowData = realtimeMode ? realtimeData : staticData;
  const workflowOptions = workflows?.workflows ?? [];

  const handleWorkflowChange = (path: string) => {
    setSelectedWorkflowPath(path);
    setSelectedStage(null);
  };

  const handleAddFolder = async () => {
    setScanMessage(null);
    try {
      const result = await scanFolder(folderInput);
      setScanMessage(
        result.found_count > 0
          ? `Added ${result.found_count} workflow${result.found_count === 1 ? '' : 's'} from ${result.base_dir}`
          : `No workflows found in ${result.base_dir} (need workflow_state.json)`
      );
      setFolderInput('');
      if (result.found_count > 0) {
        setShowAddFolder(false);
      }
    } catch {
      // Error is surfaced via listError from the hook
    }
  };

  return (
    <div className="h-screen flex flex-col bg-cream">
      {/* Top Bar */}
      <div className="bg-charcoal text-white px-6 py-4 flex items-center justify-between shadow-lg gap-4">
        <div className="min-w-0">
          <h1 className="text-2xl font-serif">CryoAgent Workflow Visualizer</h1>
          {workflowData && (
            <div className="text-xs text-gray-300 mt-1 truncate">
              {workflowData.workflow_state.metadata.project_uid} / {workflowData.workflow_state.metadata.workspace_uid}
            </div>
          )}
        </div>

        <div className="flex items-center gap-3 flex-shrink-0">
          <select
            value={selectedWorkflowPath}
            onChange={(e) => handleWorkflowChange(e.target.value)}
            className="px-4 py-2 rounded-full bg-white text-charcoal text-sm font-medium focus:outline-none focus:ring-2 focus:ring-terracotta max-w-xs"
          >
            <option value="">Select workflow...</option>
            {workflowOptions.map((workflow) => (
              <option key={workflow.path} value={workflow.path}>
                {workflow.name}
              </option>
            ))}
          </select>

          <button
            type="button"
            onClick={() => {
              setShowAddFolder((open) => !open);
              setScanMessage(null);
            }}
            className="px-4 py-2 rounded-full bg-terracotta hover:bg-terracotta-hover text-white text-sm font-medium transition-colors"
          >
            {showAddFolder ? 'Cancel' : 'Add folder'}
          </button>

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={realtimeMode}
              onChange={(e) => setRealtimeMode(e.target.checked)}
              className="w-4 h-4 rounded border-gray-300 text-terracotta focus:ring-terracotta"
            />
            <span className="text-sm">Real-time</span>
            {realtimeMode && (
              <span
                className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`}
                title={connected ? 'Connected' : 'Disconnected'}
              />
            )}
          </label>

          {loading && (
            <div className="text-xs text-gray-300">Loading...</div>
          )}
        </div>
      </div>

      {/* Add folder panel */}
      {showAddFolder && (
        <div className="bg-charcoal/95 border-b border-white/10 px-6 py-3">
          <div className="flex flex-wrap items-center gap-3">
            <label className="text-sm text-gray-300 whitespace-nowrap" htmlFor="folder-path">
              Output folder
            </label>
            <input
              id="folder-path"
              type="text"
              value={folderInput}
              onChange={(e) => setFolderInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  void handleAddFolder();
                }
              }}
              placeholder="/path/to/outputs/dynamic_mode"
              className="flex-1 min-w-[16rem] px-3 py-2 rounded-lg bg-white text-charcoal text-sm focus:outline-none focus:ring-2 focus:ring-terracotta"
            />
            <button
              type="button"
              onClick={() => void handleAddFolder()}
              disabled={scanning || !folderInput.trim()}
              className="px-4 py-2 rounded-full bg-terracotta hover:bg-terracotta-hover disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-medium transition-colors"
            >
              {scanning ? 'Scanning…' : 'Scan'}
            </button>
          </div>
          <p className="text-xs text-gray-400 mt-2">
            Paste an absolute path to a mode folder, project folder, or single trial that contains{' '}
            <code className="text-gray-300">workflow_state.json</code>.
          </p>
          {(listError || scanMessage) && (
            <p className={`text-xs mt-2 ${listError ? 'text-red-300' : 'text-green-300'}`}>
              {listError || scanMessage}
            </p>
          )}
        </div>
      )}

      {/* Empty state */}
      {!selectedWorkflowPath && (
        <div className="flex-1 flex items-center justify-center px-6">
          <div className="text-center max-w-md">
            <h2 className="text-xl font-serif text-charcoal mb-2">No workflow selected</h2>
            <p className="text-sm text-muted mb-4">
              {workflowOptions.length === 0
                ? 'Add an output folder to discover workflows, then pick one from the dropdown.'
                : 'Choose a workflow from the dropdown above.'}
            </p>
            {workflowOptions.length === 0 && !showAddFolder && (
              <button
                type="button"
                onClick={() => setShowAddFolder(true)}
                className="px-5 py-2 rounded-full bg-terracotta hover:bg-terracotta-hover text-white text-sm font-medium transition-colors"
              >
                Add folder
              </button>
            )}
          </div>
        </div>
      )}

      {/* Timeline Bar */}
      {selectedWorkflowPath && workflowData?.summary?.workflow_timeline && (
        <TimelineBar timeline={workflowData.summary.workflow_timeline} />
      )}

      {/* Main Content */}
      {selectedWorkflowPath && (
        <div className="flex-1 relative">
          <WorkflowGraph workflowData={workflowData} onNodeClick={setSelectedStage} />

          {selectedStage && (
            <DetailPanel stageData={selectedStage} onClose={() => setSelectedStage(null)} />
          )}
        </div>
      )}

      {/* Status Bar */}
      {selectedWorkflowPath && workflowData && (
        <div className="bg-white border-t-2 border-border px-6 py-2 flex items-center justify-between text-xs text-muted">
          <div>
            {workflowData.workflow_state.stages.length} stages •{' '}
            {workflowData.workflow_state.stages.filter(s => s.success).length} successful •{' '}
            {workflowData.workflow_state.stages.filter(s => !s.success).length} failed
          </div>
          <div>
            Last updated: {new Date(workflowData.last_updated).toLocaleString()}
          </div>
        </div>
      )}
    </div>
  );
}
