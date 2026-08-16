import { useEffect, useMemo } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
} from 'reactflow';
import 'reactflow/dist/style.css';
import StageNode from './StageNode';
import type { WorkflowData, StageRecord } from '../types/workflow';

const nodeTypes = { stageNode: StageNode };

interface WorkflowGraphProps {
  workflowData: WorkflowData | null;
  onNodeClick: (stage: StageRecord) => void;
}

function buildDisplayStages(workflowData: WorkflowData | null): StageRecord[] {
  const stages = [...(workflowData?.workflow_state?.stages || [])];
  const currentStage = workflowData?.workflow_state?.current_stage;
  if (!currentStage) {
    return stages;
  }
  if (stages.some((s) => s.stage === currentStage)) {
    return stages;
  }
  // Placeholder node for a stage that has begun but not yet recorded.
  stages.push({
    stage: currentStage,
    success: true,
    metrics: {},
    decisions: [],
    stage_outputs: { status: 'running' },
    timestamp: Date.now() / 1000,
  });
  return stages;
}

export default function WorkflowGraph({ workflowData, onNodeClick }: WorkflowGraphProps) {
  const currentStage = workflowData?.workflow_state?.current_stage || null;
  const stages = useMemo(() => buildDisplayStages(workflowData), [workflowData]);

  // Convert stages to React Flow nodes
  const initialNodes: Node[] = stages.map((stage, idx) => {
    const isRunning = Boolean(currentStage && stage.stage === currentStage);
    return {
      id: stage.stage,
      type: 'stageNode',
      position: { x: 50 + idx * 280, y: 200 },
      data: {
        stage: stage.stage,
        success: stage.success,
        metrics: stage.metrics,
        executionTime: stage.execution_time,
        primaryJobUid: stage.primary_job_uid,
        onClick: () => onNodeClick(stage),
        isFirst: idx === 0,
        isLast: idx === stages.length - 1,
        isRunning,
      },
      draggable: true,
    };
  });

  // Create edges connecting sequential stages
  const initialEdges: Edge[] = stages.slice(0, -1).map((stage, idx) => {
    const target = stages[idx + 1];
    const animated = Boolean(currentStage && target.stage === currentStage);
    return {
      id: `${stage.stage}-${target.stage}`,
      source: stage.stage,
      target: target.stage,
      type: 'smoothstep',
      animated: true,
      style: {
        stroke: animated ? '#2563EB' : '#C4612F',
        strokeWidth: 2,
      },
    };
  });

  // Use React Flow hooks for state management
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Update nodes and edges when workflow data changes
  useEffect(() => {
    setNodes(initialNodes);
    setEdges(initialEdges);
  }, [workflowData]);

  if (stages.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-cream">
        <div className="text-center">
          <div className="text-4xl mb-4">📊</div>
          <div className="text-xl font-serif text-ink mb-2">No workflow data</div>
          <div className="text-muted">Select a workflow to visualize</div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full bg-cream">
      <ReactFlow
        nodes={nodes}
        onNodesChange={onNodesChange}
        edges={edges}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        nodesDraggable={true}
        nodesConnectable={false}
        elementsSelectable={true}
        fitView
        minZoom={0.2}
        maxZoom={1.5}
        defaultViewport={{ x: 0, y: 0, zoom: 0.8 }}
      >
        <Background color="#E7E1D7" gap={16} />
        <Controls />
        <MiniMap
          nodeColor={(node) => {
            if (currentStage && node.id === currentStage) return '#2563EB';
            const stageData = stages.find(s => s.stage === node.id);
            return stageData?.success ? '#C4612F' : '#DC2626';
          }}
          maskColor="rgba(247, 244, 239, 0.8)"
          style={{
            backgroundColor: '#FBF9F5',
            border: '2px solid #E7E1D7',
          }}
        />
      </ReactFlow>
    </div>
  );
}
