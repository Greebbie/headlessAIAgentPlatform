import React, { useCallback, useMemo } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  Node,
  Edge,
  useNodesState,
  useEdgesState,
  MarkerType,
  NodeMouseHandler,
} from 'reactflow';
import 'reactflow/dist/style.css';
import type { WorkflowStep, NextStepRule } from '../../types';
import WorkflowNode from './WorkflowNode';

interface WorkflowCanvasProps {
  steps: WorkflowStep[];
  onSelectStep: (step: WorkflowStep | null) => void;
  selectedStepId: string | null;
}

const nodeTypes = { workflowStep: WorkflowNode };

function stepsToNodes(steps: WorkflowStep[]): Node[] {
  return steps.map((step, idx) => ({
    id: step.id,
    type: 'workflowStep',
    position: { x: 250, y: idx * 160 },
    data: {
      step,
      index: idx,
      total: steps.length,
    },
    selected: false,
  }));
}

function stepsToEdges(steps: WorkflowStep[]): Edge[] {
  const edges: Edge[] = [];

  steps.forEach((step, idx) => {
    const rules = step.next_step_rules;
    if (rules && Array.isArray(rules) && rules.length > 0) {
      rules.forEach((rule: NextStepRule, rIdx: number) => {
        const targetStep = steps.find(
          (s) => s.name === rule.goto_step || String(s.order) === String(rule.goto_step)
        );
        if (targetStep) {
          const label = rule.condition
            ? `${rule.condition.field} ${rule.condition.op} ${rule.condition.value}`
            : 'default';
          edges.push({
            id: `${step.id}-rule-${rIdx}`,
            source: step.id,
            target: targetStep.id,
            label,
            type: 'smoothstep',
            animated: !rule.condition,
            style: { stroke: rule.condition ? '#f59e0b' : '#666' },
            markerEnd: { type: MarkerType.ArrowClosed },
          });
        }
      });
    } else if (idx < steps.length - 1) {
      edges.push({
        id: `${step.id}-next`,
        source: step.id,
        target: steps[idx + 1].id,
        type: 'smoothstep',
        style: { stroke: '#666' },
        markerEnd: { type: MarkerType.ArrowClosed },
      });
    }
  });

  return edges;
}

export default function WorkflowCanvas({ steps, onSelectStep, selectedStepId }: WorkflowCanvasProps) {
  const initialNodes = useMemo(() => stepsToNodes(steps), [steps]);
  const initialEdges = useMemo(() => stepsToEdges(steps), [steps]);
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  React.useEffect(() => {
    setNodes(stepsToNodes(steps));
    setEdges(stepsToEdges(steps));
  }, [steps, setNodes, setEdges]);

  const onNodeClick: NodeMouseHandler = useCallback(
    (_event, node) => {
      const step = steps.find((s) => s.id === node.id);
      onSelectStep(step ?? null);
    },
    [steps, onSelectStep]
  );

  const onPaneClick = useCallback(() => {
    onSelectStep(null);
  }, [onSelectStep]);

  // Suppress unused variable warnings - these are required by useNodesState/useEdgesState
  void selectedStepId;

  return (
    <div style={{ height: '100%', minHeight: 500 }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        nodeTypes={nodeTypes}
        fitView
        attributionPosition="bottom-left"
      >
        <Background />
        <Controls />
        <MiniMap />
      </ReactFlow>
    </div>
  );
}
