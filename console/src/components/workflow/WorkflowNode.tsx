import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Tag } from 'antd';
import {
  FormOutlined,
  CheckCircleOutlined,
  ApiOutlined,
  QuestionCircleOutlined,
  UserOutlined,
  FlagOutlined,
} from '@ant-design/icons';
import type { WorkflowStep, StepType } from '../../types';

const STEP_CONFIG: Record<StepType, { icon: React.ReactNode; color: string; label: string }> = {
  collect: { icon: <FormOutlined />, color: '#1890ff', label: 'Collect' },
  validate: { icon: <CheckCircleOutlined />, color: '#52c41a', label: 'Validate' },
  tool_call: { icon: <ApiOutlined />, color: '#722ed1', label: 'Tool Call' },
  confirm: { icon: <QuestionCircleOutlined />, color: '#fa8c16', label: 'Confirm' },
  human_review: { icon: <UserOutlined />, color: '#eb2f96', label: 'Review' },
  complete: { icon: <FlagOutlined />, color: '#13c2c2', label: 'Complete' },
};

interface WorkflowNodeData {
  step: WorkflowStep;
  index: number;
  total: number;
}

export default function WorkflowNode({ data, selected }: NodeProps<WorkflowNodeData>) {
  const { step, index, total } = data;
  const config = STEP_CONFIG[step.step_type as StepType] || STEP_CONFIG.collect;
  const fieldCount = Array.isArray(step.fields) ? step.fields.length : 0;
  const hasRules = step.next_step_rules && Array.isArray(step.next_step_rules) && step.next_step_rules.length > 0;

  return (
    <div
      style={{
        padding: '12px 16px',
        borderRadius: 8,
        border: `2px solid ${selected ? config.color : '#333'}`,
        background: selected ? `${config.color}15` : '#1a1a2e',
        minWidth: 200,
        cursor: 'pointer',
      }}
    >
      <Handle type="target" position={Position.Top} style={{ background: '#555' }} />
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
        <span style={{ color: config.color, fontSize: 16 }}>{config.icon}</span>
        <strong style={{ color: '#fff', fontSize: 13 }}>{step.name}</strong>
        <Tag color={config.color} style={{ marginLeft: 'auto', fontSize: 10 }}>
          {config.label}
        </Tag>
      </div>
      <div style={{ fontSize: 11, color: '#999' }}>
        Step {index + 1}/{total}
        {fieldCount > 0 && <span> &middot; {fieldCount} fields</span>}
        {hasRules && <span> &middot; branching</span>}
      </div>
      {step.prompt_template && (
        <div
          style={{
            fontSize: 10,
            color: '#666',
            marginTop: 4,
            maxWidth: 200,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {step.prompt_template.slice(0, 60)}
        </div>
      )}
      <Handle type="source" position={Position.Bottom} style={{ background: '#555' }} />
    </div>
  );
}
