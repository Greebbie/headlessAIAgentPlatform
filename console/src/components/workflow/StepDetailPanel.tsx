import React from 'react';
import { Card, Form, Input, Select, InputNumber, Switch, Divider, Button } from 'antd';
import { SaveOutlined } from '@ant-design/icons';
import type { WorkflowStep, StepType, OnFailureStrategy, FieldDef } from '../../types';
import FieldBuilder from './FieldBuilder';

interface StepDetailPanelProps {
  step: WorkflowStep;
  onSave: (stepId: string, updates: Partial<WorkflowStep>) => void;
  onClose: () => void;
}

const { TextArea } = Input;

const STEP_TYPES: { value: StepType; label: string }[] = [
  { value: 'collect', label: 'Collect Info' },
  { value: 'validate', label: 'Validate' },
  { value: 'tool_call', label: 'Tool Call' },
  { value: 'confirm', label: 'Confirm' },
  { value: 'human_review', label: 'Human Review' },
  { value: 'complete', label: 'Complete' },
];

const FAILURE_STRATEGIES: { value: OnFailureStrategy; label: string }[] = [
  { value: 'retry', label: 'Retry' },
  { value: 'skip', label: 'Skip' },
  { value: 'rollback', label: 'Rollback' },
  { value: 'escalate', label: 'Escalate' },
];

export default function StepDetailPanel({ step, onSave, onClose }: StepDetailPanelProps) {
  const [form] = Form.useForm();

  React.useEffect(() => {
    form.setFieldsValue({
      name: step.name,
      step_type: step.step_type,
      order: step.order,
      prompt_template: step.prompt_template,
      on_failure: step.on_failure,
      max_retries: step.max_retries,
      requires_human_confirm: step.requires_human_confirm,
      risk_level: step.risk_level,
    });
  }, [step, form]);

  const [fields, setFields] = React.useState<FieldDef[]>(
    Array.isArray(step.fields) ? step.fields : []
  );

  React.useEffect(() => {
    setFields(Array.isArray(step.fields) ? step.fields : []);
  }, [step]);

  const handleSave = () => {
    form.validateFields().then((values) => {
      onSave(step.id, {
        ...values,
        fields: fields.length > 0 ? fields : null,
      });
    });
  };

  return (
    <Card
      title={`Edit: ${step.name}`}
      extra={<Button type="text" onClick={onClose}>Close</Button>}
      style={{ width: 360, height: '100%', overflow: 'auto' }}
      size="small"
    >
      <Form form={form} layout="vertical" size="small">
        <Form.Item label="Step Name" name="name" rules={[{ required: true }]}>
          <Input />
        </Form.Item>
        <Form.Item label="Step Type" name="step_type">
          <Select options={STEP_TYPES} />
        </Form.Item>
        <Form.Item label="Order" name="order">
          <InputNumber min={0} style={{ width: '100%' }} />
        </Form.Item>
        <Form.Item label="Prompt Template" name="prompt_template">
          <TextArea rows={3} />
        </Form.Item>

        <Divider plain>Fields</Divider>
        <FieldBuilder fields={fields} onChange={setFields} />

        <Divider plain>Error Handling</Divider>
        <Form.Item label="On Failure" name="on_failure">
          <Select options={FAILURE_STRATEGIES} />
        </Form.Item>
        <Form.Item label="Max Retries" name="max_retries">
          <InputNumber min={0} max={10} style={{ width: '100%' }} />
        </Form.Item>
        <Form.Item label="Risk Level" name="risk_level">
          <Select
            options={[
              { value: 'info', label: 'Info' },
              { value: 'warning', label: 'Warning' },
              { value: 'critical', label: 'Critical' },
            ]}
          />
        </Form.Item>
        <Form.Item name="requires_human_confirm" valuePropName="checked">
          <Switch checkedChildren="Human Confirm" unCheckedChildren="Auto" />
        </Form.Item>

        <Button type="primary" onClick={handleSave} block icon={<SaveOutlined />}>
          Save Step
        </Button>
      </Form>
    </Card>
  );
}
