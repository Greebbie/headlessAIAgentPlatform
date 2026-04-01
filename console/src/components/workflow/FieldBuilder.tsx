import React from 'react';
import { Button, Input, Select, Switch, Space, Card, Tooltip } from 'antd';
import { PlusOutlined, DeleteOutlined } from '@ant-design/icons';
import type { FieldDef } from '../../types';

interface FieldBuilderProps {
  fields: FieldDef[];
  onChange: (fields: FieldDef[]) => void;
}

const FIELD_TYPES = [
  { value: 'text', label: 'Text' },
  { value: 'number', label: 'Number' },
  { value: 'date', label: 'Date' },
  { value: 'phone', label: 'Phone' },
  { value: 'email', label: 'Email' },
  { value: 'id_card', label: 'ID Card' },
  { value: 'file', label: 'File Upload' },
  { value: 'select', label: 'Select' },
  { value: 'multi_select', label: 'Multi Select' },
  { value: 'address', label: 'Address' },
];

export default function FieldBuilder({ fields, onChange }: FieldBuilderProps) {
  const addField = () => {
    onChange([
      ...fields,
      {
        name: `field_${fields.length + 1}`,
        label: '',
        field_type: 'text',
        required: true,
        placeholder: '',
      },
    ]);
  };

  const removeField = (index: number) => {
    onChange(fields.filter((_, i) => i !== index));
  };

  const updateField = (index: number, updates: Partial<FieldDef>) => {
    onChange(fields.map((f, i) => (i === index ? { ...f, ...updates } : f)));
  };

  // Suppress unused import warning for React (needed for JSX)
  void React;

  return (
    <div>
      {fields.map((field, idx) => (
        <Card
          key={idx}
          size="small"
          style={{ marginBottom: 8 }}
          extra={
            <Tooltip title="Delete field">
              <Button
                type="text"
                danger
                icon={<DeleteOutlined />}
                onClick={() => removeField(idx)}
                size="small"
              />
            </Tooltip>
          }
        >
          <Space direction="vertical" style={{ width: '100%' }} size={4}>
            <Space style={{ width: '100%' }}>
              <Input
                placeholder="Field name (key)"
                value={field.name}
                onChange={(e) => updateField(idx, { name: e.target.value })}
                style={{ width: 140 }}
                size="small"
              />
              <Input
                placeholder="Label"
                value={field.label}
                onChange={(e) => updateField(idx, { label: e.target.value })}
                style={{ width: 140 }}
                size="small"
              />
            </Space>
            <Space>
              <Select
                value={field.field_type}
                onChange={(v) => updateField(idx, { field_type: v })}
                options={FIELD_TYPES}
                style={{ width: 120 }}
                size="small"
              />
              <Switch
                checked={field.required}
                onChange={(v) => updateField(idx, { required: v })}
                checkedChildren="Required"
                unCheckedChildren="Optional"
                size="small"
              />
            </Space>
            <Input
              placeholder="Placeholder text"
              value={field.placeholder || ''}
              onChange={(e) => updateField(idx, { placeholder: e.target.value })}
              size="small"
            />
          </Space>
        </Card>
      ))}
      <Button type="dashed" onClick={addField} block icon={<PlusOutlined />} size="small">
        Add Field
      </Button>
    </div>
  );
}
