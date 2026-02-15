import React, { useEffect, useState } from 'react';
import { Table, Button, Modal, Form, Input, Select, Switch, Space, message, Tag, Popconfirm, InputNumber } from 'antd';
import { PlusOutlined, EditOutlined, DeleteOutlined } from '@ant-design/icons';
import { agentApi, toolApi, knowledgeApi, workflowApi, llmConfigApi } from '../api';

const { TextArea } = Input;

export default function AgentsPage() {
  const [agents, setAgents] = useState<any[]>([]);
  const [tools, setTools] = useState<any[]>([]);
  const [sources, setSources] = useState<any[]>([]);
  const [workflows, setWorkflows] = useState<any[]>([]);
  const [llmConfigs, setLlmConfigs] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [editing, setEditing] = useState<any>(null);
  const [form] = Form.useForm();

  const load = async () => {
    setLoading(true);
    try {
      const [agentRes, toolRes, sourceRes, wfRes] = await Promise.all([
        agentApi.list(),
        toolApi.list(),
        knowledgeApi.listSources(),
        workflowApi.list(),
      ]);
      setAgents(agentRes.data);
      setTools(toolRes.data);
      setSources(sourceRes.data);
      setWorkflows(wfRes.data);
      // LLM configs may not be available yet
      try {
        const llmRes = await llmConfigApi.list();
        setLlmConfigs(llmRes.data);
      } catch {
        // LLM config API not available
      }
    } catch (e: any) {
      message.error('加载数据失败: ' + (e.message || '未知错误'));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const handleSave = async () => {
    try {
      const values = await form.validateFields();
      // Build tool_scope and knowledge_scope
      const payload: any = {
        name: values.name,
        description: values.description || '',
        system_prompt: values.system_prompt || '',
        llm_model: values.llm_model || null,
        workflow_id: values.workflow_id || null,
        enabled: values.enabled ?? true,
      };
      if (values.tool_ids && values.tool_ids.length > 0) {
        payload.tool_scope = {
          tool_ids: values.tool_ids,
          function_calling_enabled: values.function_calling_enabled ?? true,
          max_tool_rounds: values.max_tool_rounds ?? 5,
        };
      }
      if (values.knowledge_source_ids && values.knowledge_source_ids.length > 0) {
        payload.knowledge_scope = values.knowledge_source_ids;
      }

      // Build workflow_scope for multi-workflow binding
      if (values.workflow_ids && values.workflow_ids.length > 0) {
        const descriptions: Record<string, string> = {};
        (values.workflow_ids as string[]).forEach((wfId: string) => {
          const descKey = `wf_desc_${wfId}`;
          if (values[descKey]) {
            descriptions[wfId] = values[descKey];
          } else {
            // Auto-fill from workflow name
            const wf = workflows.find((w: any) => w.id === wfId);
            if (wf) descriptions[wfId] = wf.name;
          }
        });
        payload.workflow_scope = {
          workflow_ids: values.workflow_ids,
          descriptions,
        };
        // Clear legacy workflow_id when using multi-workflow
        payload.workflow_id = null;
      }

      if (editing) {
        await agentApi.update(editing.id, payload);
        message.success('Agent 已更新');
      } else {
        await agentApi.create(payload);
        message.success('Agent 已创建');
      }
      setModalOpen(false);
      setEditing(null);
      form.resetFields();
      load();
    } catch (e: any) {
      if (e.errorFields) return; // form validation error
      message.error('保存失败: ' + (e.response?.data?.detail || e.message || '未知错误'));
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await agentApi.delete(id);
      message.success('已删除');
      load();
    } catch (e: any) {
      message.error('删除失败: ' + (e.response?.data?.detail || e.message || '未知错误'));
    }
  };

  const openEdit = (record: any) => {
    setEditing(record);
    const toolIds = Array.isArray(record.tool_scope)
      ? record.tool_scope
      : record.tool_scope?.tool_ids || [];
    const knowledgeIds = Array.isArray(record.knowledge_scope)
      ? record.knowledge_scope
      : [];
    // Extract workflow_ids from workflow_scope or fall back to legacy workflow_id
    const workflowIds = record.workflow_scope?.workflow_ids
      || (record.workflow_id ? [record.workflow_id] : []);

    const formValues: any = {
      ...record,
      tool_ids: toolIds,
      knowledge_source_ids: knowledgeIds,
      function_calling_enabled: record.tool_scope?.function_calling_enabled ?? true,
      max_tool_rounds: record.tool_scope?.max_tool_rounds ?? 5,
      workflow_ids: workflowIds,
    };

    // Set workflow descriptions
    const descriptions = record.workflow_scope?.descriptions || {};
    workflowIds.forEach((wfId: string) => {
      formValues[`wf_desc_${wfId}`] = descriptions[wfId] || '';
    });

    form.setFieldsValue(formValues);
    setModalOpen(true);
  };

  const selectedWorkflowIds: string[] = Form.useWatch('workflow_ids', form) || [];

  const columns = [
    { title: '名称', dataIndex: 'name', key: 'name' },
    { title: '描述', dataIndex: 'description', key: 'description', ellipsis: true },
    { title: 'LLM', dataIndex: 'llm_model', key: 'llm_model', render: (v: string) => v || '默认' },
    {
      title: '工具', key: 'tools',
      render: (_: any, r: any) => {
        const ids = Array.isArray(r.tool_scope) ? r.tool_scope : r.tool_scope?.tool_ids || [];
        return ids.length > 0 ? <Tag color="purple">{ids.length} 个工具</Tag> : '-';
      },
    },
    {
      title: '工作流', key: 'workflows',
      render: (_: any, r: any) => {
        const wfIds = r.workflow_scope?.workflow_ids || (r.workflow_id ? [r.workflow_id] : []);
        if (wfIds.length === 0) return '-';
        return <Tag color="blue">{wfIds.length} 个工作流</Tag>;
      },
    },
    { title: '状态', dataIndex: 'enabled', key: 'enabled', render: (v: boolean) => v ? <Tag color="green">启用</Tag> : <Tag color="red">停用</Tag> },
    { title: '版本', dataIndex: 'version', key: 'version' },
    {
      title: '操作', key: 'actions', render: (_: any, record: any) => (
        <Space>
          <Button icon={<EditOutlined />} size="small" onClick={() => openEdit(record)}>编辑</Button>
          <Popconfirm title="确认删除此 Agent？" onConfirm={() => handleDelete(record.id)} okText="确认" cancelText="取消">
            <Button icon={<DeleteOutlined />} size="small" danger>删除</Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <h2>Agent 管理</h2>
        <Button type="primary" icon={<PlusOutlined />} onClick={() => { setEditing(null); form.resetFields(); setModalOpen(true); }}>
          创建 Agent
        </Button>
      </div>
      <Table dataSource={agents} columns={columns} rowKey="id" loading={loading} />

      <Modal title={editing ? '编辑 Agent' : '创建 Agent'} open={modalOpen} onOk={handleSave} onCancel={() => setModalOpen(false)} width={720} destroyOnClose>
        <Form form={form} layout="vertical">
          <Form.Item name="name" label="名称" rules={[{ required: true }]}>
            <Input placeholder="如：社区服务助手" />
          </Form.Item>
          <Form.Item name="description" label="描述">
            <Input placeholder="简要描述 Agent 功能" />
          </Form.Item>
          <Form.Item name="system_prompt" label="系统提示词 (Persona)">
            <TextArea rows={4} placeholder="你是XX社区的智能助手，负责..." />
          </Form.Item>
          <Form.Item name="llm_model" label="LLM 模型 (留空使用默认)">
            <Input placeholder="qwen2.5 / glm-4 / deepseek-chat" />
          </Form.Item>
          <Form.Item name="workflow_ids" label="关联工作流 (多选，LLM自动识别意图路由)">
            <Select mode="multiple" allowClear placeholder="选择工作流 (留空为纯QA模式)"
              options={workflows.map((w: any) => ({ value: w.id, label: `${w.name} (${w.steps?.length || 0} 步骤)` }))}
            />
          </Form.Item>
          {/* Dynamic description inputs for each selected workflow */}
          {selectedWorkflowIds.map((wfId: string) => {
            const wf = workflows.find((w: any) => w.id === wfId);
            return (
              <Form.Item
                key={wfId}
                name={`wf_desc_${wfId}`}
                label={`工作流描述: ${wf?.name || wfId}`}
                tooltip="帮助LLM识别用户意图并路由到正确的工作流"
              >
                <Input placeholder={`描述此工作流的用途，如：查询时间、订单办理`} />
              </Form.Item>
            );
          })}
          <Form.Item name="workflow_id" label="单工作流 (旧版兼容)" tooltip="推荐使用上方多工作流绑定">
            <Select allowClear placeholder="选择单个工作流"
              options={workflows.map((w: any) => ({ value: w.id, label: `${w.name} (${w.steps?.length || 0} 步骤)` }))}
            />
          </Form.Item>
          <Form.Item name="tool_ids" label="绑定工具">
            <Select mode="multiple" allowClear placeholder="选择可调用的工具"
              options={tools.map((t: any) => ({ value: t.id, label: `${t.name} (${t.category})` }))}
            />
          </Form.Item>
          <Form.Item name="knowledge_source_ids" label="绑定知识源">
            <Select mode="multiple" allowClear placeholder="选择知识源"
              options={sources.map((s: any) => ({ value: s.domain, label: `${s.name} (${s.source_type}, ${s.domain})` }))}
            />
          </Form.Item>
          <Form.Item name="function_calling_enabled" label="启用函数调用" valuePropName="checked" initialValue={true}>
            <Switch />
          </Form.Item>
          <Form.Item name="max_tool_rounds" label="最大工具调用轮数" initialValue={5}>
            <InputNumber min={1} max={20} />
          </Form.Item>
          <Form.Item name="enabled" label="启用" valuePropName="checked" initialValue={true}>
            <Switch />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
}
