import React, { useEffect, useState } from 'react';
import { Table, Button, Modal, Form, Input, Select, InputNumber, Switch, Space, message, Tag, List, Popconfirm } from 'antd';
import { PlusOutlined, DeleteOutlined, EditOutlined } from '@ant-design/icons';
import { workflowApi, toolApi } from '../api';

const { TextArea } = Input;

const STEP_TYPES = [
  { value: 'collect', label: '收集信息' },
  { value: 'validate', label: '校验' },
  { value: 'tool_call', label: '工具调用' },
  { value: 'confirm', label: '用户确认' },
  { value: 'human_review', label: '人工审核' },
  { value: 'complete', label: '完成' },
];

const FAILURE_ACTIONS = [
  { value: 'retry', label: '重试' },
  { value: 'skip', label: '跳过' },
  { value: 'rollback', label: '回退' },
  { value: 'escalate', label: '转人工' },
];

export default function WorkflowsPage() {
  const [workflows, setWorkflows] = useState<any[]>([]);
  const [tools, setTools] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [stepModalOpen, setStepModalOpen] = useState(false);
  const [selectedWf, setSelectedWf] = useState<any>(null);
  const [editingStep, setEditingStep] = useState<any>(null);
  const [form] = Form.useForm();
  const [stepForm] = Form.useForm();

  const load = async () => {
    setLoading(true);
    try {
      const [wfRes, toolRes] = await Promise.all([
        workflowApi.list(),
        toolApi.list(),
      ]);
      setWorkflows(wfRes.data);
      setTools(toolRes.data);
    } catch (e: any) {
      message.error('加载数据失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const toolOptions = tools.map(t => ({ value: t.id, label: `${t.name} (${t.category})` }));

  const handleCreateWf = async () => {
    try {
      const values = await form.validateFields();
      await workflowApi.create(values);
      message.success('工作流已创建');
      setModalOpen(false);
      form.resetFields();
      load();
    } catch (e: any) {
      if (e.errorFields) return;
      message.error('创建失败: ' + (e.response?.data?.detail || e.message || '未知错误'));
    }
  };

  const handleDeleteWf = async (id: string) => {
    try {
      await workflowApi.delete(id);
      message.success('已删除');
      load();
    } catch (e: any) {
      message.error('删除失败: ' + (e.response?.data?.detail || e.message || '未知错误'));
    }
  };

  const handleSaveStep = async () => {
    if (!selectedWf) return;
    try {
      const values = await stepForm.validateFields();
      if (editingStep) {
        await workflowApi.updateStep(selectedWf.id, editingStep.id, values);
        message.success('步骤已更新');
      } else {
        await workflowApi.addStep(selectedWf.id, values);
        message.success('步骤已添加');
      }
      setStepModalOpen(false);
      setEditingStep(null);
      stepForm.resetFields();
      load();
    } catch (e: any) {
      if (e.errorFields) return;
      message.error('保存失败: ' + (e.response?.data?.detail || e.message || '未知错误'));
    }
  };

  const handleDeleteStep = async (wfId: string, stepId: string) => {
    try {
      await workflowApi.deleteStep(wfId, stepId);
      message.success('步骤已删除');
      load();
    } catch (e: any) {
      message.error('删除失败: ' + (e.response?.data?.detail || e.message || '未知错误'));
    }
  };

  const openEditStep = (wf: any, step: any) => {
    setSelectedWf(wf);
    setEditingStep(step);
    stepForm.setFieldsValue(step);
    setStepModalOpen(true);
  };

  const columns = [
    { title: '名称', dataIndex: 'name', key: 'name' },
    { title: '描述', dataIndex: 'description', key: 'description', ellipsis: true },
    { title: '步骤数', key: 'steps', render: (_: any, r: any) => r.steps?.length || 0 },
    { title: '版本', dataIndex: 'version', key: 'version' },
    {
      title: '操作', key: 'actions', render: (_: any, record: any) => (
        <Space>
          <Button icon={<PlusOutlined />} size="small" onClick={() => { setSelectedWf(record); setEditingStep(null); stepForm.resetFields(); setStepModalOpen(true); }}>
            添加步骤
          </Button>
          <Popconfirm title="确认删除此工作流及其所有步骤？" onConfirm={() => handleDeleteWf(record.id)} okText="确认" cancelText="取消">
            <Button icon={<DeleteOutlined />} size="small" danger>删除</Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <h2>流程编排</h2>
        <Button type="primary" icon={<PlusOutlined />} onClick={() => { form.resetFields(); setModalOpen(true); }}>
          创建工作流
        </Button>
      </div>

      <Table
        dataSource={workflows}
        columns={columns}
        rowKey="id"
        loading={loading}
        expandable={{
          expandedRowRender: (record) => (
            <List
              size="small"
              header={<strong>步骤列表</strong>}
              dataSource={record.steps || []}
              renderItem={(step: any) => (
                <List.Item
                  actions={[
                    <Button size="small" icon={<EditOutlined />} onClick={() => openEditStep(record, step)}>编辑</Button>,
                    <Popconfirm title="确认删除此步骤？" onConfirm={() => handleDeleteStep(record.id, step.id)} okText="确认" cancelText="取消">
                      <Button size="small" danger>删除</Button>
                    </Popconfirm>,
                  ]}
                >
                  <List.Item.Meta
                    avatar={<Tag color="blue">{step.order}</Tag>}
                    title={`${step.name} (${step.step_type})`}
                    description={step.prompt_template || '无提示语'}
                  />
                  <Space>
                    {step.requires_human_confirm && <Tag color="orange">需确认</Tag>}
                    {step.tool_id && <Tag color="purple">绑定工具</Tag>}
                    <Tag>{step.on_failure}</Tag>
                  </Space>
                </List.Item>
              )}
            />
          ),
        }}
      />

      {/* Create workflow modal */}
      <Modal title="创建工作流" open={modalOpen} onOk={handleCreateWf} onCancel={() => setModalOpen(false)}>
        <Form form={form} layout="vertical">
          <Form.Item name="name" label="名称" rules={[{ required: true }]}>
            <Input placeholder="如：居住证办理" />
          </Form.Item>
          <Form.Item name="description" label="描述">
            <TextArea rows={2} placeholder="描述流程用途" />
          </Form.Item>
        </Form>
      </Modal>

      {/* Add/Edit step modal */}
      <Modal
        title={editingStep ? `编辑步骤 - ${selectedWf?.name || ''}` : `添加步骤 - ${selectedWf?.name || ''}`}
        open={stepModalOpen}
        onOk={handleSaveStep}
        onCancel={() => { setStepModalOpen(false); setEditingStep(null); }}
        width={640}
        destroyOnClose
      >
        <Form form={stepForm} layout="vertical">
          <Form.Item name="name" label="步骤名称" rules={[{ required: true }]}>
            <Input placeholder="如：填写个人信息" />
          </Form.Item>
          <Form.Item name="order" label="顺序号" rules={[{ required: true }]}>
            <InputNumber min={0} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="step_type" label="步骤类型" initialValue="collect">
            <Select options={STEP_TYPES} />
          </Form.Item>
          <Form.Item name="prompt_template" label="提示语模板">
            <TextArea rows={3} placeholder="请填写您的姓名、身份证号..." />
          </Form.Item>
          <Form.Item name="tool_id" label="绑定工具">
            <Select allowClear placeholder="选择工具 (可选)" options={toolOptions} />
          </Form.Item>
          <Form.Item name="on_failure" label="失败策略" initialValue="retry">
            <Select options={FAILURE_ACTIONS} />
          </Form.Item>
          <Form.Item name="max_retries" label="最大重试次数" initialValue={2}>
            <InputNumber min={0} max={10} />
          </Form.Item>
          <Form.Item name="requires_human_confirm" label="需要人工确认" valuePropName="checked" initialValue={false}>
            <Switch />
          </Form.Item>
          <Form.Item name="risk_level" label="风险等级" initialValue="info">
            <Select options={[
              { value: 'info', label: '普通' },
              { value: 'warning', label: '警告' },
              { value: 'critical', label: '严重' },
            ]} />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
}
