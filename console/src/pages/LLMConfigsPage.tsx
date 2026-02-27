import { useEffect, useState } from 'react';
import {
  Table, Button, Modal, Form, Input, InputNumber, Select, Switch,
  Space, message, Tag, Card, Popconfirm, Descriptions,
} from 'antd';
import {
  PlusOutlined, EditOutlined, DeleteOutlined, ExperimentOutlined,
  CrownOutlined, DownloadOutlined,
} from '@ant-design/icons';
import { llmConfigApi } from '../api';

export default function LLMConfigsPage() {
  const [configs, setConfigs] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [editing, setEditing] = useState<any>(null);
  const [templates, setTemplates] = useState<Record<string, any>>({});
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<any>(null);
  const [form] = Form.useForm();

  const load = async () => {
    setLoading(true);
    try {
      const res = await llmConfigApi.list();
      setConfigs(res.data);
    } catch {
      message.error('Failed to load LLM configs');
    } finally {
      setLoading(false);
    }
  };

  const loadTemplates = async () => {
    try {
      const res = await llmConfigApi.getTemplates();
      setTemplates(res.data);
    } catch {
      message.error('Failed to load provider templates');
    }
  };

  useEffect(() => {
    load();
    loadTemplates();
  }, []);

  const openCreate = () => {
    setEditing(null);
    form.resetFields();
    setTestResult(null);
    setModalOpen(true);
  };

  const openEdit = (record: any) => {
    setEditing(record);
    form.setFieldsValue(record);
    setTestResult(null);
    setModalOpen(true);
  };

  const handleSave = async () => {
    try {
      const values = await form.validateFields();
      if (editing) {
        await llmConfigApi.update(editing.id, values);
        message.success('LLM configuration updated');
      } else {
        await llmConfigApi.create(values);
        message.success('LLM configuration created');
      }
      setModalOpen(false);
      setEditing(null);
      form.resetFields();
      setTestResult(null);
      load();
    } catch (err: any) {
      if (err?.errorFields) return; // form validation error, handled by antd
      message.error('Failed to save LLM configuration: ' + (err?.response?.data?.detail || err.message));
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await llmConfigApi.delete(id);
      message.success('LLM configuration deleted');
      load();
    } catch (err: any) {
      message.error('Failed to delete: ' + (err?.response?.data?.detail || err.message));
    }
  };

  const handleSetDefault = async (id: string) => {
    try {
      await llmConfigApi.setDefault(id);
      message.success('Default configuration updated');
      load();
    } catch (err: any) {
      message.error('Failed to set default: ' + (err?.response?.data?.detail || err.message));
    }
  };

  const handleLoadTemplate = () => {
    const provider = form.getFieldValue('provider');
    if (!provider) {
      message.warning('Please select a provider first');
      return;
    }
    // Map provider value to template key
    let templateKey = provider;
    if (provider === 'local') templateKey = 'ollama';
    if (provider === 'vllm') templateKey = 'vllm';
    const template = templates[templateKey] || templates[provider];
    if (!template) {
      message.warning(`No template found for provider "${provider}"`);
      return;
    }
    form.setFieldsValue({
      provider: template.provider,
      base_url: template.base_url,
      model: template.model,
      temperature: template.temperature,
      top_p: template.top_p,
      max_tokens: template.max_tokens,
      timeout_ms: template.timeout_ms,
    });
    message.success(`Template loaded for ${templateKey}`);
  };

  const handleTestConfig = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      const values = await form.validateFields(['base_url', 'model']);
      const formValues = form.getFieldsValue();
      const res = await llmConfigApi.test({
        base_url: formValues.base_url || values.base_url,
        api_key: formValues.api_key || '',
        model: formValues.model || values.model,
        temperature: formValues.temperature ?? 0.3,
        max_tokens: 256,
        timeout_ms: formValues.timeout_ms ?? 30000,
      });
      setTestResult(res.data);
      if (res.data.success) {
        message.success(`Test passed (${res.data.latency_ms?.toFixed(0)}ms)`);
      } else {
        message.error(`Test failed: ${res.data.error}`);
      }
    } catch (err: any) {
      if (err?.errorFields) {
        message.warning('Please fill in base_url and model before testing');
      } else {
        message.error('Test request failed: ' + (err?.response?.data?.detail || err.message));
      }
    } finally {
      setTesting(false);
    }
  };

  const providerColor = (provider: string) => {
    switch (provider) {
      case 'openai_compatible': return 'green';
      case 'dashscope': return 'blue';
      case 'zhipu': return 'purple';
      case 'local': return 'orange';
      default: return 'default';
    }
  };

  const columns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: any) => (
        <Space>
          {name}
          {record.is_default && <Tag color="gold" icon={<CrownOutlined />}>Default</Tag>}
        </Space>
      ),
    },
    {
      title: 'Provider',
      dataIndex: 'provider',
      key: 'provider',
      render: (v: string) => <Tag color={providerColor(v)}>{v}</Tag>,
    },
    { title: 'Model', dataIndex: 'model', key: 'model' },
    { title: 'Base URL', dataIndex: 'base_url', key: 'base_url', ellipsis: true },
    {
      title: 'Temperature',
      dataIndex: 'temperature',
      key: 'temperature',
      width: 100,
    },
    {
      title: 'Max Tokens',
      dataIndex: 'max_tokens',
      key: 'max_tokens',
      width: 100,
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 180,
      render: (v: string) => v ? new Date(v).toLocaleString() : '-',
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 280,
      render: (_: any, record: any) => (
        <Space>
          {!record.is_default && (
            <Button
              icon={<CrownOutlined />}
              size="small"
              onClick={() => handleSetDefault(record.id)}
            >
              Set Default
            </Button>
          )}
          <Button
            icon={<EditOutlined />}
            size="small"
            onClick={() => openEdit(record)}
          >
            Edit
          </Button>
          <Popconfirm
            title="Delete this LLM configuration?"
            description="This action cannot be undone."
            onConfirm={() => handleDelete(record.id)}
            okText="Delete"
            cancelText="Cancel"
            okButtonProps={{ danger: true }}
          >
            <Button icon={<DeleteOutlined />} size="small" danger>
              Delete
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <h2>LLM Configuration Management</h2>
        <Button type="primary" icon={<PlusOutlined />} onClick={openCreate}>
          Create Config
        </Button>
      </div>

      <Table
        dataSource={configs}
        columns={columns}
        rowKey="id"
        loading={loading}
        pagination={{ pageSize: 10 }}
      />

      <Modal
        title={editing ? 'Edit LLM Configuration' : 'Create LLM Configuration'}
        open={modalOpen}
        onOk={handleSave}
        onCancel={() => { setModalOpen(false); setTestResult(null); }}
        width={680}
        destroyOnClose
      >
        <Form form={form} layout="vertical" initialValues={{ temperature: 0.3, top_p: 1.0, max_tokens: 2048, timeout_ms: 60000, is_default: false }}>
          <Form.Item name="name" label="Name" rules={[{ required: true, message: 'Name is required' }]}>
            <Input placeholder="e.g., Production OpenAI Config" />
          </Form.Item>

          <Form.Item name="provider" label="Provider" rules={[{ required: true, message: 'Provider is required' }]}>
            <Select
              placeholder="Select a provider"
              options={[
                { value: 'openai_compatible', label: 'OpenAI Compatible' },
                { value: 'dashscope', label: 'DashScope (Alibaba)' },
                { value: 'zhipu', label: 'ZhipuAI (GLM)' },
                { value: 'local', label: 'Local (Ollama)' },
                { value: 'vllm', label: 'vLLM (MiniMax / DeepSeek / etc.)' },
              ]}
            />
          </Form.Item>

          <Space style={{ marginBottom: 16 }}>
            <Button icon={<DownloadOutlined />} onClick={handleLoadTemplate}>
              Load Template
            </Button>
            <Button icon={<ExperimentOutlined />} loading={testing} onClick={handleTestConfig}>
              Test Config
            </Button>
          </Space>

          {testResult && (
            <Card
              size="small"
              style={{ marginBottom: 16 }}
              title={testResult.success ? 'Test Passed' : 'Test Failed'}
              headStyle={{ background: testResult.success ? '#f6ffed' : '#fff2f0' }}
            >
              <Descriptions column={1} size="small">
                {testResult.success && (
                  <>
                    <Descriptions.Item label="Response">{testResult.content}</Descriptions.Item>
                    <Descriptions.Item label="Latency">{testResult.latency_ms?.toFixed(0)}ms</Descriptions.Item>
                  </>
                )}
                {!testResult.success && (
                  <Descriptions.Item label="Error">{testResult.error}</Descriptions.Item>
                )}
              </Descriptions>
            </Card>
          )}

          <Form.Item name="base_url" label="Base URL" rules={[{ required: true, message: 'Base URL is required' }]}>
            <Input placeholder="https://api.openai.com/v1" />
          </Form.Item>

          <Form.Item name="api_key" label="API Key">
            <Input.Password placeholder="sk-... (leave empty for local models)" />
          </Form.Item>

          <Form.Item name="model" label="Model" rules={[{ required: true, message: 'Model name is required' }]}>
            <Input placeholder="gpt-4o / qwen2.5 / glm-4" />
          </Form.Item>

          <Space size="large" wrap>
            <Form.Item name="temperature" label="Temperature">
              <InputNumber min={0} max={2} step={0.1} style={{ width: 120 }} />
            </Form.Item>

            <Form.Item name="top_p" label="Top P">
              <InputNumber min={0} max={1} step={0.05} style={{ width: 120 }} />
            </Form.Item>

            <Form.Item name="max_tokens" label="Max Tokens">
              <InputNumber min={1} max={128000} style={{ width: 140 }} />
            </Form.Item>

            <Form.Item name="timeout_ms" label="Timeout (ms)">
              <InputNumber min={1000} max={600000} step={1000} style={{ width: 140 }} />
            </Form.Item>
          </Space>

          <Form.Item name="is_default" label="Set as Default" valuePropName="checked">
            <Switch />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
}
