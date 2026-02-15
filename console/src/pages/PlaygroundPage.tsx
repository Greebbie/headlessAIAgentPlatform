import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  Button,
  Collapse,
  Empty,
  Input,
  List,
  message,
  Select,
  Space,
  Spin,
  Tag,
  Timeline,
  Tooltip,
  Typography,
  Upload,
} from 'antd';
import {
  ClearOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  RobotOutlined,
  SendOutlined,
  UploadOutlined,
  UserOutlined,
} from '@ant-design/icons';
import { agentApi, invokeApi, auditApi } from '../api';

const { TextArea } = Input;
const { Text, Paragraph } = Typography;

/* ── Types ────────────────────────────────────────────── */

interface WorkflowCardData {
  step_name: string;
  step_type: string;
  prompt: string;
  fields?: Array<{
    name: string;
    label: string;
    field_type?: string;
    required?: boolean;
    placeholder?: string;
    options?: Array<{ label: string; value: string }>;
    file_config?: { max_size_mb?: number; allowed_extensions?: string[] };
  }>;
  current_step: number;
  total_steps: number;
}

interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  citations?: Citation[];
  followups?: string[];
  traceId?: string;
  metadata?: Record<string, any>;
  isStreaming?: boolean;
  error?: boolean;
  workflowCard?: WorkflowCardData;
}

interface Citation {
  source_id: string;
  source_name: string;
  content_snippet: string;
  page?: number | null;
  paragraph?: number | null;
  line_start?: number | null;
  line_end?: number | null;
  score?: number | null;
}

interface TraceEvent {
  event_type: string;
  timestamp?: string;
  latency_ms?: number;
  event_data?: any;
  retrieval_hits?: any;
  llm_meta?: any;
  tool_meta?: any;
  trace_id?: string;
}

/* ── Constants ────────────────────────────────────────── */

const EVENT_COLORS: Record<string, string> = {
  user_input: 'blue',
  intent: 'cyan',
  retrieval: 'green',
  llm_call: 'purple',
  tool_call: 'orange',
  workflow_step: 'geekblue',
  response: 'blue',
  escalation: 'red',
  error: 'red',
  risk_block: 'red',
};

/* ── Component ────────────────────────────────────────── */

export default function PlaygroundPage() {
  /* state — agents */
  const [agents, setAgents] = useState<any[]>([]);
  const [selectedAgentId, setSelectedAgentId] = useState<string | undefined>(undefined);
  const [agentsLoading, setAgentsLoading] = useState(false);

  /* state — chat */
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [sending, setSending] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [useStreaming, setUseStreaming] = useState(true);

  /* state — trace panel */
  const [tracePanelOpen, setTracePanelOpen] = useState(true);
  const [traceEvents, setTraceEvents] = useState<TraceEvent[]>([]);
  const [traceLoading, setTraceLoading] = useState(false);
  const [activeCitations, setActiveCitations] = useState<Citation[]>([]);

  /* refs */
  const chatEndRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  /* ── Load agents on mount ──────────────────────────── */

  useEffect(() => {
    setAgentsLoading(true);
    agentApi
      .list()
      .then((res) => {
        const list = Array.isArray(res.data) ? res.data : [];
        setAgents(list);
        if (list.length > 0 && !selectedAgentId) {
          setSelectedAgentId(list[0].id);
        }
      })
      .catch(() => message.error('Failed to load agents'))
      .finally(() => setAgentsLoading(false));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  /* ── Auto scroll ───────────────────────────────────── */

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  /* ── Fetch trace for a given traceId ───────────────── */

  const loadTrace = useCallback(async (traceId: string) => {
    setTraceLoading(true);
    try {
      const res = await auditApi.getTrace(traceId);
      const data = Array.isArray(res.data) ? res.data : [res.data];
      setTraceEvents(data);
    } catch {
      message.error('Failed to load trace data');
    } finally {
      setTraceLoading(false);
    }
  }, []);

  /* ── New Session ───────────────────────────────────── */

  const handleNewSession = () => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    setMessages([]);
    setSessionId(null);
    setTraceEvents([]);
    setActiveCitations([]);
    setSending(false);
  };

  /* ── Send via SSE streaming ────────────────────────── */

  const sendStreaming = async (userMessage: string) => {
    const controller = new AbortController();
    abortRef.current = controller;

    /* Append a placeholder assistant message for streaming */
    const streamingMsg: ChatMessage = {
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
      isStreaming: true,
    };
    setMessages((prev) => [...prev, streamingMsg]);

    try {
      const body: Record<string, any> = {
        agent_id: selectedAgentId!,
        message: userMessage,
        tenant_id: 'default',
      };
      if (sessionId) body.session_id = sessionId;

      const response = await fetch('/api/v1/invoke/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('ReadableStream not supported');

      const decoder = new TextDecoder();
      let buffer = '';
      let currentEvent = '';
      let currentData = '';
      let finalContent = '';
      let finalCitations: Citation[] = [];
      let finalFollowups: string[] = [];
      let finalTraceId: string | undefined;
      let finalMetadata: Record<string, any> | undefined;
      let finalWorkflowCard: WorkflowCardData | undefined;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        /* Keep the last incomplete line in the buffer */
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim();
            currentData = '';
          } else if (line.startsWith('data: ')) {
            currentData = line.slice(6);
          } else if (line === '' && currentEvent && currentData) {
            /* Complete SSE event pair received */
            try {
              const parsed = JSON.parse(currentData);

              if (currentEvent === 'answer') {
                finalContent = parsed.content || '';
                setMessages((prev) => {
                  const updated = [...prev];
                  const last = updated[updated.length - 1];
                  if (last && last.isStreaming) {
                    updated[updated.length - 1] = { ...last, content: finalContent };
                  }
                  return updated;
                });
              } else if (currentEvent === 'status') {
                /* Update streaming message with status */
                const stage = parsed.stage || 'processing';
                setMessages((prev) => {
                  const updated = [...prev];
                  const last = updated[updated.length - 1];
                  if (last && last.isStreaming && !last.content) {
                    const stageLabel =
                      stage === 'retrieval' ? 'Retrieving knowledge...' : 'Processing...';
                    updated[updated.length - 1] = { ...last, content: stageLabel };
                  }
                  return updated;
                });
              } else if (currentEvent === 'done') {
                finalTraceId = parsed.trace_id;
                finalCitations = parsed.citations || [];
                finalFollowups = parsed.followups || [];
                finalMetadata = parsed.metadata;
                finalWorkflowCard = parsed.workflow_card;
                if (parsed.session_id) setSessionId(parsed.session_id);
              } else if (currentEvent === 'error') {
                const errorMsg = parsed.error_msg || parsed.detail || 'Unknown error';
                setMessages((prev) => {
                  const updated = [...prev];
                  const last = updated[updated.length - 1];
                  if (last && last.isStreaming) {
                    updated[updated.length - 1] = {
                      ...last,
                      content: errorMsg,
                      isStreaming: false,
                      error: true,
                    };
                  }
                  return updated;
                });
                message.error(errorMsg);
              }
            } catch {
              /* ignore malformed JSON */
            }
            currentEvent = '';
            currentData = '';
          }
        }
      }

      /* Finalize the streaming message */
      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last && last.isStreaming) {
          updated[updated.length - 1] = {
            ...last,
            content: finalContent || last.content || 'No response received.',
            isStreaming: false,
            citations: finalCitations,
            followups: finalFollowups,
            traceId: finalTraceId,
            metadata: finalMetadata,
            workflowCard: finalWorkflowCard,
          };
        }
        return updated;
      });

      /* Show citations in panel */
      if (finalCitations.length > 0) setActiveCitations(finalCitations);

      /* Auto-load trace */
      if (finalTraceId) loadTrace(finalTraceId);
    } catch (err: any) {
      if (err.name === 'AbortError') return;
      const errorText = err.message || 'Streaming request failed';
      message.error(errorText);
      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last && last.isStreaming) {
          updated[updated.length - 1] = {
            ...last,
            content: errorText,
            isStreaming: false,
            error: true,
          };
        }
        return updated;
      });
    } finally {
      abortRef.current = null;
    }
  };

  /* ── Send via sync invoke (fallback) ───────────────── */

  const sendSync = async (userMessage: string) => {
    try {
      const body: Record<string, any> = {
        agent_id: selectedAgentId!,
        message: userMessage,
        tenant_id: 'default',
      };
      if (sessionId) body.session_id = sessionId;

      const res = await invokeApi.send(body);
      const data = res.data;

      setSessionId(data.session_id);

      const assistantMsg: ChatMessage = {
        role: 'assistant',
        content: data.short_answer || '',
        timestamp: Date.now(),
        citations: data.citations || [],
        followups: data.suggested_followups || [],
        traceId: data.trace_id,
        metadata: data.metadata,
        workflowCard: data.workflow_card,
      };
      setMessages((prev) => [...prev, assistantMsg]);

      if (data.citations?.length > 0) setActiveCitations(data.citations);
      if (data.trace_id) loadTrace(data.trace_id);
    } catch (err: any) {
      const errorText =
        err.response?.data?.detail || err.message || 'Request failed';
      message.error(errorText);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: errorText,
          timestamp: Date.now(),
          error: true,
        },
      ]);
    }
  };

  /* ── Send handler ──────────────────────────────────── */

  const handleSend = async () => {
    const text = inputValue.trim();
    if (!text) return;
    if (!selectedAgentId) {
      message.warning('Please select an agent first');
      return;
    }

    /* Append user message */
    const userMsg: ChatMessage = {
      role: 'user',
      content: text,
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setInputValue('');
    setSending(true);

    try {
      if (useStreaming) {
        await sendStreaming(text);
      } else {
        await sendSync(text);
      }
    } finally {
      setSending(false);
    }
  };

  /* ── Handle followup click ─────────────────────────── */

  const handleFollowup = (question: string) => {
    setInputValue(question);
  };

  /* ── Handle enter key ──────────────────────────────── */

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!sending) handleSend();
    }
  };

  /* ── Render ────────────────────────────────────────── */

  const selectedAgent = agents.find((a) => a.id === selectedAgentId);

  return (
    <div style={{ display: 'flex', height: 'calc(100vh - 136px)', gap: 0 }}>
      {/* ── Main Chat Area ────────────────────────────── */}
      <div
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          minWidth: 0,
        }}
      >
        {/* Top Bar */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0 0 12px 0',
            borderBottom: '1px solid #f0f0f0',
            marginBottom: 12,
            flexWrap: 'wrap',
            gap: 8,
          }}
        >
          <Space wrap>
            <Select
              placeholder="Select Agent"
              value={selectedAgentId}
              onChange={(v) => {
                setSelectedAgentId(v);
                handleNewSession();
              }}
              loading={agentsLoading}
              style={{ minWidth: 240 }}
              optionLabelProp="label"
              showSearch
              filterOption={(input, option) =>
                (option?.label as string)?.toLowerCase().includes(input.toLowerCase()) ?? false
              }
            >
              {agents.map((a) => (
                <Select.Option key={a.id} value={a.id} label={a.name}>
                  <Space>
                    <RobotOutlined />
                    <span>{a.name}</span>
                    {a.enabled === false && <Tag color="red">Disabled</Tag>}
                  </Space>
                </Select.Option>
              ))}
            </Select>
            {selectedAgent && (
              <Text type="secondary" style={{ fontSize: 12 }}>
                {selectedAgent.description || 'No description'}
              </Text>
            )}
          </Space>
          <Space>
            <Tooltip title={useStreaming ? 'Streaming enabled' : 'Sync mode'}>
              <Button
                size="small"
                type={useStreaming ? 'primary' : 'default'}
                onClick={() => setUseStreaming(!useStreaming)}
              >
                {useStreaming ? 'SSE' : 'Sync'}
              </Button>
            </Tooltip>
            <Button icon={<ClearOutlined />} onClick={handleNewSession}>
              New Session
            </Button>
            {sessionId && (
              <Text copyable={{ text: sessionId }} type="secondary" style={{ fontSize: 11 }}>
                Session: {sessionId.slice(0, 8)}...
              </Text>
            )}
            <Button
              icon={tracePanelOpen ? <MenuFoldOutlined /> : <MenuUnfoldOutlined />}
              onClick={() => setTracePanelOpen(!tracePanelOpen)}
              title={tracePanelOpen ? 'Hide trace panel' : 'Show trace panel'}
            />
          </Space>
        </div>

        {/* Chat Messages */}
        <div
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: '8px 0',
            display: 'flex',
            flexDirection: 'column',
            gap: 12,
          }}
        >
          {messages.length === 0 && (
            <Empty
              description={
                selectedAgentId
                  ? 'Send a message to start chatting'
                  : 'Select an agent to begin'
              }
              style={{ margin: 'auto' }}
            />
          )}

          {messages.map((msg, idx) => (
            <div
              key={idx}
              style={{
                display: 'flex',
                justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
                padding: '0 8px',
              }}
            >
              <div
                style={{
                  maxWidth: '75%',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start',
                }}
              >
                {/* Avatar + role label */}
                <Space style={{ marginBottom: 4 }}>
                  {msg.role === 'user' ? (
                    <Tag icon={<UserOutlined />} color="blue">
                      You
                    </Tag>
                  ) : (
                    <Tag icon={<RobotOutlined />} color="green">
                      Assistant
                    </Tag>
                  )}
                  <Text type="secondary" style={{ fontSize: 11 }}>
                    {new Date(msg.timestamp).toLocaleTimeString()}
                  </Text>
                </Space>

                {/* Message bubble */}
                <div
                  style={{
                    padding: '10px 14px',
                    borderRadius: msg.role === 'user' ? '12px 12px 2px 12px' : '12px 12px 12px 2px',
                    background: msg.error
                      ? '#fff2f0'
                      : msg.role === 'user'
                      ? '#1677ff'
                      : '#f5f5f5',
                    color: msg.error ? '#ff4d4f' : msg.role === 'user' ? '#fff' : '#333',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                    lineHeight: 1.6,
                    fontSize: 14,
                    border: msg.error ? '1px solid #ffccc7' : 'none',
                  }}
                >
                  {msg.isStreaming && !msg.content ? (
                    <Spin size="small" />
                  ) : (
                    msg.content
                  )}
                  {msg.isStreaming && msg.content && (
                    <Spin size="small" style={{ marginLeft: 8 }} />
                  )}
                </div>

                {/* Workflow Card with file upload */}
                {msg.workflowCard && msg.workflowCard.fields && msg.workflowCard.fields.length > 0 && (
                  <div
                    style={{
                      marginTop: 8,
                      padding: '12px 16px',
                      background: '#f6ffed',
                      border: '1px solid #b7eb8f',
                      borderRadius: 8,
                      maxWidth: '100%',
                    }}
                  >
                    <div style={{ marginBottom: 8 }}>
                      <Tag color="green">
                        步骤 {msg.workflowCard.current_step}/{msg.workflowCard.total_steps}
                      </Tag>
                      <Text strong>{msg.workflowCard.step_name}</Text>
                    </div>
                    {msg.workflowCard.fields.map((field) => (
                      <div key={field.name} style={{ marginBottom: 6 }}>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          {field.label}
                          {field.required && <span style={{ color: 'red' }}> *</span>}
                        </Text>
                        {field.field_type === 'file' ? (
                          <div style={{ marginTop: 4 }}>
                            <Upload
                              beforeUpload={() => false}
                              maxCount={1}
                              accept={field.file_config?.allowed_extensions?.map(
                                (ext: string) => ext.startsWith('.') ? ext : `.${ext}`
                              ).join(',')}
                            >
                              <Button icon={<UploadOutlined />} size="small">
                                上传文件
                                {field.file_config?.max_size_mb && (
                                  <span style={{ fontSize: 11, color: '#999' }}>
                                    {' '}(最大 {field.file_config.max_size_mb}MB)
                                  </span>
                                )}
                              </Button>
                            </Upload>
                            {field.file_config?.allowed_extensions && (
                              <Text type="secondary" style={{ fontSize: 11 }}>
                                支持格式: {field.file_config.allowed_extensions.join(', ')}
                              </Text>
                            )}
                          </div>
                        ) : (
                          <div style={{ fontSize: 12, color: '#666' }}>
                            {field.placeholder || `请输入${field.label}`}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {/* Citations */}
                {msg.citations && msg.citations.length > 0 && (
                  <div style={{ marginTop: 6 }}>
                    {msg.citations.map((c, ci) => (
                      <Tag
                        key={ci}
                        color="geekblue"
                        style={{ cursor: 'pointer', marginBottom: 2 }}
                        onClick={() => setActiveCitations(msg.citations!)}
                      >
                        [{ci + 1}] {c.source_name}
                        {c.score != null && ` (${(c.score * 100).toFixed(0)}%)`}
                      </Tag>
                    ))}
                  </div>
                )}

                {/* Followup suggestions */}
                {msg.followups && msg.followups.length > 0 && (
                  <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {msg.followups.map((q, qi) => (
                      <Tag
                        key={qi}
                        color="default"
                        style={{ cursor: 'pointer', borderStyle: 'dashed' }}
                        onClick={() => handleFollowup(q)}
                      >
                        {q}
                      </Tag>
                    ))}
                  </div>
                )}

                {/* Trace link */}
                {msg.traceId && (
                  <Button
                    type="link"
                    size="small"
                    style={{ padding: 0, marginTop: 4, fontSize: 11 }}
                    onClick={() => loadTrace(msg.traceId!)}
                  >
                    View Trace: {msg.traceId.slice(0, 8)}...
                  </Button>
                )}
              </div>
            </div>
          ))}

          <div ref={chatEndRef} />
        </div>

        {/* Input Area */}
        <div
          style={{
            borderTop: '1px solid #f0f0f0',
            paddingTop: 12,
            display: 'flex',
            gap: 8,
            alignItems: 'flex-end',
          }}
        >
          <TextArea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              selectedAgentId ? 'Type a message... (Enter to send, Shift+Enter for new line)' : 'Select an agent first'
            }
            disabled={!selectedAgentId || sending}
            autoSize={{ minRows: 1, maxRows: 4 }}
            style={{ flex: 1 }}
          />
          <Button
            type="primary"
            icon={<SendOutlined />}
            onClick={handleSend}
            loading={sending}
            disabled={!selectedAgentId || !inputValue.trim()}
          >
            Send
          </Button>
        </div>
      </div>

      {/* ── Trace / Citations Panel (Collapsible) ─────── */}
      {tracePanelOpen && (
        <div
          style={{
            width: 380,
            borderLeft: '1px solid #f0f0f0',
            paddingLeft: 16,
            marginLeft: 16,
            overflowY: 'auto',
            flexShrink: 0,
          }}
        >
          <Collapse
            defaultActiveKey={['citations', 'trace']}
            ghost
            items={[
              {
                key: 'citations',
                label: (
                  <Space>
                    <Text strong>Citations</Text>
                    {activeCitations.length > 0 && (
                      <Tag color="blue">{activeCitations.length}</Tag>
                    )}
                  </Space>
                ),
                children:
                  activeCitations.length === 0 ? (
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      No citations yet. Send a message to see retrieval results.
                    </Text>
                  ) : (
                    <List
                      size="small"
                      dataSource={activeCitations}
                      renderItem={(c, i) => (
                        <List.Item style={{ padding: '6px 0' }}>
                          <div style={{ width: '100%' }}>
                            <Space style={{ marginBottom: 4 }}>
                              <Tag color="geekblue">[{i + 1}]</Tag>
                              <Text strong style={{ fontSize: 12 }}>
                                {c.source_name}
                              </Text>
                              {c.score != null && (
                                <Tag color={c.score > 0.8 ? 'green' : c.score > 0.5 ? 'orange' : 'red'}>
                                  {(c.score * 100).toFixed(0)}%
                                </Tag>
                              )}
                            </Space>
                            <Paragraph
                              ellipsis={{ rows: 3, expandable: true, symbol: 'more' }}
                              style={{
                                fontSize: 12,
                                background: '#f5f5f5',
                                padding: '6px 8px',
                                borderRadius: 4,
                                margin: 0,
                              }}
                            >
                              {c.content_snippet}
                            </Paragraph>
                            {(c.page != null || c.line_start != null) && (
                              <Text type="secondary" style={{ fontSize: 11 }}>
                                {c.page != null && `Page ${c.page}`}
                                {c.page != null && c.line_start != null && ' | '}
                                {c.line_start != null && `Lines ${c.line_start}-${c.line_end ?? '?'}`}
                              </Text>
                            )}
                          </div>
                        </List.Item>
                      )}
                    />
                  ),
              },
              {
                key: 'trace',
                label: (
                  <Space>
                    <Text strong>Call Trace</Text>
                    {traceEvents.length > 0 && (
                      <Tag color="purple">{traceEvents.length}</Tag>
                    )}
                    {traceLoading && <Spin size="small" />}
                  </Space>
                ),
                children:
                  traceEvents.length === 0 ? (
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      No trace data. Trace will appear after a response is received.
                    </Text>
                  ) : (
                    <Timeline
                      items={traceEvents.map((t) => ({
                        color: EVENT_COLORS[t.event_type] || 'gray',
                        children: (
                          <div style={{ marginBottom: 8 }}>
                            <Space style={{ marginBottom: 4 }}>
                              <Tag color={EVENT_COLORS[t.event_type] || 'default'} style={{ fontSize: 11 }}>
                                {t.event_type}
                              </Tag>
                              {t.latency_ms != null && (
                                <Text type="secondary" style={{ fontSize: 11 }}>
                                  {t.latency_ms.toFixed(1)}ms
                                </Text>
                              )}
                            </Space>
                            {t.event_data && (
                              <pre
                                style={{
                                  fontSize: 11,
                                  maxHeight: 120,
                                  overflow: 'auto',
                                  background: '#f5f5f5',
                                  padding: 6,
                                  borderRadius: 4,
                                  margin: '4px 0 0 0',
                                  whiteSpace: 'pre-wrap',
                                  wordBreak: 'break-all',
                                }}
                              >
                                {JSON.stringify(t.event_data, null, 2)}
                              </pre>
                            )}
                            {t.retrieval_hits && (
                              <pre
                                style={{
                                  fontSize: 11,
                                  maxHeight: 120,
                                  overflow: 'auto',
                                  background: '#f0f5ff',
                                  padding: 6,
                                  borderRadius: 4,
                                  margin: '4px 0 0 0',
                                  whiteSpace: 'pre-wrap',
                                  wordBreak: 'break-all',
                                }}
                              >
                                {JSON.stringify(t.retrieval_hits, null, 2)}
                              </pre>
                            )}
                            {t.llm_meta && (
                              <pre
                                style={{
                                  fontSize: 11,
                                  maxHeight: 120,
                                  overflow: 'auto',
                                  background: '#f9f0ff',
                                  padding: 6,
                                  borderRadius: 4,
                                  margin: '4px 0 0 0',
                                  whiteSpace: 'pre-wrap',
                                  wordBreak: 'break-all',
                                }}
                              >
                                {JSON.stringify(t.llm_meta, null, 2)}
                              </pre>
                            )}
                            {t.tool_meta && (
                              <pre
                                style={{
                                  fontSize: 11,
                                  maxHeight: 120,
                                  overflow: 'auto',
                                  background: '#fff7e6',
                                  padding: 6,
                                  borderRadius: 4,
                                  margin: '4px 0 0 0',
                                  whiteSpace: 'pre-wrap',
                                  wordBreak: 'break-all',
                                }}
                              >
                                {JSON.stringify(t.tool_meta, null, 2)}
                              </pre>
                            )}
                          </div>
                        ),
                      }))}
                    />
                  ),
              },
            ]}
          />
        </div>
      )}
    </div>
  );
}
