import React from 'react';
import { Collapse, List, Space, Spin, Tag, Timeline, Typography } from 'antd';

const { Text, Paragraph } = Typography;

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
  event_data?: Record<string, unknown>;
  retrieval_hits?: unknown;
  llm_meta?: unknown;
  tool_meta?: unknown;
  trace_id?: string;
}

interface TracePanelProps {
  visible: boolean;
  citations: Citation[];
  traceEvents: TraceEvent[];
  traceLoading: boolean;
}

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
  skill_route: 'magenta',
  skill_dispatch: 'volcano',
  delegation: 'gold',
  conversational_init: 'cyan',
  pre_retrieval: 'green',
  query_rewrite: 'lime',
};

const DATA_BLOCK_STYLES: Record<string, React.CSSProperties> = {
  event_data: {
    fontSize: 11,
    maxHeight: 120,
    overflow: 'auto',
    background: '#f5f5f5',
    padding: 6,
    borderRadius: 4,
    margin: '4px 0 0 0',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-all',
  },
  retrieval_hits: {
    fontSize: 11,
    maxHeight: 120,
    overflow: 'auto',
    background: '#f0f5ff',
    padding: 6,
    borderRadius: 4,
    margin: '4px 0 0 0',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-all',
  },
  llm_meta: {
    fontSize: 11,
    maxHeight: 120,
    overflow: 'auto',
    background: '#f9f0ff',
    padding: 6,
    borderRadius: 4,
    margin: '4px 0 0 0',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-all',
  },
  tool_meta: {
    fontSize: 11,
    maxHeight: 120,
    overflow: 'auto',
    background: '#fff7e6',
    padding: 6,
    borderRadius: 4,
    margin: '4px 0 0 0',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-all',
  },
};

function TraceDataBlock({ data, styleKey }: { data: unknown; styleKey: string }) {
  if (!data) return null;
  return (
    <pre style={DATA_BLOCK_STYLES[styleKey]}>
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}

export default function TracePanel({
  visible,
  citations,
  traceEvents,
  traceLoading,
}: TracePanelProps) {
  if (!visible) return null;

  return (
    <div
      style={{
        width: 340,
        minWidth: 280,
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
                {citations.length > 0 && (
                  <Tag color="blue">{citations.length}</Tag>
                )}
              </Space>
            ),
            children:
              citations.length === 0 ? (
                <Text type="secondary" style={{ fontSize: 12 }}>
                  No citations yet. Send a message to see retrieval results.
                </Text>
              ) : (
                <List
                  size="small"
                  dataSource={citations}
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
                        <TraceDataBlock data={t.event_data} styleKey="event_data" />
                        <TraceDataBlock data={t.retrieval_hits} styleKey="retrieval_hits" />
                        <TraceDataBlock data={t.llm_meta} styleKey="llm_meta" />
                        <TraceDataBlock data={t.tool_meta} styleKey="tool_meta" />
                      </div>
                    ),
                  }))}
                />
              ),
          },
        ]}
      />
    </div>
  );
}
