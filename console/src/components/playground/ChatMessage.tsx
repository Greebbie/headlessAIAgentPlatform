import React from 'react';
import { Button, Space, Spin, Tag, Typography } from 'antd';
import { RobotOutlined, UserOutlined } from '@ant-design/icons';

const { Text } = Typography;

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

interface SkillInfo {
  skill_id?: string;
  skill_name?: string;
  skill_type?: string;
  delegated_to?: string;
}

interface ChatMessageProps {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  citations?: Citation[];
  followups?: string[];
  traceId?: string;
  isStreaming?: boolean;
  error?: boolean;
  skillInfo?: SkillInfo | null;
  onFollowup?: (text: string) => void;
  onViewCitations?: (citations: Citation[]) => void;
  onViewTrace?: (traceId: string) => void;
  /** Slot for rendering workflow card below the message bubble */
  workflowCardSlot?: React.ReactNode;
}

export default function ChatMessage({
  role,
  content,
  timestamp,
  citations,
  followups,
  traceId,
  isStreaming,
  error,
  skillInfo,
  onFollowup,
  onViewCitations,
  onViewTrace,
  workflowCardSlot,
}: ChatMessageProps) {
  const isUser = role === 'user';

  return (
    <div
      style={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        padding: '0 8px',
      }}
    >
      <div
        style={{
          maxWidth: '75%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: isUser ? 'flex-end' : 'flex-start',
        }}
      >
        {/* Avatar + role label */}
        <Space style={{ marginBottom: 4 }}>
          {isUser ? (
            <Tag icon={<UserOutlined />} color="blue">
              You
            </Tag>
          ) : (
            <Tag icon={<RobotOutlined />} color="green">
              Assistant
            </Tag>
          )}
          <Text type="secondary" style={{ fontSize: 11 }}>
            {new Date(timestamp).toLocaleTimeString()}
          </Text>
        </Space>

        {/* Message bubble */}
        <div
          style={{
            padding: '10px 14px',
            borderRadius: isUser ? '12px 12px 2px 12px' : '12px 12px 12px 2px',
            background: error
              ? '#fff2f0'
              : isUser
              ? '#1677ff'
              : '#f5f5f5',
            color: error ? '#ff4d4f' : isUser ? '#fff' : '#333',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            lineHeight: 1.6,
            fontSize: 14,
            border: error ? '1px solid #ffccc7' : 'none',
          }}
        >
          {isStreaming && !content ? (
            <Spin size="small" />
          ) : (
            content
          )}
          {isStreaming && content && (
            <Spin size="small" style={{ marginLeft: 8 }} />
          )}
        </div>

        {/* Skill info badge */}
        {skillInfo && (
          <div style={{ marginTop: 4 }}>
            <Tag color="purple" style={{ fontSize: 11 }}>
              {skillInfo.skill_name || skillInfo.skill_type}
            </Tag>
            {skillInfo.delegated_to && (
              <Tag color="gold" style={{ fontSize: 11 }}>
                delegated
              </Tag>
            )}
          </div>
        )}

        {/* Workflow Card slot */}
        {workflowCardSlot}

        {/* Citations */}
        {citations && citations.length > 0 && (
          <div style={{ marginTop: 6 }}>
            {citations.map((c, ci) => (
              <Tag
                key={ci}
                color="geekblue"
                style={{ cursor: 'pointer', marginBottom: 2 }}
                onClick={() => onViewCitations?.(citations)}
              >
                [{ci + 1}] {c.source_name}
                {c.score != null && ` (${(c.score * 100).toFixed(0)}%)`}
              </Tag>
            ))}
          </div>
        )}

        {/* Followup suggestions */}
        {followups && followups.length > 0 && (
          <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {followups.map((q, qi) => (
              <Tag
                key={qi}
                color="default"
                style={{ cursor: 'pointer', borderStyle: 'dashed' }}
                onClick={() => onFollowup?.(q)}
              >
                {q}
              </Tag>
            ))}
          </div>
        )}

        {/* Trace link */}
        {traceId && (
          <Button
            type="link"
            size="small"
            style={{ padding: 0, marginTop: 4, fontSize: 11 }}
            onClick={() => onViewTrace?.(traceId)}
          >
            View Trace: {traceId.slice(0, 8)}...
          </Button>
        )}
      </div>
    </div>
  );
}
