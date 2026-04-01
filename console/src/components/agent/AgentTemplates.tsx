import React from 'react';
import { Card, Row, Col, Tag } from 'antd';
import {
  CustomerServiceOutlined,
  QuestionCircleOutlined,
  FormOutlined,
  ApartmentOutlined,
} from '@ant-design/icons';

export interface AgentTemplate {
  key: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  system_prompt: string;
  color: string;
  tags: string[];
}

export const TEMPLATES: AgentTemplate[] = [
  {
    key: 'customer_service',
    name: 'Customer Service',
    description: 'Handles customer inquiries with knowledge base and escalation',
    icon: <CustomerServiceOutlined />,
    system_prompt:
      'You are a professional customer service agent. Answer questions using the knowledge base. If you cannot find an answer, escalate to a human agent. Be polite, concise, and helpful.',
    color: '#1890ff',
    tags: ['Knowledge', 'Escalation'],
  },
  {
    key: 'faq_bot',
    name: 'FAQ Bot',
    description: 'Simple Q&A bot powered by knowledge retrieval',
    icon: <QuestionCircleOutlined />,
    system_prompt:
      "You are an FAQ assistant. Answer questions based on the provided knowledge base. If the answer is not in the knowledge base, politely say you don't have that information.",
    color: '#52c41a',
    tags: ['Knowledge', 'Simple'],
  },
  {
    key: 'data_collection',
    name: 'Data Collection',
    description: 'Collects structured data through guided workflows',
    icon: <FormOutlined />,
    system_prompt:
      'You are a data collection assistant. Guide users through the required workflow steps to collect their information. Be patient and helpful. Validate inputs when possible.',
    color: '#fa8c16',
    tags: ['Workflow', 'Forms'],
  },
  {
    key: 'multi_agent',
    name: 'Multi-Agent Router',
    description: 'Routes requests to specialized sub-agents',
    icon: <ApartmentOutlined />,
    system_prompt:
      'You are a routing agent. Analyze user requests and delegate them to the most appropriate specialized agent. Provide context when delegating.',
    color: '#722ed1',
    tags: ['Delegation', 'Advanced'],
  },
];

interface AgentTemplatesProps {
  onSelect: (template: AgentTemplate) => void;
}

export default function AgentTemplates({ onSelect }: AgentTemplatesProps) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 12, color: '#999', marginBottom: 8 }}>Quick Start: Select a template</div>
      <Row gutter={[8, 8]}>
        {TEMPLATES.map((t) => (
          <Col span={12} key={t.key}>
            <Card
              size="small"
              hoverable
              onClick={() => onSelect(t)}
              style={{ cursor: 'pointer' }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: 20, color: t.color }}>{t.icon}</span>
                <div>
                  <div style={{ fontWeight: 600, fontSize: 13 }}>{t.name}</div>
                  <div style={{ fontSize: 11, color: '#999' }}>{t.description}</div>
                </div>
              </div>
              <div style={{ marginTop: 4 }}>
                {t.tags.map((tag) => (
                  <Tag key={tag} style={{ fontSize: 10 }}>{tag}</Tag>
                ))}
              </div>
            </Card>
          </Col>
        ))}
      </Row>
    </div>
  );
}
