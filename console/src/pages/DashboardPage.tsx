import { useEffect, useState, useCallback } from 'react';
import { Card, Col, Row, Statistic, Spin, Button, Alert, Tag, Table, Typography } from 'antd';
import {
  MessageOutlined,
  SearchOutlined,
  ApiOutlined,
  UserSwitchOutlined,
  RobotOutlined,
  ReloadOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { useTranslation } from 'react-i18next';
import { auditApi, agentApi, performanceApi } from '../api';
import type { AuditMetrics } from '../types';

// Generate sample trend data (last 24 hours) since historical API isn't implemented yet
function generateTrendData() {
  const data = [];
  const now = new Date();
  for (let i = 23; i >= 0; i--) {
    const hour = new Date(now.getTime() - i * 3600000);
    data.push({
      time: `${hour.getHours().toString().padStart(2, '0')}:00`,
      requests: Math.floor(Math.random() * 100) + 20,
      latency: Math.floor(Math.random() * 300) + 100,
      errors: Math.floor(Math.random() * 5),
    });
  }
  return data;
}

interface CircuitStatus {
  service: string;
  state: 'closed' | 'open' | 'half_open';
  failures: number;
  successes: number;
}

export default function DashboardPage() {
  const { t } = useTranslation();
  const [metrics, setMetrics] = useState<AuditMetrics | null>(null);
  const [agentCount, setAgentCount] = useState(0);
  const [circuits, setCircuits] = useState<CircuitStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [trendData] = useState(generateTrendData);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [metricsRes, agentsRes, circuitRes] = await Promise.all([
        auditApi.getMetrics('default', 24),
        agentApi.list(),
        performanceApi.getCircuitBreakerStatus().catch(() => ({ data: { circuits: [] } })),
      ]);
      setMetrics(metricsRes.data);
      setAgentCount(Array.isArray(agentsRes.data) ? agentsRes.data.length : 0);
      setCircuits(circuitRes.data?.circuits || []);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Unknown error';
      setError(`${t('common.error')}: ${msg}`);
    } finally {
      setLoading(false);
    }
  }, [t]);

  useEffect(() => {
    loadData();
    const timer = setInterval(loadData, 60000);
    return () => clearInterval(timer);
  }, [loadData]);

  if (loading && !metrics) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', padding: 48 }}>
        <Spin size="large" />
      </div>
    );
  }

  const m = metrics || {} as Record<string, number>;

  const circuitColumns = [
    { title: 'Service', dataIndex: 'service', key: 'service' },
    {
      title: 'State',
      dataIndex: 'state',
      key: 'state',
      render: (state: string) => {
        const config: Record<string, { color: string; icon: React.ReactNode }> = {
          closed: { color: 'success', icon: <CheckCircleOutlined /> },
          open: { color: 'error', icon: <CloseCircleOutlined /> },
          half_open: { color: 'warning', icon: <ExclamationCircleOutlined /> },
        };
        const c = config[state] || config.closed;
        return <Tag color={c.color} icon={c.icon}>{state.toUpperCase()}</Tag>;
      },
    },
    { title: 'Failures', dataIndex: 'failures', key: 'failures' },
    { title: 'Successes', dataIndex: 'successes', key: 'successes' },
  ];

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <h2>{t('dashboard.title')} ({t('dashboard.last24h')})</h2>
        <Button icon={<ReloadOutlined />} onClick={loadData} loading={loading}>
          {t('common.refresh')}
        </Button>
      </div>

      {error && <Alert message={error} type="error" style={{ marginBottom: 16 }} closable />}

      {/* Metric Cards */}
      <Row gutter={[16, 16]}>
        <Col xs={12} sm={8} md={4}>
          <Card size="small">
            <Statistic title={t('dashboard.activeAgents')} value={agentCount} prefix={<RobotOutlined />} />
          </Card>
        </Col>
        <Col xs={12} sm={8} md={5}>
          <Card size="small">
            <Statistic title={t('dashboard.totalRequests')} value={(m as any).total_invocations || 0} prefix={<MessageOutlined />} />
          </Card>
        </Col>
        <Col xs={12} sm={8} md={5}>
          <Card size="small">
            <Statistic title={t('dashboard.retrieval')} value={(m as any).retrieval_count || 0} prefix={<SearchOutlined />} />
          </Card>
        </Col>
        <Col xs={12} sm={8} md={5}>
          <Card size="small">
            <Statistic title={t('dashboard.toolCalls')} value={(m as any).tool_call_count || 0} prefix={<ApiOutlined />} />
          </Card>
        </Col>
        <Col xs={12} sm={8} md={5}>
          <Card size="small">
            <Statistic title={t('dashboard.escalationRate')} value={(m as any).escalation_rate || 0} suffix="%" prefix={<UserSwitchOutlined />} />
          </Card>
        </Col>
      </Row>

      {/* Charts Row */}
      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} lg={16}>
          <Card title={t('dashboard.requestVolume')} size="small">
            <ResponsiveContainer width="100%" height={240}>
              <AreaChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#303030" />
                <XAxis dataKey="time" stroke="#999" fontSize={11} />
                <YAxis stroke="#999" fontSize={11} />
                <Tooltip
                  contentStyle={{ background: '#1a1a2e', border: '1px solid #333', borderRadius: 6 }}
                  labelStyle={{ color: '#999' }}
                />
                <Area type="monotone" dataKey="requests" stroke="#1890ff" fill="#1890ff" fillOpacity={0.2} name="Requests" />
                <Area type="monotone" dataKey="errors" stroke="#ff4d4f" fill="#ff4d4f" fillOpacity={0.2} name="Errors" />
              </AreaChart>
            </ResponsiveContainer>
          </Card>
        </Col>
        <Col xs={24} lg={8}>
          <Card title={t('dashboard.latencyChart')} size="small">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#303030" />
                <XAxis dataKey="time" stroke="#999" fontSize={10} interval={2} />
                <YAxis stroke="#999" fontSize={11} />
                <Tooltip
                  contentStyle={{ background: '#1a1a2e', border: '1px solid #333', borderRadius: 6 }}
                  labelStyle={{ color: '#999' }}
                />
                <Bar dataKey="latency" fill="#722ed1" name="Latency" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </Col>
      </Row>

      {/* Latency Cards + Circuit Breaker */}
      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={12} md={6}>
          <Card size="small">
            <Statistic title={t('dashboard.avgLatency')} value={(m as any).avg_retrieval_latency_ms || 0} suffix="ms" precision={1} />
          </Card>
        </Col>
        <Col xs={12} md={6}>
          <Card size="small">
            <Statistic title={t('dashboard.avgLLMLatency')} value={(m as any).avg_llm_latency_ms || 0} suffix="ms" precision={1} />
          </Card>
        </Col>
        <Col xs={24} md={12}>
          <Card
            title={<><ThunderboltOutlined /> {t('dashboard.circuitBreaker')}</>}
            size="small"
          >
            {circuits.length === 0 ? (
              <Typography.Text type="secondary">{t('dashboard.noCircuits')}</Typography.Text>
            ) : (
              <Table
                dataSource={circuits}
                columns={circuitColumns}
                size="small"
                pagination={false}
                rowKey="service"
              />
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
}
