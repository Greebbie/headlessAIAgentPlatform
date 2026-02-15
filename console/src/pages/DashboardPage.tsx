import React, { useEffect, useState, useCallback } from 'react';
import { Card, Col, Row, Statistic, Spin, message, Button, Alert } from 'antd';
import {
  MessageOutlined,
  SearchOutlined,
  ApiOutlined,
  UserSwitchOutlined,
  RobotOutlined,
  ReloadOutlined,
} from '@ant-design/icons';
import { auditApi, agentApi } from '../api';

export default function DashboardPage() {
  const [metrics, setMetrics] = useState<any>(null);
  const [agentCount, setAgentCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [metricsRes, agentsRes] = await Promise.all([
        auditApi.getMetrics('default', 24),
        agentApi.list(),
      ]);
      setMetrics(metricsRes.data);
      setAgentCount(agentsRes.data?.length || 0);
    } catch (e: any) {
      setError('加载指标数据失败: ' + (e.message || '未知错误'));
      message.error('加载仪表盘数据失败');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
    const timer = setInterval(loadData, 60000);
    return () => clearInterval(timer);
  }, [loadData]);

  if (loading && !metrics) return <Spin size="large" />;

  const m = metrics || {};

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <h2>运营仪表盘 (最近24小时)</h2>
        <Button icon={<ReloadOutlined />} onClick={loadData} loading={loading}>刷新</Button>
      </div>

      {error && <Alert message={error} type="error" style={{ marginBottom: 16 }} closable />}

      <Row gutter={16} style={{ marginTop: 8 }}>
        <Col span={4}>
          <Card>
            <Statistic title="Agent 数量" value={agentCount} prefix={<RobotOutlined />} />
          </Card>
        </Col>
        <Col span={5}>
          <Card>
            <Statistic title="总调用次数" value={m.total_invocations || 0} prefix={<MessageOutlined />} />
          </Card>
        </Col>
        <Col span={5}>
          <Card>
            <Statistic title="检索次数" value={m.retrieval_count || 0} prefix={<SearchOutlined />} />
          </Card>
        </Col>
        <Col span={5}>
          <Card>
            <Statistic title="工具调用" value={m.tool_call_count || 0} prefix={<ApiOutlined />} />
          </Card>
        </Col>
        <Col span={5}>
          <Card>
            <Statistic title="转人工率" value={m.escalation_rate || 0} suffix="%" prefix={<UserSwitchOutlined />} />
          </Card>
        </Col>
      </Row>
      <Row gutter={16} style={{ marginTop: 16 }}>
        <Col span={8}>
          <Card>
            <Statistic title="平均检索延迟" value={m.avg_retrieval_latency_ms || 0} suffix="ms" precision={1} />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic title="平均LLM延迟" value={m.avg_llm_latency_ms || 0} suffix="ms" precision={1} />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic title="风控拦截" value={m.risk_block_count || 0} />
          </Card>
        </Col>
      </Row>
    </div>
  );
}
