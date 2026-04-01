import { useEffect, useState, useCallback } from 'react';
import { Card, Row, Col, Tag, Typography, Spin, Button, Descriptions } from 'antd';
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  ExclamationCircleOutlined,
  ReloadOutlined,
  DatabaseOutlined,
  CloudServerOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { healthApi } from '../api';

const { Title, Text } = Typography;

interface ComponentHealth {
  status: string;
  error?: string;
  count?: number;
  total?: number;
  open?: number;
}

interface HealthStatus {
  status: string;
  version: string;
  components: Record<string, ComponentHealth>;
}

const STATUS_CONFIG: Record<string, { color: string; icon: React.ReactNode }> = {
  healthy: { color: 'success', icon: <CheckCircleOutlined /> },
  unhealthy: { color: 'error', icon: <CloseCircleOutlined /> },
  degraded: { color: 'warning', icon: <ExclamationCircleOutlined /> },
  not_initialized: { color: 'default', icon: <ExclamationCircleOutlined /> },
};

const COMPONENT_ICONS: Record<string, React.ReactNode> = {
  database: <DatabaseOutlined />,
  vector_store: <CloudServerOutlined />,
  circuit_breakers: <ThunderboltOutlined />,
};

export default function HealthPage() {
  const { t } = useTranslation();
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadHealth = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await healthApi.check();
      setHealth(res.data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Health check failed');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadHealth();
    const timer = setInterval(loadHealth, 30000);
    return () => clearInterval(timer);
  }, [loadHealth]);

  if (loading && !health) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', padding: 48 }}>
        <Spin size="large" />
      </div>
    );
  }

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <Title level={3}>System Health</Title>
        <Button icon={<ReloadOutlined />} onClick={loadHealth} loading={loading}>
          {t('common.refresh')}
        </Button>
      </div>

      {error && (
        <Card style={{ marginBottom: 16 }}>
          <Tag color="error">Unreachable</Tag> {error}
        </Card>
      )}

      {health && (
        <>
          <Card style={{ marginBottom: 16 }}>
            <Descriptions column={3}>
              <Descriptions.Item label="Overall Status">
                <Tag
                  color={health.status === 'ok' ? 'success' : 'warning'}
                  icon={health.status === 'ok' ? <CheckCircleOutlined /> : <ExclamationCircleOutlined />}
                >
                  {health.status.toUpperCase()}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Version">{health.version}</Descriptions.Item>
              <Descriptions.Item label="Components">{Object.keys(health.components).length}</Descriptions.Item>
            </Descriptions>
          </Card>

          <Row gutter={[16, 16]}>
            {Object.entries(health.components).map(([name, comp]) => {
              const cfg = STATUS_CONFIG[comp.status] || STATUS_CONFIG.healthy;
              return (
                <Col xs={24} sm={12} md={8} key={name}>
                  <Card
                    title={
                      <span>
                        {COMPONENT_ICONS[name] || <CloudServerOutlined />}{' '}
                        {name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                      </span>
                    }
                    extra={<Tag color={cfg.color} icon={cfg.icon}>{comp.status}</Tag>}
                  >
                    {comp.error && <Text type="danger" style={{ fontSize: 12 }}>{comp.error}</Text>}
                    {comp.count != null && <div><Text type="secondary">Vectors: {comp.count}</Text></div>}
                    {comp.total != null && (
                      <div><Text type="secondary">Circuits: {comp.total} (open: {comp.open || 0})</Text></div>
                    )}
                    {!comp.error && comp.count == null && comp.total == null && (
                      <Text type="secondary">Running normally</Text>
                    )}
                  </Card>
                </Col>
              );
            })}
          </Row>
        </>
      )}
    </div>
  );
}
