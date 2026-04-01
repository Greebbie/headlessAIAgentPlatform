import React from 'react';
import { Spin, Empty, Button, Result } from 'antd';
import { ReloadOutlined } from '@ant-design/icons';

interface LoadingStateProps {
  loading?: boolean;
  error?: string | null;
  empty?: boolean;
  emptyText?: string;
  emptyAction?: { label: string; onClick: () => void };
  onRetry?: () => void;
  children: React.ReactNode;
}

export default function LoadingState({
  loading,
  error,
  empty,
  emptyText = 'No data',
  emptyAction,
  onRetry,
  children,
}: LoadingStateProps) {
  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', padding: 48 }}>
        <Spin size="large" />
      </div>
    );
  }

  if (error) {
    return (
      <Result
        status="error"
        title="Error"
        subTitle={error}
        extra={onRetry && <Button icon={<ReloadOutlined />} onClick={onRetry}>Retry</Button>}
      />
    );
  }

  if (empty) {
    return (
      <Empty description={emptyText}>
        {emptyAction && (
          <Button type="primary" onClick={emptyAction.onClick}>
            {emptyAction.label}
          </Button>
        )}
      </Empty>
    );
  }

  return <>{children}</>;
}
