import React from 'react';
import { Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import { Layout, Menu } from 'antd';
import {
  RobotOutlined,
  ApartmentOutlined,
  BookOutlined,
  ApiOutlined,
  AuditOutlined,
  DashboardOutlined,
  ExperimentOutlined,
  SettingOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons';

import AgentsPage from './pages/AgentsPage';
import WorkflowsPage from './pages/WorkflowsPage';
import KnowledgePage from './pages/KnowledgePage';
import ToolsPage from './pages/ToolsPage';
import AuditPage from './pages/AuditPage';
import DashboardPage from './pages/DashboardPage';
import PlaygroundPage from './pages/PlaygroundPage';
import LLMConfigsPage from './pages/LLMConfigsPage';
import SettingsPage from './pages/SettingsPage';

const { Header, Sider, Content } = Layout;

const menuItems = [
  { key: '/', icon: <DashboardOutlined />, label: '仪表盘' },
  { key: '/playground', icon: <ExperimentOutlined />, label: '测试台' },
  { key: '/agents', icon: <RobotOutlined />, label: 'Agent 管理' },
  { key: '/workflows', icon: <ApartmentOutlined />, label: '流程编排' },
  { key: '/knowledge', icon: <BookOutlined />, label: '知识管理' },
  { key: '/tools', icon: <ApiOutlined />, label: '工具管理' },
  { key: '/llm-configs', icon: <ThunderboltOutlined />, label: 'LLM 配置' },
  { key: '/audit', icon: <AuditOutlined />, label: '审计回放' },
  { key: '/settings', icon: <SettingOutlined />, label: '系统设置' },
];

export default function App() {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider theme="dark" width={200}>
        <div style={{ height: 48, margin: 16, color: '#fff', fontSize: 18, fontWeight: 'bold', textAlign: 'center', lineHeight: '48px' }}>
          HlAB Console
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
        />
      </Sider>
      <Layout>
        <Header style={{ background: '#fff', padding: '0 24px', fontSize: 16, fontWeight: 500 }}>
          Headless AI Agent Builder - 可视化控制台
        </Header>
        <Content style={{ margin: 24, padding: 24, background: '#fff', borderRadius: 8 }}>
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/playground" element={<PlaygroundPage />} />
            <Route path="/agents" element={<AgentsPage />} />
            <Route path="/workflows" element={<WorkflowsPage />} />
            <Route path="/knowledge" element={<KnowledgePage />} />
            <Route path="/tools" element={<ToolsPage />} />
            <Route path="/llm-configs" element={<LLMConfigsPage />} />
            <Route path="/audit" element={<AuditPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </Content>
      </Layout>
    </Layout>
  );
}
