import axios from 'axios';

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
});

// ── Agents ──────────────────────────────────────────
export const agentApi = {
  list: (tenantId = 'default') => api.get('/agents/', { params: { tenant_id: tenantId } }),
  create: (data: any) => api.post('/agents/', data),
  update: (id: string, data: any) => api.put(`/agents/${id}`, data),
  delete: (id: string) => api.delete(`/agents/${id}`),
};

// ── Workflows ───────────────────────────────────────
export const workflowApi = {
  list: (tenantId = 'default') => api.get('/workflows/', { params: { tenant_id: tenantId } }),
  create: (data: any) => api.post('/workflows/', data),
  update: (id: string, data: any) => api.put(`/workflows/${id}`, data),
  delete: (id: string) => api.delete(`/workflows/${id}`),
  addStep: (workflowId: string, data: any) => api.post(`/workflows/${workflowId}/steps`, data),
  updateStep: (workflowId: string, stepId: string, data: any) => api.put(`/workflows/${workflowId}/steps/${stepId}`, data),
  deleteStep: (workflowId: string, stepId: string) => api.delete(`/workflows/${workflowId}/steps/${stepId}`),
};

// ── Knowledge ───────────────────────────────────────
export const knowledgeApi = {
  listSources: (tenantId = 'default') => api.get('/knowledge/sources', { params: { tenant_id: tenantId } }),
  createSource: (data: any) => api.post('/knowledge/sources', data),
  deleteSource: (id: string) => api.delete(`/knowledge/sources/${id}`),
  addKV: (data: any) => api.post('/knowledge/kv', data),
  addFAQ: (data: any) => api.post('/knowledge/faq', data),
  search: (data: any) => api.post('/knowledge/search', data),
  upload: (formData: FormData) => api.post('/knowledge/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
  }),
  listChunks: (sourceId: string) => api.get(`/knowledge/sources/${sourceId}/chunks`),
};

// ── Tools ───────────────────────────────────────────
export const toolApi = {
  list: (tenantId = 'default') => api.get('/tools/', { params: { tenant_id: tenantId } }),
  create: (data: any) => api.post('/tools/', data),
  update: (id: string, data: any) => api.put(`/tools/${id}`, data),
  delete: (id: string) => api.delete(`/tools/${id}`),
  test: (data: any) => api.post('/tools/test', data),
};

// ── Audit ───────────────────────────────────────────
export const auditApi = {
  getTrace: (traceId: string) => api.get(`/audit/traces/${traceId}`),
  getSessionTraces: (sessionId: string) => api.get(`/audit/sessions/${sessionId}/traces`),
  getMetrics: (tenantId = 'default', hours = 24) => api.get('/audit/metrics', { params: { tenant_id: tenantId, hours } }),
  listTraces: (params?: { tenant_id?: string; limit?: number; offset?: number; event_type?: string }) =>
    api.get('/audit/traces', { params }),
};

// ── Invoke (for testing) ────────────────────────────
export const invokeApi = {
  send: (data: any) => api.post('/invoke', data),
};

// ── LLM Configs ─────────────────────────────────────
export const llmConfigApi = {
  list: (tenantId = 'default') => api.get('/llm-configs/', { params: { tenant_id: tenantId } }),
  create: (data: any) => api.post('/llm-configs/', data),
  update: (id: string, data: any) => api.put(`/llm-configs/${id}`, data),
  delete: (id: string) => api.delete(`/llm-configs/${id}`),
  setDefault: (id: string) => api.post(`/llm-configs/set-default/${id}`),
  getTemplates: () => api.get('/llm-configs/templates'),
  test: (data: any) => api.post('/llm-configs/test', data),
};

// ── Performance ─────────────────────────────────────
export const performanceApi = {
  getPresets: () => api.get('/performance/presets'),
  applyPreset: (preset: string) => api.post('/performance/presets/apply', { preset }),
  getCurrentConfig: () => api.get('/performance/current-config'),
  updateConfig: (data: any) => api.post('/performance/update-config', data),
};

// ── Agent Capabilities ────────────────────────────
export const agentCapabilitiesApi = {
  get: (agentId: string) => api.get(`/agents/${agentId}/capabilities`),
  update: (agentId: string, data: any) => api.put(`/agents/${agentId}/capabilities`, data),
};

// ── Skills ─────────────────────────────────────────
export const skillApi = {
  list: (tenantId = 'default', managedBy?: string) => api.get('/skills/', { params: { tenant_id: tenantId, managed_by: managedBy } }),
  create: (data: any) => api.post('/skills/', data),
  update: (id: string, data: any) => api.put(`/skills/${id}`, data),
  delete: (id: string) => api.delete(`/skills/${id}`),
};

// ── Agent Connections ──────────────────────────────
export const agentConnectionApi = {
  list: (tenantId = 'default', agentId?: string) => api.get('/agent-connections/', { params: { tenant_id: tenantId, agent_id: agentId } }),
  create: (data: any) => api.post('/agent-connections/', data),
  delete: (id: string) => api.delete(`/agent-connections/${id}`),
};

export default api;
