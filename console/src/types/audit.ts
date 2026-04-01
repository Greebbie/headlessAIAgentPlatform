export interface AuditTrace {
  id: string;
  trace_id: string;
  session_id: string;
  agent_id: string;
  tenant_id: string;
  event_type: string;
  event_data: Record<string, unknown> | null;
  retrieval_hits: Record<string, unknown> | null;
  llm_meta: Record<string, unknown> | null;
  tool_meta: Record<string, unknown> | null;
  workflow_meta: Record<string, unknown> | null;
  escalation_reason?: string;
  latency_ms?: number;
  timestamp?: string;
}

export interface AuditTraceSummary {
  id: string;
  trace_id: string;
  session_id: string;
  agent_id: string;
  event_type: string;
  latency_ms: number | null;
  timestamp: string | null;
}

export interface AuditTraceListResponse {
  total: number;
  offset: number;
  limit: number;
  items: AuditTraceSummary[];
}

export interface AuditMetrics {
  total_requests: number;
  avg_latency_ms: number;
  error_rate: number;
  escalation_rate: number;
  top_agents: Array<{ agent_id: string; count: number }>;
}
