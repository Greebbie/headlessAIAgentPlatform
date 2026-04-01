export interface InvokeRequest {
  agent_id: string;
  message: string;
  session_id?: string;
  tenant_id?: string;
  form_data?: Record<string, unknown>;
}

export interface InvokeResponse {
  session_id: string;
  trace_id: string;
  short_answer: string;
  expanded_answer?: string;
  citations: Citation[];
  suggested_followups: string[];
  workflow_card?: WorkflowCard;
  workflow_status?: string;
  escalated: boolean;
  escalation_reason?: string;
  skill_info?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

export interface Citation {
  source_id: string;
  source_name: string;
  content_snippet: string;
  page?: number | null;
  paragraph?: number | null;
  line_start?: number | null;
  line_end?: number | null;
  score?: number | null;
}

export interface WorkflowCardField {
  name: string;
  label: string;
  field_type?: string;
  required?: boolean;
  placeholder?: string;
  options?: Array<{ label: string; value: string }>;
  file_config?: { max_size_mb?: number; allowed_extensions?: string[] };
}

export interface WorkflowCard {
  step_name: string;
  step_type: string;
  prompt: string;
  fields?: WorkflowCardField[];
  current_step: number;
  total_steps: number;
  collected_data?: Record<string, unknown>;
  webhook_result?: Record<string, unknown>;
}

export interface SSEEvent {
  event: 'status' | 'answer' | 'done' | 'error';
  data: Record<string, unknown>;
}
