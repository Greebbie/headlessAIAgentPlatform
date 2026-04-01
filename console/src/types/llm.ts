export interface LLMConfig {
  id: string;
  name: string;
  provider: string;
  base_url: string;
  api_key: string;
  model: string;
  temperature: number;
  max_tokens: number;
  timeout_ms: number;
  is_default: boolean;
  tenant_id: string;
  created_at: string;
  updated_at: string;
}

export interface LLMConfigCreate {
  name: string;
  provider: string;
  base_url: string;
  api_key?: string;
  model: string;
  temperature?: number;
  max_tokens?: number;
  timeout_ms?: number;
  is_default?: boolean;
  tenant_id?: string;
}

export type LLMConfigUpdate = Partial<LLMConfigCreate>;

export interface LLMTemplate {
  provider: string;
  name: string;
  base_url: string;
  model: string;
  description: string;
}
