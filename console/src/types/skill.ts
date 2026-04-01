export interface Skill {
  id: string;
  name: string;
  description: string;
  skill_type: 'knowledge_qa' | 'workflow' | 'tool_call' | 'delegate' | 'chitchat';
  trigger_config: Record<string, unknown> | null;
  execution_config: Record<string, unknown> | null;
  priority: number;
  managed_by: string | null;
  tenant_id: string;
  enabled: boolean;
  created_at: string;
  updated_at: string;
}

export interface SkillCreate {
  name: string;
  description?: string;
  skill_type: string;
  trigger_config?: Record<string, unknown>;
  execution_config?: Record<string, unknown>;
  priority?: number;
  tenant_id?: string;
}

export type SkillUpdate = Partial<SkillCreate>;
