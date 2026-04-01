export interface KnowledgeSource {
  id: string;
  name: string;
  source_type: 'document' | 'faq' | 'structured_table' | 'kv_entity';
  source_uri: string;
  domain: string;
  tenant_id: string;
  status: 'pending' | 'processing' | 'ready' | 'error';
  metadata: Record<string, unknown> | null;
  chunk_count: number;
  created_at: string;
  updated_at: string;
}

export interface KnowledgeChunk {
  id: string;
  source_id: string;
  content: string;
  entity_key: string | null;
  domain: string;
  chunk_index: number;
  created_at: string;
}

export interface KnowledgeSourceCreate {
  name: string;
  source_type: string;
  domain?: string;
  tenant_id?: string;
}

export interface RetrievalRequest {
  query: string;
  domain?: string;
  top_k?: number;
  use_fast_channel?: boolean;
  use_rag_channel?: boolean;
}

export interface RetrievalHit {
  chunk_id: string;
  content: string;
  score: number;
  source: string;
  channel: string;
}

export interface RetrievalResponse {
  hits: RetrievalHit[];
  fast_answer: string | null;
  query: string;
}
