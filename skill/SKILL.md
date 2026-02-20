---
name: rag-pipeline
description: Build production RAG pipelines - Voyage 4 embeddings, hybrid search, CRAG self-correction, agentic retrieval, RAGAS evaluation
allowed-tools: [Bash, Read, Write, Edit, Task, Glob, Grep, WebSearch]
keywords: [rag, retrieval, augmented, generation, embedding, vector, chunking, contextual, hybrid-search, reranking, llm, ai, pipeline, voyage, pgvector, agentic-rag, self-rag, crag, colpali, graph-rag]
---

# RAG Pipeline Builder (2026 Production Edition)

Build production-grade RAG systems based on comprehensive research across 30+ techniques, benchmarks, and papers.

## Reference Stack

| Layer | Component | Spec |
|-------|-----------|------|
| Embedding | **Voyage `voyage-4-large`** | 1024dim, MoE, Matryoshka, 32K context, multilingual |
| Reranking | **Voyage `rerank-2`** | Cross-encoder reranking |
| Vector DB | **pgvector + pgvectorscale** | 471 QPS at 50M vectors (10x faster than Qdrant) |
| Keyword Search | **PostgreSQL tsvector** | Same DB, no extra infra |
| LLM | **Claude (API or CLI)** | Generation + context enrichment |
| Evaluation | **RAGAS** | Faithfulness, precision, recall |
| Monitoring | **Langfuse** (open-source) | Production observability |

## Invocation

```
/rag-pipeline [strategy]
```

## Prerequisites Check (Auto-run on skill invocation)

Before starting the question flow, automatically check for required API keys:

```python
# Run this check via Bash at skill start
import os, subprocess

checks = {
    "VOYAGE_API_KEY": os.environ.get("VOYAGE_API_KEY"),
    "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
}

# Also check .env files in project root
for env_file in [".env", ".env.local"]:
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, val = line.strip().split("=", 1)
                    if key.strip() in checks and not checks[key.strip()]:
                        checks[key.strip()] = val.strip()
```

**Check flow:**

1. Search for `VOYAGE_API_KEY` in: `$VOYAGE_API_KEY` env → `.env` → `.env.local` → `~/.bashrc` → `~/.zshrc`
2. Search for `ANTHROPIC_API_KEY` in same locations
3. If **VOYAGE_API_KEY missing**:
   - Display: "Voyage AI API 키가 필요합니다."
   - Guide: "1. https://dash.voyageai.com/ 접속 → 2. 가입 (무료 50M 토큰/월) → 3. API Keys에서 키 생성"
   - Ask: "키를 .env 파일에 추가할까요?" → `echo 'VOYAGE_API_KEY=your-key-here' >> .env`
   - **Alternative**: "Voyage 없이 진행하려면 로컬 모델(bge-m3)로 대체 가능합니다."
4. If **ANTHROPIC_API_KEY missing**:
   - Display: "Claude API 키가 필요합니다 (contextual enrichment + generation용)."
   - Guide: "https://console.anthropic.com/ → API Keys"
   - **Alternative**: "Claude CLI가 설치되어 있으면 API 키 없이 CLI 모드로 사용 가능합니다."
   - CLI 모드 설정: `export RAG_LLM_MODE=cli`
5. If both present → proceed to Question Flow

**Fallback stack (API 키 없을 때):**

| Layer | With Voyage API | Without (Local fallback) |
|-------|----------------|--------------------------|
| Embedding | voyage-4-large | bge-m3 (open-source, CPU OK) |
| Reranking | rerank-2 | Cross-encoder: ms-marco-MiniLM-L-6-v2 |
| LLM | Claude API | Ollama (llama3, local) |

## Question Flow

### Phase 1: Scale

```yaml
question: "RAG 시스템이 처리할 문서 규모는?"
header: "Scale"
options:
  - label: "Small (< 50 pages)"
    description: "Full Context 전략 — 통째로 넣기 + 프롬프트 캐싱"
  - label: "Medium (50-500 pages)"
    description: "Advanced RAG — Contextual + Hybrid + Rerank (추천)"
  - label: "Large (500+ pages)"
    description: "Full Pipeline — + Graph RAG + Agentic + Self-correction"
  - label: "모르겠음"
    description: "문서 분석 후 자동 판단"
```

### Phase 2: Use Case

```yaml
question: "어떤 종류의 문서?"
header: "Use case"
options:
  - label: "내부 문서 / 지식베이스"
    description: "사내 위키, 매뉴얼, SOP"
  - label: "법률 / 계약서"
    description: "정확한 조항 검색. ColBERT 리랭킹 추천"
  - label: "PDF / 표 / 이미지 포함"
    description: "멀티모달 RAG (ColPali) 적용"
  - label: "코드 문서 / API"
    description: "코드 특화 청킹 + 구조 보존"
```

### Phase 3: Self-correction 수준

```yaml
question: "답변 신뢰도 검증 수준은?"
header: "Validation"
options:
  - label: "Basic — Reranking만"
    description: "빠르고 저렴. 대부분 충분"
  - label: "CRAG — 검색 품질 자동 평가"
    description: "검색 결과가 나쁘면 자동 폐기 + 웹 검색 폴백"
  - label: "Self-RAG — 검색 필요성까지 판단"
    description: "검색 없이 답할 수 있으면 직접 답변. 효율 최적화"
  - label: "Agentic — 에이전트가 전체 제어"
    description: "가장 정확. 멀티스텝 검색, 자가 평가, 재검색"
```

---

## Architecture Overview (10 Types)

| Type | Core | When |
|------|------|------|
| **Naive RAG** | chunk → embed → search → generate | Prototype only |
| **Advanced RAG** | + hybrid search + reranking | General production |
| **Modular RAG** | Swappable components | Custom requirements |
| **Graph RAG** | Knowledge graph + community summaries | Multi-hop reasoning |
| **Agentic RAG** | Agent decides search strategy | Complex queries |
| **Self-RAG** | Model judges if retrieval needed | Efficiency optimization |
| **Corrective RAG** | Validates retrieval quality, web fallback | High reliability |
| **Adaptive RAG** | Auto-switches strategy per query type | Mixed query types |
| **Multimodal RAG** | Images/tables/charts via ColPali | PDFs, infographics |
| **Long Context + RAG** | Hybrid of full context and retrieval | Large docs + accuracy |

**2026 production baseline**: Advanced RAG + Agentic patterns + Self-correction

---

## INGESTION PIPELINE

### Step 1: Chunking Strategy

**Benchmark results:**

| Strategy | Recall | Notes |
|----------|--------|-------|
| Fixed-size (400-512 tokens) | 85-90% | Simple, predictable |
| Semantic chunking | +9% vs fixed | Splits at meaning boundaries |
| Page-level | 0.648 accuracy | NVIDIA benchmark winner, lowest variance |
| Late Chunking | ≈ Contextual | No LLM needed, cost-efficient |
| Contextual Retrieval | -67% failure | Best accuracy, highest cost |

**Recommended: Late Chunking + Contextual Enrichment hybrid**

```python
# chunking.py
from dataclasses import dataclass, field

@dataclass
class Chunk:
    content: str
    metadata: dict = field(default_factory=dict)
    context: str = ""      # Contextual description
    summary: str = ""      # Search-optimized summary
    embedding: list = field(default_factory=list)

def semantic_chunk(text: str, source: str, max_tokens: int = 512) -> list[Chunk]:
    """Split at paragraph boundaries, respecting semantic coherence."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_parts = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para.split())
        if current_len + para_len > max_tokens and current_parts:
            chunks.append(Chunk(
                content="\n\n".join(current_parts),
                metadata={"source": source, "chunk_index": len(chunks)}
            ))
            # Keep last paragraph for overlap
            current_parts = [current_parts[-1], para]
            current_len = len(current_parts[0].split()) + para_len
        else:
            current_parts.append(para)
            current_len += para_len

    if current_parts:
        chunks.append(Chunk(
            content="\n\n".join(current_parts),
            metadata={"source": source, "chunk_index": len(chunks)}
        ))
    return chunks
```

### Step 2: LLM Wrapper (API or CLI)

Voyage handles embedding/reranking only. LLM (enrichment, CRAG, generation) is separate — use **Claude API** or **Claude CLI**, your choice.

```python
# llm.py — LLM abstraction layer
import os
import subprocess

MODE = os.environ.get("RAG_LLM_MODE", "api")  # "api" or "cli"

if MODE == "api":
    import anthropic
    _client = anthropic.Anthropic()

def llm(prompt: str, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 300) -> str:
    """Call Claude via API or CLI."""
    if MODE == "api":
        resp = _client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text
    else:
        # Claude CLI — no API key needed if already authenticated
        result = subprocess.run(
            ["claude", "-p", prompt, "--model", model],
            capture_output=True, text=True, timeout=60
        )
        return result.stdout.strip()
```

**선택 기준:**

| | Claude API | Claude CLI |
|--|-----------|-----------|
| 속도 | 빠름 (직접 호출) | 약간 느림 (프로세스 스폰) |
| 인증 | ANTHROPIC_API_KEY 필요 | `claude` 로그인만 되어있으면 됨 |
| 비용 | API 사용량 과금 | CLI 플랜에 포함 |
| 배치 처리 | 병렬 가능 | 순차 권장 |
| 프로덕션 | 권장 | 개발/프로토타입에 적합 |

### Step 3: Contextual Enrichment

Anthropic's method: prepend context description to each chunk before embedding.
**Reduces retrieval failure by 67%.**

```python
# enrichment.py
from llm import llm

CONTEXT_PROMPT = """<document>
{document}
</document>

<chunk>
{chunk}
</chunk>

Write 2-3 sentences explaining:
1. What document/section this chunk belongs to
2. Key entities, dates, or terms for retrieval
Respond with ONLY the context."""

SUMMARY_PROMPT = """Summarize in 1-2 sentences focused on searchable facts and entities:

{chunk}"""

def enrich_chunk(full_doc: str, chunk: Chunk) -> Chunk:
    """Add context + search summary to chunk."""
    chunk.context = llm(
        CONTEXT_PROMPT.format(document=full_doc, chunk=chunk.content),
        max_tokens=200
    )
    chunk.summary = llm(
        SUMMARY_PROMPT.format(chunk=chunk.content),
        max_tokens=100
    )
    return chunk
```

### Step 4: Embedding with Voyage 4

```python
# embedding.py
import voyageai

vo = voyageai.Client()  # VOYAGE_API_KEY env var

def embed_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Embed chunks using Voyage 4 large. Embed the SUMMARY, not raw content."""
    # Batch embed summaries (search index)
    texts = [f"{c.context}\n{c.summary}" for c in chunks]

    # Voyage supports batching up to 128 texts
    for i in range(0, len(texts), 128):
        batch = texts[i:i+128]
        result = vo.embed(
            batch,
            model="voyage-4-large",
            input_type="document",
            output_dimension=1024,     # Matryoshka: 256/512/1024/2048
            output_dtype="float"       # or "int8", "binary" for compression
        )
        for j, emb in enumerate(result.embeddings):
            chunks[i+j].embedding = emb

    return chunks

def embed_query(query: str) -> list[float]:
    """Embed a search query."""
    result = vo.embed(
        [query],
        model="voyage-4-large",
        input_type="query",
        output_dimension=1024
    )
    return result.embeddings[0]
```

**Cost optimization options:**
- `output_dimension=512` → 절반 크기, ~97% 성능 유지 (Matryoshka)
- `output_dtype="int8"` → 4x 메모리 절감, 96% 성능 유지
- `output_dtype="binary"` → 32x 압축, 92-96% 성능 유지

### Step 5: Storage (pgvector + Hybrid Index)

```sql
-- schema.sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    context TEXT NOT NULL,
    summary TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(1024),
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('simple', coalesce(summary, '') || ' ' || coalesce(context, ''))
    ) STORED,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for semantic search (pgvectorscale: 471 QPS at 50M vectors)
CREATE INDEX idx_embedding ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- GIN index for keyword search
CREATE INDEX idx_search ON chunks USING gin (search_vector);

-- Metadata filtering
CREATE INDEX idx_metadata ON chunks USING gin (metadata);
```

```python
# storage.py
import psycopg2
from pgvector.psycopg2 import register_vector

class ChunkStore:
    def __init__(self, db_url: str):
        self.conn = psycopg2.connect(db_url)
        register_vector(self.conn)

    def store(self, chunk: Chunk):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chunks (content, context, summary, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s)
            """, (chunk.content, chunk.context, chunk.summary,
                  json.dumps(chunk.metadata), chunk.embedding))
        self.conn.commit()

    def hybrid_search(self, query_embedding: list, query_text: str, top_k: int = 20) -> list[dict]:
        """Reciprocal Rank Fusion (RRF) of semantic + keyword search."""
        with self.conn.cursor() as cur:
            cur.execute("""
                WITH semantic AS (
                    SELECT id, content, context, summary, metadata,
                           ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS rank_s
                    FROM chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                ),
                keyword AS (
                    SELECT id,
                           ROW_NUMBER() OVER (ORDER BY ts_rank(search_vector, plainto_tsquery('simple', %s)) DESC) AS rank_k
                    FROM chunks
                    WHERE search_vector @@ plainto_tsquery('simple', %s)
                    LIMIT %s
                )
                SELECT s.id, s.content, s.context, s.summary, s.metadata,
                       (1.0 / (60 + s.rank_s)) + COALESCE(1.0 / (60 + k.rank_k), 0) AS rrf_score
                FROM semantic s
                LEFT JOIN keyword k ON s.id = k.id
                ORDER BY rrf_score DESC
                LIMIT %s
            """, (query_embedding, query_embedding, top_k * 2,
                  query_text, query_text, top_k * 2, top_k))

            columns = ['id', 'content', 'context', 'summary', 'metadata', 'rrf_score']
            return [dict(zip(columns, row)) for row in cur.fetchall()]
```

---

## QUERY PIPELINE

### Step 6: Reranking with Voyage rerank-2

```python
# reranker.py
import voyageai

vo = voyageai.Client()

def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank using Voyage rerank-2 cross-encoder."""
    documents = [f"{c['context']}\n\n{c['content']}" for c in chunks]

    result = vo.rerank(
        query=query,
        documents=documents,
        model="rerank-2",
        top_k=top_k
    )

    return [chunks[r.index] for r in result.results]
```

### Step 7: Self-Correction (CRAG Pattern)

```python
# crag.py
"""
Corrective RAG: evaluate retrieval quality before generation.
5 poisoned documents can manipulate 90% of responses — validation is essential.
"""
from llm import llm

EVAL_PROMPT = """Query: {query}

Retrieved document:
{document}

Is this document relevant to answering the query?
Respond with exactly one word: CORRECT, AMBIGUOUS, or INCORRECT."""

def evaluate_retrieval(query: str, chunks: list[dict]) -> tuple[list[dict], str]:
    """Evaluate and correct retrieval results."""
    correct, ambiguous = [], []

    for chunk in chunks:
        verdict = llm(
            EVAL_PROMPT.format(query=query, document=chunk['content'][:1000]),
            max_tokens=10
        ).strip().upper()

        if verdict == "CORRECT":
            correct.append(chunk)
        elif verdict == "AMBIGUOUS":
            ambiguous.append(chunk)

    if correct:
        return correct, "correct"
    elif ambiguous:
        # Refine: extract key info only
        return ambiguous, "ambiguous"
    else:
        # All incorrect: fallback to web search or broader retrieval
        return [], "incorrect"
```

### Step 8: Full Query Pipeline

```python
# pipeline.py
from llm import llm

SYSTEM_PROMPT = (
    "Answer based on the provided context. "
    "Cite sources using [Source: filename] format. "
    "If the context doesn't contain enough information, say so clearly."
)

def query(user_query: str, store: ChunkStore, top_k: int = 5, use_crag: bool = True) -> dict:
    """Complete RAG query pipeline."""

    # 1. Embed query
    query_emb = embed_query(user_query)

    # 2. Hybrid search (RRF: semantic + keyword)
    candidates = store.hybrid_search(query_emb, user_query, top_k=top_k * 3)

    # 3. Rerank with Voyage rerank-2
    reranked = rerank(user_query, candidates, top_k=top_k * 2)

    # 4. CRAG validation (optional but recommended)
    if use_crag:
        validated, status = evaluate_retrieval(user_query, reranked)
        if status == "incorrect":
            return {"answer": "관련 문서를 찾지 못했습니다.", "sources": [], "status": "no_results"}
        final_chunks = validated[:top_k]
    else:
        final_chunks = reranked[:top_k]

    # 5. Build context from ORIGINAL content (not summaries)
    context = "\n\n---\n\n".join([
        f"[Source: {c['metadata'].get('source', '?')}]\n{c['context']}\n\n{c['content']}"
        for c in final_chunks
    ])

    # 6. Generate with Claude (API or CLI)
    answer = llm(
        f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {user_query}",
        model="claude-sonnet-4-20250514",
        max_tokens=4096
    )

    return {
        "answer": answer,
        "sources": [c['metadata'] for c in final_chunks],
        "status": "success",
        "chunks_used": len(final_chunks)
    }
```

---

## ADVANCED PATTERNS

### Agentic RAG (for complex queries)

> **Note:** Agentic RAG uses Claude tool_use, so requires **API mode** (`RAG_LLM_MODE=api`).
> CLI mode doesn't support tool_use natively.

```python
# agentic_rag.py
"""Agent-driven retrieval: plan → search → validate → re-search if needed. (API mode only)"""
import anthropic
anthropic_client = anthropic.Anthropic()

TOOLS = [
    {
        "name": "search_documents",
        "description": "Search the knowledge base. Returns top relevant chunks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "default": 5}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_document_list",
        "description": "List all available document sources.",
        "input_schema": {"type": "object", "properties": {}}
    }
]

AGENT_SYSTEM = """You are a research agent with access to a document database.

Process:
1. Analyze the question — identify what information is needed
2. Search with multiple query angles (synonyms, related terms)
3. Evaluate if results answer the question
4. If insufficient, refine and search again (max 3 iterations)
5. Synthesize answer with citations

Never guess. If info isn't in the documents, say so."""

def agentic_query(user_query: str, store: ChunkStore) -> str:
    """Let Claude drive the retrieval process."""
    messages = [{"role": "user", "content": user_query}]

    for _ in range(5):  # Max turns
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=AGENT_SYSTEM,
            tools=TOOLS,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            return response.content[0].text

        # Handle tool calls
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "search_documents":
                    q_emb = embed_query(block.input["query"])
                    results = store.hybrid_search(q_emb, block.input["query"], block.input.get("top_k", 5))
                    reranked = rerank(block.input["query"], results, top_k=5)
                    tool_result = "\n\n".join([
                        f"[{r['metadata'].get('source', '?')}] {r['content'][:500]}"
                        for r in reranked
                    ])
                elif block.name == "get_document_list":
                    # Return unique sources
                    tool_result = "Available sources: ..."

                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": block.id, "content": tool_result}
                ]})

    return "Max iterations reached."
```

### Graph RAG (for multi-hop reasoning)

Use Microsoft's GraphRAG or build lightweight version:
```
Documents → Entity extraction → Knowledge graph (PostgreSQL recursive CTE or Neo4j)
Query → Entity recognition → Graph traversal → Related chunks → LLM
```
- Comprehensiveness: 72-83% improvement over standard RAG
- Diversity: 62-82% improvement
- **Trade-off**: High build cost, complex maintenance. Use only when multi-hop reasoning is critical.

### Multimodal RAG (ColPali for PDFs with images/tables)

```
Document pages → (treated as images) → ColPali VLM embedding → Vector store
Query → ColPali search → Vision LLM generates answer
```
- No OCR or layout detection needed
- Outperforms all text-based methods on ViDoRe benchmark
- Use for: financial reports, scientific papers, infographics

---

## EVALUATION (RAGAS)

```python
# evaluation.py
"""
RAG evaluation with RAGAS framework.
Install: pip install ragas
"""
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall, answer_relevancy
from datasets import Dataset

def evaluate_rag(questions: list, answers: list, contexts: list, ground_truths: list) -> dict:
    """Evaluate RAG pipeline with RAGAS metrics."""
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,          # list of list of strings
        "ground_truth": ground_truths
    })

    result = evaluate(
        dataset,
        metrics=[faithfulness, context_precision, context_recall, answer_relevancy]
    )
    return result

# Target scores:
# Faithfulness > 0.9 (answers grounded in context)
# Context Precision > 0.8 (retrieved docs are relevant)
# Context Recall > 0.8 (found all needed info)
# Answer Relevancy > 0.85 (answers match questions)
```

### Golden Dataset

Build 50-100 question-answer-source triplets for your domain. This is the single most important investment for RAG quality.

---

## PRODUCTION OPERATIONS

### Monitoring (Langfuse)

```python
# monitoring.py
from langfuse import Langfuse

langfuse = Langfuse()

def traced_query(user_query: str, store: ChunkStore) -> dict:
    trace = langfuse.trace(name="rag-query", input=user_query)

    # Track retrieval
    retrieval_span = trace.span(name="hybrid-search")
    candidates = store.hybrid_search(...)
    retrieval_span.end(output={"count": len(candidates)})

    # Track reranking
    rerank_span = trace.span(name="rerank")
    reranked = rerank(...)
    rerank_span.end(output={"count": len(reranked)})

    # Track generation
    gen_span = trace.span(name="generation")
    answer = ...
    gen_span.end(output={"tokens": ...})

    trace.update(output=answer)
    return answer
```

### Security

| Threat | Risk | Defense |
|--------|------|---------|
| Prompt Injection via RAG | Very high | Input/output filtering, RAG Triad validation |
| Data Poisoning | Very high (5 docs = 90% manipulation) | Source verification, anomaly detection |
| Adversarial Embeddings | High | Embedding integrity checks |
| Indirect Prompt Injection | High | Multi-layer defense (73% → 8.7% success rate) |

### Cost Optimization

| Strategy | Savings |
|----------|---------|
| Semantic caching (Redis) | 15-30% on repeat queries |
| Voyage int8 quantization | 4x memory, 96% quality |
| Binary quantization | 32x compression, 92% quality |
| Batch embedding | 10x cheaper vs individual |
| 2-stage search (binary → rerank) | 10x faster, 96.45% precision |
| Prompt caching (Anthropic) | 90% cheaper on cached prefix |

### Incremental Updates

```python
def update_document(doc_path: str, store: ChunkStore):
    """Update embeddings when a document changes."""
    # 1. Delete old chunks for this source
    store.delete_by_source(doc_path)
    # 2. Re-chunk, re-enrich, re-embed
    text = load_document(doc_path)
    chunks = semantic_chunk(text, doc_path)
    for chunk in chunks:
        enrich_chunk(text, chunk)
    embed_chunks(chunks)
    for chunk in chunks:
        store.store(chunk)
```

---

## DEPENDENCIES

```
# requirements.txt
anthropic>=0.45.0
voyageai>=0.3.0
pgvector>=0.3.0
psycopg2-binary>=2.9
pydantic>=2.0
fastapi>=0.115.0
ragas>=0.2.0
langfuse>=2.0
datasets>=3.0
```

## RAG vs Long Context Decision

| Situation | Choice |
|-----------|--------|
| Docs < 50p, rarely change | Long Context (full insert) |
| Docs > 200p, frequent updates | RAG |
| Cost-sensitive | RAG (far cheaper) |
| Max accuracy, budget ok | Long Context |
| Multi-turn conversation | RAG (keeps prompts lean) |
| Real-time data | RAG |

## Key Principles

1. **Hybrid search is mandatory** — BM25 only: 0.72 recall → Hybrid: 0.91 recall
2. **Reranking is cheap insurance** — Voyage rerank-2 dramatically improves precision
3. **Search on summaries, deliver originals** — Library catalog pattern
4. **CRAG validation prevents poisoning** — 5 malicious docs can corrupt 90% of responses
5. **Evaluate or you're guessing** — RAGAS golden dataset is non-negotiable
6. **Contextual enrichment: -67% failure** — Anthropic's proven method
7. **Matryoshka dimensions save money** — 512dim ≈ 97% of 1024dim quality

## References

- [Voyage 4 Model Family (MoE Architecture)](https://blog.voyageai.com/2026/01/15/voyage-4/)
- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [CRAG Paper (arXiv)](https://arxiv.org/abs/2401.15884)
- [Agentic RAG Survey (arXiv)](https://arxiv.org/abs/2501.09136)
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
- [Late Chunking Paper](https://arxiv.org/pdf/2409.04701)
- [Stanford "Lost in the Middle"](https://arxiv.org/abs/2307.03172)
- [RAGAS Evaluation](https://docs.ragas.io/en/latest/concepts/metrics/)
- [NirDiamant RAG Techniques (30+)](https://github.com/NirDiamant/RAG_Techniques)
- [ColPali Multimodal RAG](https://huggingface.co/blog/manu/colpali)
- [Blended RAG (Dense+Sparse+ColBERT)](https://infiniflow.org/blog/best-hybrid-search-solution)
- [RAG Security Threat Model](https://arxiv.org/pdf/2509.20324)
- [RAG Cost Optimization](https://thedataguy.pro/blog/2025/07/the-economics-of-rag-cost-optimization-for-production-systems/)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [pgvectorscale Benchmarks](https://www.tigerdata.com/blog/pgvector-vs-qdrant)
