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
| Embedding | **Voyage `voyage-4-large`** (Small/Medium) | 1024dim, MoE, Matryoshka, 32K context, multilingual |
| Embedding (Large docs) | **Voyage `voyage-context-3`** | 긴 문서 청크에 문서 전체 맥락 자동 주입. `/v1/contextualizedembeddings` API |
| Reranking | **Voyage `rerank-2`** | Cross-encoder reranking |
| Vector DB | **pgvector + pgvectorscale** | 471 QPS at 50M vectors (10x faster than Qdrant) |
| Keyword Search | **PostgreSQL tsvector** | Same DB, no extra infra |
| LLM | **Any (Claude / Gemini / OpenAI)** | Generation + context enrichment |
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
   - Display: "Claude API 키가 없습니다. LLM 대안을 선택하세요:"
   - **Option A**: Claude CLI 모드 → `export RAG_LLM_MODE=claude-cli` (API 키 불필요)
   - **Option B**: Gemini Flash (가장 저렴) → `export RAG_LLM_MODE=gemini` + `GEMINI_API_KEY`
   - **Option C**: OpenAI → `export RAG_LLM_MODE=openai` + `OPENAI_API_KEY`
   - **Option D**: Claude API → "https://console.anthropic.com/ → API Keys"
5. Also check for `GEMINI_API_KEY`, `OPENAI_API_KEY` — auto-detect available provider
6. If at least one LLM provider available → proceed to Question Flow

**Fallback stack (API 키 없을 때):**

| Layer | With API Keys | Without (Fallback) |
|-------|--------------|---------------------|
| Embedding | voyage-4-large | bge-m3 (open-source, CPU OK) |
| Reranking | rerank-2 | Cross-encoder: ms-marco-MiniLM-L-6-v2 |
| LLM | Claude API / Gemini / OpenAI | Claude CLI (plan) or Ollama (local) |

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

### Phase 3: LLM 선택

```yaml
question: "답변 생성 / CRAG 검증 / 맥락 강화에 사용할 LLM은?"
header: "LLM"
options:
  - label: "Gemini Flash (Recommended)"
    description: "GEMINI_API_KEY 필요. 가장 저렴하고 빠름. RAG의 LLM 작업은 단순해서 충분"
  - label: "Claude CLI"
    description: "API 키 불필요. Claude Code 플랜만 있으면 됨"
  - label: "OpenAI"
    description: "OPENAI_API_KEY 필요. GPT-4o-mini 등 범용"
  - label: "Claude API"
    description: "ANTHROPIC_API_KEY 필요. tool_use 지원. Agentic RAG 필요 시"
```

> **Note**: 임베딩/리랭킹은 항상 Voyage입니다. 여기서 선택하는 건 LLM(맥락 강화, CRAG 판단, 답변 생성)만 해당됩니다.
> Agentic RAG를 사용하려면 tool_use가 필요하므로 Claude API / Gemini / OpenAI 중 선택해야 합니다 (CLI 불가).

### Phase 4: Self-correction 수준

```yaml
question: "답변 신뢰도 검증 수준은?"
header: "Validation"
options:
  - label: "Basic — Reranking만"
    description: "빠르고 저렴. 대부분 충분"
  - label: "CRAG — 검색 품질 자동 평가"
    description: "검색 결과가 나쁘면 자동 폐기 + DuckDuckGo 웹 검색 폴백 (무료, 키 불필요)"
  - label: "Self-RAG — 검색 필요성까지 판단"
    description: "검색 없이 답할 수 있으면 직접 답변. 효율 최적화"
  - label: "Agentic — 에이전트가 전체 제어"
    description: "가장 정확. 멀티스텝 검색, 자가 평가, 재검색. tool_use 필요 (CLI 불가)"
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

### Step 0: Foundation Modules (항상 먼저 생성)

```python
# exceptions.py — 커스텀 예외 계층
class RagError(Exception):
    """Base exception for all RAG pipeline errors."""

class LLMError(RagError):
    """LLM API call failed."""

class RateLimitError(LLMError):
    """429 Rate Limit exceeded — retriable."""

class EmbeddingError(RagError):
    """Embedding API call failed."""

class StorageError(RagError):
    """Database connection or query failed."""

class ValidationError(RagError):
    """Input validation failed."""

class CRAGError(RagError):
    """CRAG validation logic failed."""
```

```python
# config.py — 환경변수 중앙 관리 (pydantic-settings)
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    voyage_api_key: str = ""          # "" = bge-m3 로컬 폴백 사용
    database_url: str = "postgresql://rag:ragpass@localhost:5432/ragdb"
    rag_llm_mode: str = "gemini"
    gemini_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    llm_max_retries: int = 3
    llm_retry_min_wait: float = 1.0
    llm_retry_max_wait: float = 60.0
    llm_timeout: int = 120
    db_pool_min: int = 2
    db_pool_max: int = 10
    max_query_length: int = 2000

    model_config = {"env_file": ".env", "extra": "ignore"}

settings = Settings()
```

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


def late_chunk(
    text: str,
    source: str,
    vo: "voyageai.Client",
    max_tokens: int = 512,
    use_late_chunking: bool = False,
) -> list[Chunk]:
    """
    Late Chunking via voyage-context-3 (500p+ 긴 문서 전용).

    voyage-context-3는 청크 리스트를 한 번에 처리해 각 청크에
    문서 전체 맥락을 자동 주입합니다. Late Chunking 논문 방식의
    Voyage 공식 구현체입니다. Jina-v3 late chunking 대비 +23.66%.

    use_late_chunking=False (기본값) → semantic_chunk()로 폴백
    use_late_chunking=True → voyage-context-3 사용 (Large 규모 권장)

    Reference:
      - https://arxiv.org/pdf/2409.04701 (Late Chunking paper)
      - https://docs.voyageai.com/docs/contextualized-chunk-embeddings
    """
    if not use_late_chunking:
        return semantic_chunk(text, source, max_tokens)

    # 1. 텍스트를 청크로 분리
    raw_chunks = semantic_chunk(text, source, max_tokens)
    if not raw_chunks:
        return []

    chunk_texts = [c.content for c in raw_chunks]

    # 2. voyage-context-3로 문서 전체 맥락을 보존한 임베딩
    #    inputs는 List[List[str]] — 같은 문서의 청크를 inner list로 묶음
    result = vo.embed_chunks(
        chunks=[chunk_texts],           # 이 문서의 모든 청크를 하나의 그룹으로
        model="voyage-context-3",
        input_type="document",
        output_dimension=1024,
    )

    # 3. 임베딩을 Chunk 객체에 주입
    for i, chunk in enumerate(raw_chunks):
        chunk.embedding = result.embeddings[0][i]   # [문서0][청크i]
        chunk.metadata["method"] = "voyage-context-3"

    return raw_chunks
```

### Step 2: LLM Wrapper (API or CLI)

Voyage handles embedding/reranking only. LLM (enrichment, CRAG, generation) is separate — choose any provider.

```python
# llm.py — Multi-LLM abstraction layer with retry
import os
import logging
import subprocess
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_sleep_log
from exceptions import LLMError, RateLimitError
from config import settings

logger = logging.getLogger(__name__)
MODE = settings.rag_llm_mode

def _retry_config(exc_types: tuple):
    return dict(
        wait=wait_exponential(
            multiplier=1,
            min=settings.llm_retry_min_wait,
            max=settings.llm_retry_max_wait,
        ),
        stop=stop_after_attempt(settings.llm_max_retries),
        retry=retry_if_exception_type(exc_types),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )

@retry(**_retry_config((LLMError, RateLimitError)))
def _call_claude_api(prompt: str, max_tokens: int, model: str) -> str:
    import anthropic
    try:
        resp = anthropic.Anthropic().messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text
    except anthropic.RateLimitError as e:
        raise RateLimitError(str(e)) from e
    except anthropic.APIError as e:
        raise LLMError(str(e)) from e

@retry(**_retry_config((LLMError,)))
def _call_claude_cli(prompt: str, max_tokens: int, model: str) -> str:
    result = subprocess.run(
        ["claude", "--print", prompt, "--model", model],
        capture_output=True, text=True,
        timeout=settings.llm_timeout,
    )
    if result.returncode != 0:
        raise LLMError(f"claude CLI failed (rc={result.returncode}): {result.stderr[:200]}")
    return result.stdout.strip()

@retry(**_retry_config((LLMError, RateLimitError)))
def _call_gemini(prompt: str, max_tokens: int, model: str) -> str:
    from google import genai
    from google.genai import types
    try:
        client = genai.Client(api_key=settings.gemini_api_key)
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=max_tokens),
        )
        return resp.text
    except Exception as e:
        if "429" in str(e) or "RATE_LIMIT" in str(e).upper():
            raise RateLimitError(str(e)) from e
        raise LLMError(str(e)) from e

@retry(**_retry_config((LLMError, RateLimitError)))
def _call_openai(prompt: str, max_tokens: int, model: str) -> str:
    from openai import OpenAI, RateLimitError as OAIRateLimit, APIError
    try:
        resp = OpenAI().chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content
    except OAIRateLimit as e:
        raise RateLimitError(str(e)) from e
    except APIError as e:
        raise LLMError(str(e)) from e

def llm(prompt: str, max_tokens: int = 300, model: str | None = None) -> str:
    """Call LLM via configured provider. model overrides RAG_LLM_MODEL env var."""
    _model = model or os.environ.get("RAG_LLM_MODEL", {
        "claude-api": "claude-haiku-4-5-20251001",
        "claude-cli": "claude-haiku-4-5-20251001",
        "gemini":     "gemini-2.0-flash",
        "openai":     "gpt-4o-mini",
    }.get(MODE, "gemini-2.0-flash"))

    if MODE == "claude-api":   return _call_claude_api(prompt, max_tokens, _model)
    if MODE == "claude-cli":   return _call_claude_cli(prompt, max_tokens, _model)
    if MODE == "gemini":       return _call_gemini(prompt, max_tokens, _model)
    if MODE == "openai":       return _call_openai(prompt, max_tokens, _model)
    raise ValueError(f"Unknown RAG_LLM_MODE: {MODE}")


def get_api_client():
    """Return (client, model_name, provider) for tool_use capable providers."""
    if MODE == "claude-api":
        import anthropic
        return (
            anthropic.Anthropic(),
            os.environ.get("RAG_LLM_MODEL", "claude-haiku-4-5-20251001"),
            "anthropic",
        )
    elif MODE == "gemini":
        model_name = os.environ.get("RAG_LLM_MODEL", "gemini-2.0-flash")
        return None, model_name, "gemini"
    elif MODE == "openai":
        from openai import OpenAI
        return None, os.environ.get("RAG_LLM_MODEL", "gpt-4o-mini"), "openai"
    else:
        raise ValueError(
            "Agentic RAG requires tool_use. "
            "Set RAG_LLM_MODE to claude-api."
        )
```

**설정 예시:**

```bash
# Gemini Flash (기본값, 가장 저렴하고 빠름 — RAG LLM 작업에 충분)
export RAG_LLM_MODE=gemini
export GEMINI_API_KEY=AI...

# Claude CLI (API 키 불필요, 플랜만 있으면 됨)
export RAG_LLM_MODE=claude-cli

# OpenAI
export RAG_LLM_MODE=openai
export OPENAI_API_KEY=sk-...

# Claude API (Agentic RAG 등 tool_use 필요 시)
export RAG_LLM_MODE=claude-api
export ANTHROPIC_API_KEY=sk-ant-...
```

**선택 기준:**

> **왜 Gemini Flash가 기본값인가?** RAG에서 LLM은 맥락 강화(한 줄 요약), CRAG 검증(분류), 답변 생성(읽고 요약) 같은 단순 작업만 합니다. 이런 작업에서 모델 간 품질 차이는 거의 없고, 비용 차이만 큽니다.

| | **Gemini Flash (기본)** | Claude CLI | OpenAI | Claude API |
|--|------------------------|-----------|--------|-----------|
| 비용 | **가장 저렴** | 플랜 포함 | 사용량 과금 | 사용량 과금 |
| 속도 | **가장 빠름** | 느림 | 빠름 | 빠름 |
| tool_use | 지원 | 미지원 | 지원 | 지원 |
| Agentic RAG | 가능 | 불가 | 가능 | 가능 |
| 추천 | **일반 용도** | API 키 없을 때 | 범용 | Agentic 필요 시 |

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
# embedding.py — with rate limit retry + bge-m3 fallback
import logging
import voyageai
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_sleep_log
from exceptions import EmbeddingError, RateLimitError
from config import settings
from chunking import Chunk       # Chunk 타입 참조

logger = logging.getLogger(__name__)

# Voyage 키 없으면 로컬 bge-m3 폴백
if settings.voyage_api_key:
    vo = voyageai.Client(api_key=settings.voyage_api_key)
    _USE_LOCAL = False
else:
    from sentence_transformers import SentenceTransformer as _ST
    _local_model = _ST("BAAI/bge-m3")
    _USE_LOCAL = True
    logger.warning("VOYAGE_API_KEY not set — using local bge-m3 (slower, no reranking)")

def _embed_batch_raw(texts: list[str], input_type: str, dimension: int) -> list[list[float]]:
    """Inner call with rate-limit retry (max 5 attempts, exponential backoff up to 2 min).
    Falls back to local bge-m3 if VOYAGE_API_KEY is not set."""
    if _USE_LOCAL:
        # bge-m3 폴백: dimension 무시 (1024 고정), input_type 미지원
        embeddings = _local_model.encode(texts, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]

    @retry(
        wait=wait_exponential(multiplier=2, min=2, max=120),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((RateLimitError, EmbeddingError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call():
        try:
            result = vo.embed(
                texts,
                model="voyage-4-large",
                input_type=input_type,       # "document" vs "query" — MUST match usage
                output_dimension=dimension,
                output_dtype="float",
            )
            return result.embeddings
        except voyageai.error.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except Exception as e:
            raise EmbeddingError(str(e)) from e
    return _call()

def embed_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Embed chunks using Voyage 4 large. Embeds SUMMARY+CONTEXT, not raw content."""
    texts = [f"{c.context}\n{c.summary}" for c in chunks]
    for i in range(0, len(texts), 128):          # Voyage batch limit: 128
        batch = texts[i:i+128]
        embeddings = _embed_batch_raw(batch, input_type="document", dimension=1024)
        for j, emb in enumerate(embeddings):
            chunks[i+j].embedding = emb
    return chunks

def embed_query(query: str) -> list[float]:
    """Embed a search query. Uses input_type='query' — different from documents."""
    return _embed_batch_raw([query], input_type="query", dimension=1024)[0]
```

**Cost optimization options:**
- `output_dimension=512` → 절반 크기, ~97% 성능 유지 (Matryoshka)
- `output_dtype="int8"` → 4x 메모리 절감, 96% 성능 유지
- `output_dtype="binary"` → 32x 압축, 92-96% 성능 유지

### Docker Setup (PostgreSQL + pgvector)

```yaml
# docker-compose.yml — Claude Code가 자동 생성하는 템플릿
services:
  postgres:
    image: pgvector/pgvector:pg17
    environment:
      POSTGRES_DB: ragdb
      POSTGRES_USER: rag
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-ragpass}
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./schema.sql:/docker-entrypoint-initdb.d/schema.sql

volumes:
  pgdata:
```

> `docker compose up -d` 실행 후 아래 `schema.sql`이 자동 적용됩니다.

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
# storage.py — psycopg3 + ConnectionPool (replaces psycopg2)
import json
import logging
import psycopg
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from config import settings
from exceptions import StorageError
from chunking import Chunk

logger = logging.getLogger(__name__)

_RRF_SQL = """
WITH semantic AS (
    SELECT id, content, context, summary, metadata,
           ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS rank_s
    FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s
),
keyword AS (
    SELECT id,
           ROW_NUMBER() OVER (ORDER BY ts_rank(search_vector, plainto_tsquery('simple', %s)) DESC) AS rank_k
    FROM chunks
    WHERE search_vector @@ plainto_tsquery('simple', %s)
    LIMIT %s
)
SELECT s.id, s.content, s.context, s.summary, s.metadata,
       (1.0/(60+s.rank_s)) + COALESCE(1.0/(60+k.rank_k), 0) AS rrf_score
FROM semantic s
LEFT JOIN keyword k ON s.id = k.id
ORDER BY rrf_score DESC LIMIT %s
"""

class ChunkStore:
    _pool: ConnectionPool | None = None

    def __init__(self):
        if ChunkStore._pool is None:
            try:
                ChunkStore._pool = ConnectionPool(
                    settings.database_url,
                    min_size=settings.db_pool_min,
                    max_size=settings.db_pool_max,
                    open=True,
                )
                logger.info("DB connection pool initialized (min=%d max=%d)",
                            settings.db_pool_min, settings.db_pool_max)
            except Exception as e:
                raise StorageError(f"Failed to create connection pool: {e}") from e

    def store(self, chunk: Chunk) -> None:
        with ChunkStore._pool.connection() as conn:
            conn.execute(
                "INSERT INTO chunks (content, context, summary, metadata, embedding) "
                "VALUES (%s, %s, %s, %s, %s)",
                (chunk.content, chunk.context, chunk.summary,
                 json.dumps(chunk.metadata), chunk.embedding),
            )

    def store_batch(self, chunks: list[Chunk]) -> None:
        """Batch insert — faster than individual store() calls."""
        with ChunkStore._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (content, context, summary, metadata, embedding) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    [(c.content, c.context, c.summary,
                      json.dumps(c.metadata), c.embedding) for c in chunks],
                )

    def hybrid_search(self, query_embedding: list, query_text: str, top_k: int = 20) -> list[dict]:
        """Reciprocal Rank Fusion (RRF) of semantic + keyword search."""
        with ChunkStore._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(_RRF_SQL, (
                    query_embedding, query_embedding, top_k * 2,
                    query_text, query_text, top_k * 2, top_k,
                ))
                return cur.fetchall()

    def delete_by_source(self, source: str) -> None:
        with ChunkStore._pool.connection() as conn:
            conn.execute("DELETE FROM chunks WHERE metadata->>'source' = %s", (source,))

    @classmethod
    def close_pool(cls) -> None:
        if cls._pool:
            cls._pool.close()
            cls._pool = None
            logger.info("DB connection pool closed")
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

import json
import re
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class Verdict(str, Enum):
    CORRECT   = "CORRECT"
    INCORRECT = "INCORRECT"
    AMBIGUOUS = "AMBIGUOUS"

@dataclass
class CRAGResult:
    verdict: Verdict
    confidence: float = 0.5
    reason: str = ""

# JSON-first prompt — more stable than "respond with one word"
EVAL_PROMPT = """You are a factual relevance evaluator.

Query: {query}

Retrieved document:
\"\"\"
{document}
\"\"\"

Does this document contain information relevant to the query?

Respond ONLY with valid JSON (no other text):
{{
  "verdict": "CORRECT" | "INCORRECT" | "AMBIGUOUS",
  "confidence": <float 0.0-1.0>,
  "reason": "<one sentence>"
}}"""

def _parse_verdict(response: str) -> CRAGResult:
    """JSON-first parsing with string-match fallback for robustness."""
    json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return CRAGResult(
                verdict=Verdict(data["verdict"].upper()),
                confidence=float(data.get("confidence", 0.5)),
                reason=data.get("reason", ""),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.debug("JSON parse failed, falling back to string match")

    # Fallback: keyword matching (backward compatible)
    upper = response.upper()
    if "INCORRECT" in upper: return CRAGResult(Verdict.INCORRECT, 0.5)
    if "AMBIGUOUS" in upper: return CRAGResult(Verdict.AMBIGUOUS, 0.5)
    return CRAGResult(Verdict.CORRECT, 0.5)

def evaluate_retrieval(query: str, chunks: list[dict]) -> tuple[list[dict], str]:
    """Evaluate and correct retrieval results with structured verdicts."""
    correct, ambiguous = [], []

    for chunk in chunks:
        response = llm(
            EVAL_PROMPT.format(query=query, document=chunk['content'][:1000]),
            max_tokens=150,
        )
        result = _parse_verdict(response)

        if result.verdict == Verdict.CORRECT:
            correct.append(chunk)
        elif result.verdict == Verdict.AMBIGUOUS:
            ambiguous.append(chunk)

    if correct:   return correct, "correct"
    if ambiguous: return ambiguous, "ambiguous"
    return [], "incorrect"
```

### Step 8: Full Query Pipeline

```python
# pipeline.py — with input validation + prompt injection defense
import re
import logging
import requests
from pydantic import BaseModel, field_validator
from llm import llm
from embedding import embed_query
from reranker import rerank
from crag import evaluate_retrieval
from storage import ChunkStore
from exceptions import ValidationError
from config import settings

def _web_search_fallback(query: str, top_k: int = 5) -> list[dict]:
    """CRAG 웹 폴백: DuckDuckGo API (무료, 키 불필요).
    CRAG 논문 원본 동작 — 내부 검색 실패 시 웹으로 보완."""
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=5,
        )
        data = resp.json()
        results = []
        for item in data.get("RelatedTopics", [])[:top_k]:
            if isinstance(item, dict) and item.get("Text"):
                results.append({
                    "content": item["Text"],
                    "context": "",
                    "summary": item["Text"][:200],
                    "metadata": {"source": "web", "url": item.get("FirstURL", "")},
                })
        return results
    except Exception as e:
        logger.warning(f"Web search fallback failed: {e}")
        return []

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Answer based on the provided context. "
    "Cite sources using [Source: filename] format. "
    "If the context doesn't contain enough information, say so clearly."
)

# Prompt injection patterns to block
_INJECTION_PATTERNS = [
    r'ignore\s+(previous|all|above)',
    r'forget\s+(everything|all|previous)',
    r'\bsystem\s*:',
    r'\bassistant\s*:',
    r'<\s*(system|assistant|user)\s*>',
    r'you\s+are\s+now',
]

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_crag: bool = True

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        v = v.strip().replace('\x00', '').replace('\r', '')
        if not v:
            raise ValueError("Query cannot be empty")
        if len(v) > settings.max_query_length:
            raise ValueError(f"Query too long (max {settings.max_query_length} chars)")
        for pattern in _INJECTION_PATTERNS:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(f"Query contains disallowed pattern: {pattern}")
        return v

def query(user_query: str, store: "ChunkStore", top_k: int = 5, use_crag: bool = True) -> dict:
    """Complete RAG query pipeline with input validation."""
    req = QueryRequest(query=user_query, top_k=top_k, use_crag=use_crag)

    # 1. Embed query
    query_emb = embed_query(req.query)

    # 2. Hybrid search (RRF: semantic + keyword)
    candidates = store.hybrid_search(query_emb, req.query, top_k=req.top_k * 3)

    # 3. Rerank with Voyage rerank-2
    reranked = rerank(req.query, candidates, top_k=req.top_k * 2)

    # 4. CRAG validation (optional but recommended)
    if req.use_crag:
        validated, status = evaluate_retrieval(req.query, reranked)
        if status == "incorrect":
            # Web search fallback (CRAG paper 원본 동작)
            web_chunks = _web_search_fallback(req.query, top_k=req.top_k)
            if web_chunks:
                final_chunks = web_chunks
            else:
                return {"answer": "관련 문서를 찾지 못했습니다.", "sources": [], "status": "no_results"}
        else:
            final_chunks = validated[:req.top_k]
    else:
        final_chunks = reranked[:req.top_k]

    # 5. Build context from ORIGINAL content (not summaries)
    context = "\n\n---\n\n".join([
        f"[Source: {c['metadata'].get('source', '?')}]\n{c['context']}\n\n{c['content']}"
        for c in final_chunks
    ])

    # 6. Generate — context and query are structurally separated (injection mitigation)
    answer = llm(
        f"{SYSTEM_PROMPT}\n\n<context>\n{context}\n</context>\n\n<question>{req.query}</question>",
        max_tokens=4096,
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

> **Note:** Agentic RAG requires Anthropic tool_use today, so set `RAG_LLM_MODE=claude-api`.
> CLI mode (`claude-cli`) does not support tool_use.

```python
# agentic_rag.py
"""Agent-driven retrieval: plan → search → validate → re-search if needed.
Requires API mode (RAG_LLM_MODE=claude-api).
Implementation below uses Anthropic Messages API.
"""
from llm import get_api_client
client, model_name, provider = get_api_client()
if provider != "anthropic":
    raise NotImplementedError("Agentic RAG currently supports only RAG_LLM_MODE=claude-api in this template.")

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
        response = client.messages.create(
            model=model_name,
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
from embedding import embed_query
from reranker import rerank
from llm import llm

langfuse = Langfuse()

def traced_query(user_query: str, store: ChunkStore) -> dict:
    trace = langfuse.trace(name="rag-query", input=user_query)

    # Track retrieval
    retrieval_span = trace.span(name="hybrid-search")
    query_emb = embed_query(user_query)
    candidates = store.hybrid_search(query_emb, user_query, top_k=20)
    retrieval_span.end(output={"count": len(candidates)})

    # Track reranking
    rerank_span = trace.span(name="rerank")
    reranked = rerank(user_query, candidates, top_k=5)
    rerank_span.end(output={"count": len(reranked)})

    # Track generation
    gen_span = trace.span(name="generation")
    context = "\n\n---\n\n".join([c['content'][:500] for c in reranked])
    answer = llm(f"Context:\n{context}\n\nQuestion: {user_query}", max_tokens=4096)
    gen_span.end(output={"answer_length": len(answer)})

    trace.update(output=answer)
    return {"answer": answer, "sources": [c['metadata'] for c in reranked]}
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
# Core
anthropic>=0.45.0,<1.0
voyageai>=0.3.0,<1.0
pgvector>=0.3.0,<1.0
psycopg[binary]>=3.1,<4.0       # psycopg3 (replaces psycopg2)
psycopg-pool>=3.1,<4.0          # ConnectionPool
pydantic>=2.5,<3.0
pydantic-settings>=2.0,<3.0     # BaseSettings + .env support
tenacity>=8.2,<10.0             # Retry with exponential backoff
fastapi>=0.115.0,<1.0
ragas>=0.2.0,<1.0
langfuse>=2.0,<4.0
datasets>=3.0,<4.0
# Multi-LLM (install as needed based on RAG_LLM_MODE)
google-genai>=1.0.0,<2.0         # RAG_LLM_MODE=gemini (신규 SDK: from google import genai)
openai>=1.50.0,<2.0              # RAG_LLM_MODE=openai
# Development / Testing
pytest>=8.0,<9.0
pytest-cov>=5.0,<6.0
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

## Automation — Claude Code가 자동으로 하는 것

이 스킬을 사용하면 Claude Code가 아래를 **전부 자동 생성 + 실행**합니다:

| 작업 | Claude Code 자동 | 사용자 |
|------|:----------------:|:------:|
| 프로젝트 구조 생성 | ✅ | |
| `docker-compose.yml` (PostgreSQL + pgvector) | ✅ | |
| `docker compose up -d` 실행 | ✅ | |
| DB 스키마 생성 (`schema.sql` 실행) | ✅ | |
| 파이프라인 코드 전체 생성 | ✅ | |
| `.env` 파일 생성 + API 키 설정 | ✅ | |
| `ingest.py` 생성 + 인제스션 실행 | ✅ | |
| 챗봇 API 서버 생성 (`app.py`) | ✅ | |
| PDF 파일 준비 | | ✅ |
| Voyage API 키 발급 (무료 가입) | | ✅ |

**사용자는 PDF 파일 + Voyage API 키만 준비하면 됩니다.** 나머지는 전부 자동.

---

## ingest.py 자동 생성 가이드

질문 흐름(Phase 1-4) 완료 후 Claude Code가 선택에 맞는 `ingest.py`를 생성합니다.

### Small 규모 (< 50p)

```python
# ingest.py — Small: no chunking, direct embedding
import sys, os
from pathlib import Path
from config import settings
from chunking import semantic_chunk
from enrichment import enrich_chunk
from embedding import embed_chunks
from storage import ChunkStore

def load_document(path: str) -> str:
    import fitz  # pymupdf
    doc = fitz.open(path)
    return "\n\n".join(page.get_text() for page in doc)

def ingest(file_path: str):
    store = ChunkStore()
    text = load_document(file_path)
    chunks = semantic_chunk(text, source=Path(file_path).name, max_tokens=8192)  # large chunks
    for chunk in chunks:
        enrich_chunk(text, chunk)
    embed_chunks(chunks)
    store.store_batch(chunks)
    print(f"Ingested {len(chunks)} chunks from {file_path}")

if __name__ == "__main__":
    for path in sys.argv[1:]:
        ingest(path)
    ChunkStore.close_pool()
```

### Medium 규모 (50-500p)

```python
# ingest.py — Medium: semantic_chunk + contextual enrichment
import sys
from pathlib import Path
from config import settings
from chunking import semantic_chunk
from enrichment import enrich_chunk
from embedding import embed_chunks
from storage import ChunkStore

def load_document(path: str) -> str:
    import fitz
    doc = fitz.open(path)
    return "\n\n".join(page.get_text() for page in doc)

def ingest(file_path: str):
    store = ChunkStore()
    text = load_document(file_path)
    chunks = semantic_chunk(text, source=Path(file_path).name)
    print(f"  Chunked: {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        enrich_chunk(text, chunk)       # LLM context + summary per chunk
        if (i + 1) % 10 == 0:
            print(f"  Enriched {i+1}/{len(chunks)}...")
    embed_chunks(chunks)
    store.store_batch(chunks)
    print(f"  Stored {len(chunks)} chunks")

if __name__ == "__main__":
    paths = list(sys.argv[1:])
    if not paths:
        print("Usage: python ingest.py <file.pdf> [file2.pdf ...]")
        sys.exit(1)
    for path in paths:
        print(f"Ingesting {path}...")
        ingest(path)
    ChunkStore.close_pool()
    print("Done.")
```

### Large 규모 (500p+)

> **임베딩 모델**: `voyage-context-3` (`/v1/contextualizedembeddings`)
> 긴 문서는 청크 간 맥락 보존이 필수입니다. voyage-4-large(독립 임베딩)가 아닌
> voyage-context-3(문서 전체 맥락 자동 주입)를 사용하세요.
> Jina-v3 late chunking 대비 +23.66% 성능 향상.

```python
# ingest.py — Large: voyage-context-3 + parallel batch processing
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import voyageai
from chunking import late_chunk
from enrichment import enrich_chunk
from storage import ChunkStore

def load_document(path: str) -> str:
    import fitz
    return "\n\n".join(page.get_text() for page in fitz.open(path))

def process_file(file_path: str, store: ChunkStore):
    vo = voyageai.Client()
    text = load_document(file_path)
    # use_late_chunking=True → voyage-context-3 사용 (Large 규모 필수)
    chunks = late_chunk(text, Path(file_path).name, vo, use_late_chunking=True)
    for chunk in chunks:
        enrich_chunk(text, chunk)
    # NOTE: embed_chunks()는 voyage-4-large를 쓰므로 Large에선 호출 불필요
    # late_chunk()가 이미 voyage-context-3로 임베딩까지 완료함
    store.store_batch(chunks)
    return len(chunks)

if __name__ == "__main__":
    paths = sys.argv[1:]
    store = ChunkStore()
    workers = min(4, len(paths))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_file, p, store): p for p in paths}
        for future in as_completed(futures):
            path = futures[future]
            count = future.result()
            print(f"{path}: {count} chunks ingested")
    ChunkStore.close_pool()
```

### 실행 순서 (Claude Code가 자동 수행)

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. Docker DB 시작
docker compose up -d
sleep 3  # DB 초기화 대기

# 3. 문서 인제스션
python ingest.py ./data/*.pdf

# 4. API 서버 시작
python app.py  # FastAPI on :8000
```

---

## TESTING

Production-grade RAG pipelines need unit tests. This section provides the testing scaffold so Claude Code can generate a complete test suite alongside the pipeline.

### Two-Layer Architecture

```
Layer 1: validate_skill.py  — SKILL.md 문법 + 필수 패턴 검증 (py_compile + regex)
Layer 2: pytest unit tests  — 외부 API 전부 mock, 순수 로직만 검증
```

### Mock Targets (중요)

Python의 `from x import y` 패턴은 patch 대상이 달라진다:

| 모듈이 사용하는 방식 | patch 대상 |
|---------------------|-----------|
| `from llm import llm` (crag.py) | `patch("crag.llm")` |
| `voyageai.Client()` (embedding.py) | `patch("embedding.vo")` |
| `from psycopg_pool import ConnectionPool` (storage.py) | `patch("storage.ConnectionPool")` |
| `_embed_batch_raw` (embedding.py 내부) | `patch("embedding._embed_batch_raw")` |

### conftest.py (SKILL.md Named Block 추출기)

SKILL.md 코드 블록을 pytest 실행 전에 임시 디렉토리로 추출해 `sys.path`에 추가한다.
이 방식 덕분에 test 파일 최상단에서 `from crag import _parse_verdict` 같은 import가 작동.

```python
# Example: root conftest.py (place at project root, NOT in tests/)
"""
루트 conftest.py — SKILL.md Python 블록을 conftest 로드 시점에 추출해 sys.path에 추가.
"""

import re
import sys
import atexit
import shutil
import tempfile
from pathlib import Path

import pytest

SKILL_MD = Path(__file__).parent / "skill" / "SKILL.md"


def _extract_named_blocks(skill_path: Path) -> dict[str, str]:
    text = skill_path.read_text(encoding="utf-8")
    raw_blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)

    blocks = {}
    for block in raw_blocks:
        first_line = block.split("\n")[0].strip()
        match = re.match(r"^#\s+([\w]+\.py)", first_line)
        if match:
            name = match.group(1)
            if name not in blocks:
                blocks[name] = block
    return blocks


def _setup_skill_modules() -> str:
    """SKILL.md 블록을 임시 디렉토리에 쓰고 경로 반환."""
    tmp_dir = tempfile.mkdtemp(prefix="ragskill_modules_")
    blocks = _extract_named_blocks(SKILL_MD)
    for filename, code in blocks.items():
        Path(tmp_dir, filename).write_text(code, encoding="utf-8")
    return tmp_dir


# conftest.py 로드(= collection 전) 시점에 즉시 실행
_modules_dir = _setup_skill_modules()
sys.path.insert(0, _modules_dir)

# 프로세스 종료 시 임시 디렉토리 정리
atexit.register(shutil.rmtree, _modules_dir, True)


@pytest.fixture
def mock_voyage():
    from unittest.mock import MagicMock, patch
    with patch("voyageai.Client") as mock:
        client = MagicMock()
        client.embed.return_value = MagicMock(embeddings=[[0.1] * 1024])
        client.rerank.return_value = MagicMock(
            results=[MagicMock(index=0, relevance_score=0.9)]
        )
        client.embed_chunks.return_value = MagicMock(embeddings=[[[0.1] * 1024]])
        mock.return_value = client
        yield client


@pytest.fixture
def mock_pool():
    """ConnectionPool 클래스 mock을 yield (인스턴스가 아닌 클래스).
    mock_pool          = ConnectionPool 클래스 mock (call_count 확인용)
    mock_pool.return_value = pool 인스턴스
    mock_pool.return_value.connection.return_value = conn
    """
    from unittest.mock import MagicMock, patch
    with patch("storage.ConnectionPool") as mock_cls:
        pool = MagicMock()
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = [
            {
                "id": "1",
                "content": "test content",
                "context": "test context",
                "summary": "test summary",
                "metadata": {"source": "test.pdf"},
                "rrf_score": 0.9,
            }
        ]
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        conn.cursor.return_value = cursor
        pool.connection.return_value = conn
        mock_cls.return_value = pool
        yield mock_cls  # ⚠️ 반드시 mock_cls를 yield (pool이 아님)


@pytest.fixture
def sample_chunks():
    from chunking import Chunk
    return [
        Chunk(
            content="Exercise improves cardiovascular health.",
            metadata={"source": "test.pdf", "chunk_index": 0},
            context="From a sports science paper about exercise.",
            summary="Exercise benefits cardiovascular system.",
            embedding=[0.1] * 1024,
        ),
        Chunk(
            content="Mental toughness helps athletes perform under pressure.",
            metadata={"source": "mental.pdf", "chunk_index": 0},
            context="From a sports psychology paper.",
            summary="Mental toughness and performance.",
            embedding=[0.2] * 1024,
        ),
    ]


@pytest.fixture
def sample_chunk_dicts():
    return [
        {
            "id": "1",
            "content": "Exercise improves cardiovascular health and VO2max.",
            "context": "From a sports physiology paper on aerobic capacity.",
            "summary": "Exercise cardiovascular benefits.",
            "metadata": {"source": "sports.pdf"},
            "rrf_score": 0.9,
        },
        {
            "id": "2",
            "content": "Mental toughness is a key predictor of athletic success.",
            "context": "From a sports psychology study.",
            "summary": "Mental toughness and athletic performance.",
            "metadata": {"source": "mental.pdf"},
            "rrf_score": 0.7,
        },
    ]
```

> **⚠️ 주의**: `tests/conftest.py`와 루트 `conftest.py`에 같은 fixture 이름이 있으면 하위 디렉토리 conftest 우선. 중복 정의 금지.

### validate_skill.py (Layer 1)

```python
# tests/validate_skill.py
"""
SKILL.md Named Block 검증 — py_compile + 필수 패턴 체크
"""
import re
import sys
import py_compile
import tempfile
from pathlib import Path

SKILL_MD = Path(__file__).parent.parent / "skill" / "SKILL.md"

REQUIRED_PATTERNS: dict[str, list[str]] = {
    "pipeline.py": [
        "from embedding import embed_query",
        "from reranker import rerank",
        "from crag import evaluate_retrieval",
        "from storage import ChunkStore",
    ],
    "embedding.py": ["from chunking import Chunk"],
    "storage.py": ["from chunking import Chunk"],
    "config.py": ['voyage_api_key: str = ""'],
    "crag.py": ["class Verdict"],
}


def extract_named_blocks(skill_path: Path) -> dict[str, str]:
    text = skill_path.read_text(encoding="utf-8")
    raw_blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    blocks = {}
    for block in raw_blocks:
        first_line = block.split("\n")[0].strip()
        match = re.match(r"^#\s+([\w]+\.py)", first_line)
        if match:
            name = match.group(1)
            if name not in blocks:
                blocks[name] = block
    return blocks


def check_syntax(name: str, code: str) -> list[str]:
    errors = []
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        tmp_path = f.name
    try:
        py_compile.compile(tmp_path, doraise=True)
    except py_compile.PyCompileError as e:
        errors.append(f"SyntaxError in {name}: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return errors


def check_required_patterns(name: str, code: str) -> list[str]:
    warnings = []
    patterns = REQUIRED_PATTERNS.get(name, [])
    for pattern in patterns:
        if pattern not in code:
            warnings.append(f"WARN [{name}]: missing '{pattern}'")
    return warnings


def main() -> int:
    blocks = extract_named_blocks(SKILL_MD)
    all_errors, all_warnings = [], []

    for name, code in blocks.items():
        all_errors.extend(check_syntax(name, code))
        all_warnings.extend(check_required_patterns(name, code))

    for w in all_warnings:
        print(w)
    for e in all_errors:
        print(e, file=sys.stderr)

    print(f"\n{len(all_errors)} errors, {len(all_warnings)} warnings — {'PASS' if not all_errors else 'FAIL'}")
    return 0 if not all_errors else 1


if __name__ == "__main__":
    sys.exit(main())
```

### GitHub Actions CI

```yaml
# .github/workflows/ci.yml
name: RAG Skill CI

on:
  push:
    branches: ["**"]
  pull_request:

jobs:
  validate-skill:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: python tests/validate_skill.py

  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install test dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt
      - name: Run unit tests with coverage
        run: pytest tests/unit/ -v --cov=. --cov-report=xml --cov-fail-under=60
```

### requirements-dev.txt

```
pytest>=8.0,<9.0
pytest-cov>=5.0,<6.0
```

### 실행 명령

```bash
# SKILL.md 검증
python tests/validate_skill.py

# 단위 테스트 (coverage 포함)
pytest tests/unit/ -v --cov=. --cov-fail-under=60

# 특정 모듈만
pytest tests/unit/test_crag.py -v
```

---

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
