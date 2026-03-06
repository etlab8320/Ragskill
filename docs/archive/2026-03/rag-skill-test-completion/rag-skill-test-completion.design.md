# Design: rag-skill-test-completion

> Evaluation, Monitoring, Agentic RAG 테스트 + 스텁 수정 상세 설계

## 1. Overview

Plan에서 정의한 3개 모듈 테스트 + 1개 스텁 수정의 구현 상세.

**변경 파일:**

| File | Action | Description |
|------|--------|-------------|
| `skill/SKILL.md` | Edit | agentic_rag.py `get_document_list` 스텁 → 실제 구현 |
| `tests/unit/test_evaluation.py` | Create | RAGAS evaluation 테스트 |
| `tests/unit/test_monitoring.py` | Create | Langfuse monitoring 테스트 |
| `tests/unit/test_agentic_rag.py` | Create | Agentic RAG 테스트 |
| `conftest.py` | Edit | agentic_rag.py import 에러 방지 fixture 추가 |

---

## 2. SKILL.md 수정: agentic_rag.py 스텁

### 2.1 현재 (L1094-1096)

```python
elif block.name == "get_document_list":
    # Return unique sources
    tool_result = "Available sources: ..."
```

### 2.2 수정 후

```python
elif block.name == "get_document_list":
    sources = store.get_unique_sources()
    tool_result = "Available sources:\n" + "\n".join(
        f"- {s}" for s in sources
    ) if sources else "No documents ingested yet."
```

### 2.3 import 보완

agentic_rag.py 모듈 상단에 누락된 import 추가:

```python
from embedding import embed_query
from reranker import rerank
```

> Note: `ChunkStore`는 함수 인자로 받으므로 import 불필요 (duck typing).

---

## 3. test_evaluation.py 설계

### 3.1 Mock 전략

evaluation.py는 module-level import가 많음 (`ragas`, `langchain_*`). conftest.py의 블록 추출로 파일은 생성되지만, 실제 패키지가 없으면 import 실패.

**해결**: `sys.modules` 사전에 mock 모듈 등록 (테스트 파일 상단에서).

```python
# test_evaluation.py 상단 — ragas/langchain mock 등록
import sys
from unittest.mock import MagicMock

# Pre-register mock modules before evaluation.py import
for mod in [
    "ragas", "ragas.evaluate", "ragas.dataset_schema",
    "ragas.metrics", "ragas.metrics.collections", "ragas.llms",
    "langchain_google_genai", "langchain_openai", "langchain_anthropic",
]:
    sys.modules.setdefault(mod, MagicMock())
```

### 3.2 테스트 목록

| Class | Test | 검증 포인트 |
|-------|------|------------|
| `TestGetRagasLlm` | `test_gemini_mode` | settings.rag_llm_mode="gemini" → ChatGoogleGenerativeAI 생성 |
| | `test_openai_mode` | settings.rag_llm_mode="openai" → ChatOpenAI 생성 |
| | `test_claude_api_mode` | settings.rag_llm_mode="claude-api" → ChatAnthropic 생성 |
| | `test_invalid_mode_raises` | rag_llm_mode="ollama" → ValueError |
| | `test_gemini_uses_env_model` | RAG_LLM_MODEL 환경변수 반영 |
| | `test_openai_uses_env_model` | RAG_LLM_MODEL 환경변수 반영 |
| | `test_claude_uses_env_model` | RAG_LLM_MODEL 환경변수 반영 |
| | `test_gemini_default_model` | 기본 모델 "gemini-2.0-flash" |
| | `test_claude_default_model` | 기본 모델 "claude-haiku-4-5-20251001" |
| `TestEvaluateRag` | `test_creates_samples` | SingleTurnSample 올바르게 생성 |
| | `test_calls_ragas_evaluate` | ragas.evaluate 호출됨 |
| | `test_passes_four_metrics` | faithfulness, context_precision, context_recall, answer_relevancy |
| | `test_returns_result` | evaluate 반환값 그대로 반환 |
| | `test_zip_alignment` | questions/answers/contexts/ground_truths 길이 일치 검증 |
| | `test_empty_input` | 빈 리스트 → 빈 dataset |

**총 15개 테스트**

---

## 4. test_monitoring.py 설계

### 4.1 Mock 전략

monitoring.py는 `langfuse`, `embedding`, `reranker`, `llm`, `storage` 모듈에 의존.

```python
# test_monitoring.py 상단
import sys
from unittest.mock import MagicMock

sys.modules.setdefault("langfuse", MagicMock())
```

### 4.2 테스트 목록

| Class | Test | 검증 포인트 |
|-------|------|------------|
| `TestTracedQuery` | `test_creates_trace` | langfuse.trace(name="rag-query") 호출 |
| | `test_trace_receives_input` | trace input = user_query |
| | `test_three_spans_created` | hybrid-search, rerank, generation 3개 span |
| | `test_search_span_output` | retrieval span end에 count 포함 |
| | `test_rerank_span_output` | rerank span end에 count 포함 |
| | `test_gen_span_output` | gen span end에 answer_length 포함 |
| | `test_returns_answer_and_sources` | {"answer": str, "sources": list} |
| | `test_trace_update_with_answer` | trace.update(output=answer) 호출 |
| | `test_sources_from_metadata` | sources = reranked 결과의 metadata 리스트 |
| | `test_context_truncated_to_500` | 각 청크 content[:500] 적용 |

**총 10개 테스트**

---

## 5. test_agentic_rag.py 설계

### 5.1 Mock 전략

agentic_rag.py는 **module-level 코드 실행** 문제가 있음:

```python
client, model_name, provider = get_api_client()
if provider != "anthropic":
    raise NotImplementedError(...)
```

**해결**: import 전에 `llm.get_api_client`를 mock.

```python
# test_agentic_rag.py 상단
import sys
from unittest.mock import MagicMock, patch

# Mock llm module before agentic_rag imports it
mock_llm_mod = MagicMock()
mock_llm_mod.get_api_client.return_value = (MagicMock(), "claude-haiku-4-5-20251001", "anthropic")
sys.modules["llm"] = mock_llm_mod
```

### 5.2 테스트 목록

| Class | Test | 검증 포인트 |
|-------|------|------------|
| `TestToolSchema` | `test_tools_has_two_entries` | TOOLS 리스트 길이 2 |
| | `test_search_tool_schema` | search_documents name + input_schema 구조 |
| | `test_document_list_tool_schema` | get_document_list name + input_schema |
| `TestAgentSystem` | `test_system_prompt_exists` | AGENT_SYSTEM 비어있지 않음 |
| | `test_system_prompt_mentions_iterations` | "max 3 iterations" 포함 |
| `TestAgenticQuery` | `test_end_turn_returns_text` | stop_reason="end_turn" → content[0].text 반환 |
| | `test_search_documents_dispatched` | tool_use block → embed_query + hybrid_search + rerank 호출 |
| | `test_get_document_list_dispatched` | tool_use block → store.get_unique_sources 호출 |
| | `test_get_document_list_empty` | sources 빈 리스트 → "No documents" 메시지 |
| | `test_max_iterations_returns_message` | 5회 루프 후 "Max iterations reached." 반환 |
| | `test_tool_result_appended_to_messages` | messages에 assistant+tool_result 추가됨 |
| | `test_reranked_results_formatted` | "[source] content" 포맷 |
| | `test_search_default_top_k` | top_k 미지정시 기본값 5 |
| | `test_multiple_tool_calls_in_sequence` | 2회 tool_use → 최종 end_turn |
| | `test_provider_not_anthropic_raises` | provider="openai" → NotImplementedError |

**총 15개 테스트**

---

## 6. conftest.py 수정

agentic_rag.py의 module-level 실행 문제 대응. 기존 `_extract_named_blocks`는 그대로 유지하되, agentic_rag.py가 import될 때 `llm.get_api_client`가 mock되어 있어야 함.

**방법**: 각 테스트 파일 상단에서 `sys.modules` 조작 (conftest 수정 최소화).

conftest.py에는 신규 fixture만 추가:

```python
@pytest.fixture
def mock_anthropic_client():
    """Anthropic Messages API mock"""
    from unittest.mock import MagicMock
    client = MagicMock()
    return client
```

---

## 7. Implementation Order

```
1. SKILL.md 수정 — get_document_list 스텁 + import 보완
2. test_evaluation.py 생성 (15 tests)
3. test_monitoring.py 생성 (10 tests)
4. test_agentic_rag.py 생성 (15 tests)
5. conftest.py fixture 추가
6. 전체 pytest 실행 확인
```

**예상 결과**: 97 → 137개 테스트 (40개 추가)

---

## 8. 제약 사항

- 외부 패키지(ragas, langfuse, anthropic SDK) 설치 불필요 — 모두 mock
- 기존 테스트에 영향 없음 — 신규 파일만 추가
- conftest.py 변경 최소화 — sys.modules 조작은 각 테스트 파일 내에서
