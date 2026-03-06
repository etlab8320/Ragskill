# Plan: rag-skill-test-completion

> RAG 스킬 테스트 커버리지 완성 — Evaluation, Monitoring, Agentic RAG

## Executive Summary

| Item | Detail |
|------|--------|
| Feature | rag-skill-test-completion |
| Created | 2026-03-06 |
| Target | v1.5.0 (테스트 완성) |

### Value Delivered

| Perspective | Description |
|-------------|-------------|
| **Problem** | SKILL.md에 코드가 있지만 Evaluation, Monitoring, Agentic RAG 3개 모듈에 테스트가 없어 스킬 신뢰도 저하 |
| **Solution** | 기존 테스트 패턴(mock + 순수 함수 직접 테스트)으로 3개 모듈 테스트 추가 |
| **Function UX Effect** | 97개 → ~140개 테스트, 모든 SKILL.md 모듈에 대응 테스트 존재 |
| **Core Value** | GitHub에 공개된 스킬의 완성도와 신뢰성 확보 |

---

## 1. Background

현재 RAG Pipeline 스킬 구조:
- `skill/SKILL.md` — 레퍼런스 코드 (named Python blocks)
- `conftest.py` — SKILL.md에서 블록 추출 → 임시 디렉토리 → sys.path 등록
- `tests/unit/` — 8개 테스트 파일, 97개 테스트

### 테스트 없는 모듈

| 모듈 | SKILL.md 위치 | 핵심 함수 | 이슈 |
|------|---------------|-----------|------|
| `evaluation.py` | L1828-1910 | `_get_ragas_llm()`, `evaluate_rag()` | 외부 의존성(RAGAS, Langchain) mock 필요 |
| `monitoring.py` | L1969-1999 | `traced_query()` | 외부 의존성(Langfuse) mock 필요 |
| `agentic_rag.py` | L1025-1104 | `agentic_query()` | `get_document_list` 스텁, import 누락 |

---

## 2. Goals

1. **test_evaluation.py** — `_get_ragas_llm()` LLM 선택 로직 + `evaluate_rag()` 호출 검증 (~15개 테스트)
2. **test_monitoring.py** — Langfuse trace/span 생성 검증 (~10개 테스트)
3. **test_agentic_rag.py** — tool dispatch, 반복 로직, 스텁 수정 검증 (~15개 테스트)
4. **agentic_rag.py 스텁 수정** — `get_document_list` 실제 구현으로 교체

---

## 3. Scope

### In Scope
- 3개 테스트 파일 생성 (tests/unit/)
- agentic_rag.py `get_document_list` 스텁을 실제 로직으로 교체 (SKILL.md 내)
- 기존 conftest.py 패턴 (mock_pool, mock_llm 등) 활용

### Out of Scope
- Self-RAG, Multimodal, Adaptive RAG (메뉴 소개 항목 — 구현 불필요)
- 실제 API 호출 테스트 (모두 mock)
- CI/CD 설정

---

## 4. Test Strategy

기존 패턴 준수:
- **순수 함수** → 직접 테스트 (mock 불필요)
- **외부 의존성** → `unittest.mock.patch`로 모듈 레벨 mock
- **DB 연동** → `conftest.py`의 `mock_pool` fixture 재사용

### 4.1 test_evaluation.py (~15 tests)

| 테스트 | 대상 | Mock |
|--------|------|------|
| `_get_ragas_llm()` gemini 경로 | LLM 선택 | `langchain_google_genai` |
| `_get_ragas_llm()` openai 경로 | LLM 선택 | `langchain_openai` |
| `_get_ragas_llm()` claude-api 경로 | LLM 선택 | `langchain_anthropic` |
| `_get_ragas_llm()` 잘못된 mode | ValueError raise | settings mock |
| `evaluate_rag()` 정상 실행 | RAGAS evaluate 호출 | `ragas.evaluate` |
| `evaluate_rag()` 빈 입력 | edge case | `ragas.evaluate` |
| `SingleTurnSample` 생성 검증 | 데이터 구조 | `ragas` |

### 4.2 test_monitoring.py (~10 tests)

| 테스트 | 대상 | Mock |
|--------|------|------|
| `traced_query()` trace 생성 | Langfuse trace | `langfuse.Langfuse` |
| span 3단계 (search, rerank, gen) | span lifecycle | Langfuse + embed + rerank |
| 결과 구조 (answer, sources) | 반환값 | 전체 mock |

### 4.3 test_agentic_rag.py (~15 tests)

| 테스트 | 대상 | Mock |
|--------|------|------|
| `search_documents` tool dispatch | tool routing | Anthropic client |
| `get_document_list` tool dispatch | tool routing (수정 후) | Anthropic client + store |
| `end_turn` 정상 종료 | 대화 종료 | Anthropic client |
| max iterations (5) 도달 | 루프 제한 | Anthropic client |
| TOOLS 스키마 검증 | 정적 구조 | 없음 |
| AGENT_SYSTEM 프롬프트 존재 | 정적 구조 | 없음 |

---

## 5. Implementation Order

1. **agentic_rag.py 스텁 수정** (SKILL.md 편집)
2. **test_evaluation.py** 작성
3. **test_monitoring.py** 작성
4. **test_agentic_rag.py** 작성
5. 전체 테스트 실행 + 확인

---

## 6. Risk

| Risk | Mitigation |
|------|-----------|
| RAGAS import 구조 변경 | mock으로 격리, 실제 패키지 불필요 |
| agentic_rag.py module-level 코드 실행 | import 시 mock 필요 (conftest에서 처리) |
| conftest.py 블록 추출 패턴 의존 | 기존 `# module.py` 네이밍 규칙 유지 |
