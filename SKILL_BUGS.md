# SKILL.md 버그 및 개선사항 추적

이 문서는 ragskill-tests 구현 중 발견된 SKILL.md의 버그, 누락 사항, 개선 필요 사항을 기록합니다.

---

## ✅ 수정 완료

### BUG-01: storage.py — `from chunking import Chunk` 누락
- **파일**: `storage.py` 블록
- **증상**: `store_batch()` 실행 시 `NameError: name 'Chunk' is not defined`
- **원인**: storage.py에서 `Chunk` 타입을 사용하지만 import 구문이 없었음
- **수정**: SKILL.md storage.py 블록 상단에 `from chunking import Chunk` 추가
- **테스트**: `test_store_batch_uses_executemany` → PASS

### BUG-03: enrichment.py — `from chunking import Chunk` 누락
- **파일**: `enrichment.py` 블록
- **증상**: `smart_ingest` import 시 `NameError: name 'Chunk' is not defined`
- **원인**: `enrich_chunk(full_doc: str, chunk: Chunk)` 시그니처에 `Chunk` 사용하지만 import 없음
- **수정**: SKILL.md enrichment.py 블록에 `from chunking import Chunk` 추가
- **테스트**: `TestSmartIngest` 전체 → PASS (2026-02-21)

### BUG-02: llm.py — 구버전 google.generativeai SDK 사용
- **파일**: `llm.py` 블록 (`_call_gemini` 함수)
- **증상**: `import google.generativeai as genai` → `DeprecationWarning` 또는 `ImportError` (신규 설치 시)
- **원인**: google.generativeai는 deprecated. 신규 SDK는 `from google import genai; genai.Client()` 방식
- **수정**: `from google import genai; genai.Client(api_key=...)` + `types.GenerateContentConfig` ✅ FIXED
- **requirements**: `google-generativeai` → `google-genai>=1.0.0` ✅ FIXED

---

## ⚠️ 알려진 한계 (개선 권장)

### LIMIT-01: validate_skill.py — Example Block 필터링 불완전
- **파일**: `tests/validate_skill.py`
- **설명**: SKILL.md의 python 코드 블록 중 "# filename.py" 주석 없는 예시 블록(ingest.py 등 여러 버전)은 필터링하지만, 첫 번째로 매칭되는 블록만 사용 (중복 블록 처리 제한)
- **영향**: ingest.py가 Small/Medium/Large 3가지 버전으로 존재 → 첫 번째 버전만 추출
- **개선**: 버전 선택 로직 또는 합성 파일명(ingest_small.py 등) 도입 검토

### LIMIT-02: embedding.py — bge-m3 폴백 시 차원 고정
- **파일**: `embedding.py` 블록
- **설명**: `_USE_LOCAL = True`일 때 bge-m3는 1024차원 고정. `output_dimension` 파라미터 무시
- **영향**: Matryoshka 차원 최적화(512dim 등) 불가
- **개선**: 폴백 문서에 명시적으로 bge-m3 한계 기재 권장

### LIMIT-03: conftest.py — pytest fixture 충돌 주의
- **발견 경위**: ragskill-tests 구현 중 `tests/conftest.py`와 root `conftest.py`에 `mock_pool` 중복 정의 시 하위 디렉토리 conftest 우선 적용됨
- **증상**: `mock_pool.call_count == 0` (클래스 mock 대신 인스턴스가 주입됨)
- **교훈**: 다중 conftest.py 사용 시 동명 fixture 중복 정의 금지

### LIMIT-04: late_chunk() — embed_chunks() 호출 후 embed_chunks() 중복 호출 위험
- **파일**: Large ingest.py 예시
- **설명**: `late_chunk(use_late_chunking=True)` 호출 후 `embed_chunks()` 재호출 시 voyage-4-large로 덮어씌움
- **현재 주석**: `# NOTE: embed_chunks()는 voyage-4-large를 쓰므로 Large에선 호출 불필요`
- **개선**: 경고를 더 명확히 하거나 `embed_chunks()` 내부에서 이미 임베딩된 청크 스킵 로직 추가

---

## 📋 구현 누락 (미구현)

### ~~MISS-01: GraphRAG 실제 구현 코드 없음~~ ✅ FIXED (2026-02-21)
- `graph_rag.py` 블록 추가: `GraphStore`, `build_graph()`, `summarize_communities()`, `graph_augment()`
- `schema.sql`에 `graph_nodes`, `graph_edges`, `graph_communities` 테이블 추가
- BFS connected components 커뮤니티 탐지 + Recursive CTE 그래프 순회 구현

### MISS-02: ColPali Multimodal RAG 구현 코드 없음
- 설명만 있고 실제 코드 없음 (ColPali는 외부 라이브러리 의존성 높음)

### MISS-03: RAGAS 평가 코드 실행 검증 미완
- `evaluation.py` 블록은 있으나 실제 RAGAS API 변경으로 동작 보장 어려움
- `ragas>=0.2.0` 기준 검증 필요

---

## 🔧 테스트 구현 시 발견된 개선사항

### IMPROVE-01: crag.py — `from llm import llm` 패턴
- **설명**: crag.py가 `from llm import llm`으로 임포트하므로 mock 시 `patch("crag.llm")` 필요
- **교훈**: 모듈 레벨 임포트 시 patch 대상은 사용 모듈 기준 (`patch("crag.llm")`, `patch("embedding.vo")` 등)

### IMPROVE-02: ConnectionPool open=True 파라미터
- **설명**: psycopg_pool v3에서 `ConnectionPool(open=True)` → 즉시 연결 시도. 테스트 시 반드시 mock 필요
- **현재**: SKILL.md에 `open=True` 명시 → mock 없이 ChunkStore() 호출 시 실제 DB 연결 시도

---

*최종 업데이트: 2026-02-20*
*발견: ragskill-tests 구현 (feature/ragskill-tests)*
