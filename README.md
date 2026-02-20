# RAG Pipeline Skill for Claude Code

> 30+ 기법/논문/벤치마크 리서치 기반의 프로덕션 RAG 파이프라인 구축 스킬

Claude Code에서 `/rag-pipeline` 명령으로 사용합니다.
"RAG 만들어줘" 같은 요청을 하면 규모 판단 → 전략 선택 → 코드 생성까지 자동으로 가이드합니다.

## 설치

```bash
curl -fsSL https://raw.githubusercontent.com/etlab8320/Ragskill/v1.1.0/install.sh | bash
```

또는 수동 설치:

```bash
mkdir -p ~/.claude/skills/rag-pipeline
curl -fsSL https://raw.githubusercontent.com/etlab8320/Ragskill/v1.1.0/skill/SKILL.md \
  -o ~/.claude/skills/rag-pipeline/SKILL.md
```

## 사용법

Claude Code에서:

```
/rag-pipeline
```

자동으로 4단계 질문을 통해 최적 전략을 선택합니다:

1. **규모** — Small (< 50p) / Medium (50-500p) / Large (500p+)
2. **문서 유형** — 내부문서, 법률, PDF/이미지, 코드
3. **LLM 선택** — Gemini Flash (기본) / Claude CLI / OpenAI / Claude API
4. **검증 수준** — Basic / CRAG / Self-RAG / Agentic

## 핵심 스택

| 레이어 | 컴포넌트 | 스펙 |
|--------|----------|------|
| 임베딩 (일반) | **Voyage `voyage-4-large`** | MoE, Matryoshka (256~2048dim), 32K 컨텍스트, 다국어 |
| 임베딩 (장문) | **Voyage `voyage-context-3`** | 500p+ 문서, Cross-chunk context 보존, Late Chunking 네이티브 지원 |
| 리랭킹 | **Voyage `rerank-2`** | Cross-encoder |
| 벡터 DB | **pgvector + pgvectorscale** | 50M 벡터에서 471 QPS |
| 키워드 | **PostgreSQL tsvector** | 같은 DB, 추가 인프라 없음 |
| LLM | **Any (Claude / Gemini / OpenAI)** | 생성 + 맥락 강화, 원하는 LLM 선택 가능 |
| 평가 | **RAGAS** | Faithfulness, Precision, Recall |
| 모니터링 | **Langfuse** (오픈소스) | 프로덕션 관측성 |

## 일반 RAG vs 이 스킬의 차이

일반 RAG는 `검색 → 답변`으로 끝납니다. 이 스킬은 **4단계 검증 파이프라인**으로 검색 품질을 근본적으로 끌어올립니다.

```
일반 RAG:  질문 → 임베딩 검색 → LLM 답변  (끝)
이 스킬:   질문 → ① 하이브리드 검색 → ② 리랭킹 → ③ CRAG 검증 → LLM 답변
                    + ④ 맥락 강화 (인제스션 시)
```

### 누가 뭘 하는가?

| 단계 | 담당 | 모델/도구 | 역할 |
|------|------|-----------|------|
| 임베딩 | **Voyage** | `voyage-4-large` | 문서/쿼리를 벡터로 변환 |
| 시맨틱 검색 | **pgvector** | PostgreSQL 확장 | 벡터 유사도 검색 |
| 키워드 검색 | **PostgreSQL** | `tsvector` + `ts_rank` | BM25 스타일 텍스트 검색 |
| RRF 융합 | **PostgreSQL** | SQL 쿼리 | 시맨틱 + 키워드 결과 합산 |
| 리랭킹 | **Voyage** | `rerank-2` | 검색 결과 정밀 재정렬 |
| CRAG 검증 | **LLM** | Claude / Gemini / OpenAI | 검색 결과 품질 판단 |
| 맥락 강화 | **LLM** | Claude / Gemini / OpenAI | 청크에 문맥 설명 부착 |
| 답변 생성 | **LLM** | Claude / Gemini / OpenAI | 최종 답변 생성 |

> **정리**: Voyage = 임베딩 + 리랭킹 (검색 품질), PostgreSQL = 저장 + 검색, LLM = 판단 + 생성.
> LLM은 교체 가능하지만 **Voyage와 하이브리드 검색은 이 파이프라인의 핵심**입니다.

### 왜 이 4단계가 중요한가?

**① 하이브리드 검색 — "못 찾는 문서가 없다"** `Voyage + PostgreSQL`

임베딩(시맨틱) 검색만으로는 정확한 용어나 고유명사를 놓칩니다. 키워드(BM25) 검색만으로는 의미를 못 잡습니다. 둘을 RRF로 합치면 **Recall 0.72 → 0.91** (+26%). 이게 가장 큰 성능 차이를 만듭니다.

**② 리랭킹 — "찾은 것 중 진짜 관련된 것만"** `Voyage rerank-2`

1차 검색이 20개를 가져오면, Voyage rerank-2가 Cross-encoder로 정밀하게 재정렬합니다. 비용 대비 정밀도 향상이 가장 큰 단계입니다.

**③ CRAG 검증 — "쓰레기가 들어가면 쓰레기가 나온다"** `LLM (아무거나)`

검색 결과가 진짜 쓸만한지 LLM이 자동 판단합니다 (CORRECT / AMBIGUOUS / INCORRECT). **독소 문서 5개만으로 RAG 응답의 90%를 조작할 수 있다**는 연구 결과가 있습니다. CRAG 다층 방어로 공격 성공률 73% → **8.7%**로 감소.

**④ 맥락 강화 — "사서가 책에 메모를 붙여둔다"** `LLM (아무거나)`

인제스션 단계에서 LLM이 각 청크에 "이 내용은 XX 문서의 YY 섹션에 속합니다"라는 설명을 부착합니다. Anthropic 연구 결과, 검색 실패율 **67% 감소**.

> **핵심 인사이트**: Voyage(임베딩/리랭킹)와 PostgreSQL(하이브리드 검색)이 **정확한 문서를 찾는** 핵심이고, LLM은 마지막에 "읽고 답변하는" 역할이라 아무거나 써도 됩니다. 그래서 이 스킬은 검색 파이프라인에 집중하고, LLM은 Claude / Gemini / OpenAI 중 선택 가능합니다.

## 파이프라인 구조

### 인제스션 (문서 → 저장)

```
문서 → 시맨틱 청킹 (512토큰, 오버랩)
     → 맥락 강화 (Anthropic Contextual Retrieval: -67% 검색 실패)
     → 검색용 요약 생성 ("요약으로 검색, 원문으로 전달")
     → Voyage 4 임베딩 (Matryoshka 1024dim)
     → pgvector + tsvector 하이브리드 저장
```

### 쿼리 (질문 → 답변)

```
쿼리 → Voyage 4 임베딩
     → 하이브리드 검색 (시맨틱 + 키워드, RRF 융합)
     → Voyage rerank-2 리랭킹
     → CRAG 검증 (CORRECT/AMBIGUOUS/INCORRECT)
     → 원문 기반 Claude 답변 생성
```

## 지원하는 RAG 아키텍처 (10종)

| 타입 | 핵심 | 사용 시점 |
|------|------|-----------|
| Naive RAG | chunk → embed → search → generate | 프로토타입 |
| **Advanced RAG** | + 하이브리드 검색 + 리랭킹 | 일반 프로덕션 |
| Graph RAG | 지식 그래프 + 커뮤니티 요약 | 멀티홉 추론 |
| **Agentic RAG** | 에이전트가 검색 전략 결정 | 복잡한 쿼리 |
| **Corrective RAG** | 검색 품질 검증 + 웹 폴백 | 높은 신뢰도 |
| Self-RAG | 모델이 검색 필요 여부 판단 | 효율 최적화 |
| Multimodal RAG | ColPali (이미지/표/차트) | PDF, 인포그래픽 |

## 핵심 원칙

1. **하이브리드 검색은 필수** — BM25만: Recall 0.72 → 하이브리드: **0.91**
2. **리랭킹은 싼 보험** — Voyage rerank-2로 정밀도 대폭 향상
3. **요약으로 검색, 원문으로 전달** — 도서관 카탈로그 패턴
4. **CRAG 검증으로 독소 방어** — 독소 문서 5개로 응답 90% 조작 가능
5. **평가 없이는 도박** — RAGAS 골든 데이터셋 필수
6. **맥락 강화: -67% 실패** — Anthropic 검증 방법
7. **Matryoshka로 비용 절감** — 512dim ≈ 97% 품질

## 주요 연구 결과

| 기법 | 효과 |
|------|------|
| Contextual Retrieval | 검색 실패율 **67% 감소** |
| Hybrid Search (RRF) | Recall 0.72 → **0.91** |
| CRAG 다층 방어 | 공격 성공률 73% → **8.7%** |
| Matryoshka 512dim | 97% 성능 유지, 절반 크기 |
| int8 양자화 | 4x 메모리 절감, 96% 성능 |
| Graph RAG | 포괄성 **72-83% 향상** |

## 사용자가 할 일 vs Claude Code가 할 일

| 작업 | Claude Code 자동 | 사용자 |
|------|:----------------:|:------:|
| 프로젝트 구조 생성 | ✅ | |
| Docker (PostgreSQL + pgvector) 세팅 + 실행 | ✅ | |
| DB 스키마 생성 | ✅ | |
| 파이프라인 코드 전체 생성 | ✅ | |
| `.env` 파일 생성 + API 키 설정 | ✅ | |
| 인제스션 스크립트 생성 + 실행 | ✅ | |
| 챗봇 API 서버 생성 | ✅ | |
| **PDF 파일 준비** | | ✅ |
| **Voyage API 키 발급** ([무료 가입](https://dash.voyageai.com/)) | | ✅ |

> **사용자는 PDF 파일 + Voyage API 키만 준비하면 됩니다.** 나머지는 전부 Claude Code가 자동으로 생성하고 실행합니다.

## API 키

| 키 | 발급처 | 비용 |
|----|--------|------|
| `VOYAGE_API_KEY` | [dash.voyageai.com](https://dash.voyageai.com/) | 무료 50M 토큰/월 |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/) | 사용량 기반 |

- API 키가 없어도 로컬 모델(bge-m3)로 폴백 가능합니다.
- **Claude CLI가 설치되어 있으면 `ANTHROPIC_API_KEY` 없이 CLI 모드로 사용 가능합니다.**

## LLM 모드 선택 (Multi-LLM)

`RAG_LLM_MODE` 환경변수로 LLM 제공자를 선택합니다:

```bash
export RAG_LLM_MODE=gemini       # Gemini Flash (기본값, 가장 저렴)
export RAG_LLM_MODE=claude-cli   # Claude CLI (API 키 불필요)
export RAG_LLM_MODE=openai       # OpenAI
export RAG_LLM_MODE=claude-api   # Claude API (Agentic 필요 시)
```

> **왜 Gemini Flash가 기본값인가?** RAG에서 LLM은 맥락 강화(한 줄 요약), CRAG 검증(분류), 답변 생성(읽고 요약) 같은 단순 작업만 합니다. 이런 작업에서 모델 간 품질 차이는 거의 없고, 비용 차이만 큽니다.

| 항목 | **Gemini Flash (기본)** | Claude CLI | OpenAI | Claude API |
|------|------------------------|-----------|--------|-----------|
| 비용 | **가장 저렴** | 플랜 포함 | 사용량 과금 | 사용량 과금 |
| 속도 | **가장 빠름** | 느림 | 빠름 | 빠름 |
| tool_use | 지원 | 미지원 | 지원 | 지원 |
| Agentic RAG | 가능 | 불가 | 가능 | 가능 |
| 추천 | **일반 용도** | API 키 없을 때 | 범용 | Agentic 필요 시 |

## 포함된 코드 템플릿

스킬 파일에 바로 사용 가능한 Python 코드가 포함되어 있습니다:

- `exceptions.py` — 커스텀 예외 계층 (RagError → LLMError → RateLimitError 등)
- `config.py` — pydantic BaseSettings 중앙 설정 (env var 통합 관리)
- `llm.py` — LLM 추상화 레이어 (API/CLI 자동 전환, tenacity 재시도)
- `chunking.py` — 시맨틱 청킹 + Late Chunking (voyage-context-3)
- `enrichment.py` — 맥락 강화 (Anthropic 방식)
- `embedding.py` — Voyage 4 임베딩 (배치, Matryoshka, Rate-limit 재시도)
- `storage.py` — pgvector 하이브리드 저장/검색 (RRF SQL, psycopg3 ConnectionPool)
- `reranker.py` — Voyage rerank-2
- `crag.py` — CRAG 자가 수정 (JSON 구조화 응답 + Verdict enum)
- `pipeline.py` — 풀 쿼리 파이프라인 (QueryRequest 입력 검증 + 인젝션 방어)
- `ingest.py` — 인제스션 스크립트 3종 (Small/Medium/Large 규모별)
- `agentic_rag.py` — 에이전틱 RAG (Claude tool_use)
- `evaluation.py` — RAGAS 평가
- `monitoring.py` — Langfuse 모니터링
- `schema.sql` — pgvector 테이블 + 인덱스

## 참고 자료

- [Voyage 4 Model Family (MoE)](https://blog.voyageai.com/2026/01/15/voyage-4/)
- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [CRAG Paper](https://arxiv.org/abs/2401.15884)
- [Agentic RAG Survey](https://arxiv.org/abs/2501.09136)
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
- [Late Chunking](https://arxiv.org/pdf/2409.04701)
- [Stanford "Lost in the Middle"](https://arxiv.org/abs/2307.03172)
- [RAGAS Evaluation](https://docs.ragas.io/en/latest/concepts/metrics/)
- [NirDiamant RAG Techniques (30+)](https://github.com/NirDiamant/RAG_Techniques)
- [ColPali Multimodal RAG](https://huggingface.co/blog/manu/colpali)
- [RAG Security Threat Model](https://arxiv.org/pdf/2509.20324)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [pgvectorscale Benchmarks](https://www.tigerdata.com/blog/pgvector-vs-qdrant)

## License

MIT
