# RAG Pipeline Skill for Claude Code

> 30+ 기법/논문/벤치마크 리서치 기반의 프로덕션 RAG 파이프라인 구축 스킬

Claude Code에서 `/rag-pipeline` 명령으로 사용합니다.
"RAG 만들어줘" 같은 요청을 하면 규모 판단 → 전략 선택 → 코드 생성까지 자동으로 가이드합니다.

## 설치

```bash
curl -fsSL https://raw.githubusercontent.com/etlab8320/Ragskill/main/install.sh | bash
```

또는 수동 설치:

```bash
mkdir -p ~/.claude/skills/rag-pipeline
curl -fsSL https://raw.githubusercontent.com/etlab8320/Ragskill/main/skill/SKILL.md \
  -o ~/.claude/skills/rag-pipeline/SKILL.md
```

## 사용법

Claude Code에서:

```
/rag-pipeline
```

자동으로 3단계 질문을 통해 최적 전략을 선택합니다:

1. **규모** — Small (< 50p) / Medium (50-500p) / Large (500p+)
2. **문서 유형** — 내부문서, 법률, PDF/이미지, 코드
3. **검증 수준** — Basic / CRAG / Self-RAG / Agentic

## 핵심 스택

| 레이어 | 컴포넌트 | 스펙 |
|--------|----------|------|
| 임베딩 | **Voyage `voyage-4-large`** | MoE, Matryoshka (256~2048dim), 32K 컨텍스트, 다국어 |
| 리랭킹 | **Voyage `rerank-2`** | Cross-encoder |
| 벡터 DB | **pgvector + pgvectorscale** | 50M 벡터에서 471 QPS |
| 키워드 | **PostgreSQL tsvector** | 같은 DB, 추가 인프라 없음 |
| LLM | **Claude API** | 생성 + 맥락 강화 |
| 평가 | **RAGAS** | Faithfulness, Precision, Recall |
| 모니터링 | **Langfuse** (오픈소스) | 프로덕션 관측성 |

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

## API 키

| 키 | 발급처 | 비용 |
|----|--------|------|
| `VOYAGE_API_KEY` | [dash.voyageai.com](https://dash.voyageai.com/) | 무료 50M 토큰/월 |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/) | 사용량 기반 |

API 키가 없어도 로컬 모델(bge-m3)로 폴백 가능합니다.

## 포함된 코드 템플릿

스킬 파일에 바로 사용 가능한 Python 코드가 포함되어 있습니다:

- `chunking.py` — 시맨틱 청킹 (오버랩)
- `enrichment.py` — 맥락 강화 (Anthropic 방식)
- `embedding.py` — Voyage 4 임베딩 (배치, Matryoshka)
- `storage.py` — pgvector 하이브리드 저장/검색 (RRF SQL)
- `reranker.py` — Voyage rerank-2
- `crag.py` — CRAG 자가 수정
- `pipeline.py` — 풀 쿼리 파이프라인
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
