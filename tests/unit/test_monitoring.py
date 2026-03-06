"""
test_monitoring.py — Langfuse monitoring 모듈 테스트

테스트 전략:
  - traced_query(): Langfuse trace/span lifecycle 검증
  - 외부 의존성(langfuse, embedding, reranker, llm) 모두 mock
"""

import sys
import pytest
from unittest.mock import MagicMock, patch, call

# External dependency mocks are centralized in conftest.py


@pytest.fixture
def mock_langfuse_env():
    """Langfuse + embedding + reranker + llm + storage 전체 mock"""
    trace = MagicMock()
    search_span = MagicMock()
    rerank_span = MagicMock()
    gen_span = MagicMock()
    trace.span.side_effect = [search_span, rerank_span, gen_span]

    mock_lf = MagicMock()
    mock_lf.trace.return_value = trace

    mock_store = MagicMock()
    mock_store.hybrid_search.return_value = [
        {"content": "Exercise improves health." * 50, "metadata": {"source": "sports.pdf"}},
        {"content": "Mental toughness matters.", "metadata": {"source": "mental.pdf"}},
    ]

    with patch("monitoring.langfuse", mock_lf), \
         patch("monitoring.embed_query", return_value=[0.1] * 1024), \
         patch("monitoring.rerank", return_value=[
             {"content": "Exercise improves health." * 50, "metadata": {"source": "sports.pdf"}},
             {"content": "Mental toughness matters.", "metadata": {"source": "mental.pdf"}},
         ]), \
         patch("monitoring.llm", return_value="Exercise improves cardiovascular health."):
        yield {
            "langfuse": mock_lf,
            "trace": trace,
            "search_span": search_span,
            "rerank_span": rerank_span,
            "gen_span": gen_span,
            "store": mock_store,
        }


class TestTracedQuery:
    """traced_query() — Langfuse tracing 검증"""

    def test_creates_trace(self, mock_langfuse_env):
        """langfuse.trace(name='rag-query') 호출"""
        from monitoring import traced_query
        traced_query("What is VO2max?", mock_langfuse_env["store"])
        mock_langfuse_env["langfuse"].trace.assert_called_once_with(
            name="rag-query", input="What is VO2max?"
        )

    def test_trace_receives_input(self, mock_langfuse_env):
        """trace input = user_query"""
        from monitoring import traced_query
        traced_query("test query", mock_langfuse_env["store"])
        call_kwargs = mock_langfuse_env["langfuse"].trace.call_args
        assert call_kwargs.kwargs["input"] == "test query"

    def test_three_spans_created(self, mock_langfuse_env):
        """hybrid-search, rerank, generation 3개 span 생성"""
        from monitoring import traced_query
        traced_query("query", mock_langfuse_env["store"])
        trace = mock_langfuse_env["trace"]
        assert trace.span.call_count == 3
        span_names = [c.kwargs["name"] for c in trace.span.call_args_list]
        assert span_names == ["hybrid-search", "rerank", "generation"]

    def test_search_span_output(self, mock_langfuse_env):
        """retrieval span end에 count 포함"""
        from monitoring import traced_query
        traced_query("query", mock_langfuse_env["store"])
        search_span = mock_langfuse_env["search_span"]
        search_span.end.assert_called_once()
        end_kwargs = search_span.end.call_args.kwargs
        assert "count" in end_kwargs.get("output", {})

    def test_rerank_span_output(self, mock_langfuse_env):
        """rerank span end에 count 포함"""
        from monitoring import traced_query
        traced_query("query", mock_langfuse_env["store"])
        rerank_span = mock_langfuse_env["rerank_span"]
        rerank_span.end.assert_called_once()
        end_kwargs = rerank_span.end.call_args.kwargs
        assert "count" in end_kwargs.get("output", {})

    def test_gen_span_output(self, mock_langfuse_env):
        """generation span end에 answer_length 포함"""
        from monitoring import traced_query
        traced_query("query", mock_langfuse_env["store"])
        gen_span = mock_langfuse_env["gen_span"]
        gen_span.end.assert_called_once()
        end_kwargs = gen_span.end.call_args.kwargs
        assert "answer_length" in end_kwargs.get("output", {})

    def test_returns_answer_and_sources(self, mock_langfuse_env):
        """반환값: {'answer': str, 'sources': list}"""
        from monitoring import traced_query
        result = traced_query("query", mock_langfuse_env["store"])
        assert "answer" in result
        assert "sources" in result
        assert isinstance(result["sources"], list)

    def test_trace_update_with_answer(self, mock_langfuse_env):
        """trace.update(output=answer) 호출"""
        from monitoring import traced_query
        traced_query("query", mock_langfuse_env["store"])
        trace = mock_langfuse_env["trace"]
        trace.update.assert_called_once()
        update_kwargs = trace.update.call_args.kwargs
        assert "output" in update_kwargs

    def test_sources_from_metadata(self, mock_langfuse_env):
        """sources = reranked 결과의 metadata 리스트"""
        from monitoring import traced_query
        result = traced_query("query", mock_langfuse_env["store"])
        assert len(result["sources"]) == 2
        assert result["sources"][0] == {"source": "sports.pdf"}

    def test_context_truncated_to_500(self, mock_langfuse_env):
        """각 청크 content[:500] 적용되어 LLM에 전달"""
        from monitoring import traced_query
        with patch("monitoring.llm") as mock_llm:
            mock_llm.return_value = "answer"
            traced_query("query", mock_langfuse_env["store"])
            prompt = mock_llm.call_args[0][0]
            # Long content (50 * "Exercise improves health.") should be truncated
            # Each chunk segment in context should be <= 500 chars
            segments = prompt.split("---")
            for seg in segments:
                # Context segments are trimmed by content[:500]
                assert len(seg.strip()) <= 600  # Some overhead from formatting
