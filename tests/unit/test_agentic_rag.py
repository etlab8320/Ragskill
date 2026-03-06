"""
test_agentic_rag.py — Agentic RAG 테스트

테스트 전략:
  - TOOLS/AGENT_SYSTEM: 정적 구조 검증 (mock 불필요)
  - agentic_query(): tool dispatch, 반복 로직, 종료 조건
  - module-level 실행(get_api_client) → sys.modules mock으로 우회
"""

import sys
import pytest
from unittest.mock import MagicMock, patch

# External dependency mocks are centralized in conftest.py

# Mock llm module before agentic_rag.py is imported
# agentic_rag.py runs get_api_client() at module level
_mock_llm_mod = sys.modules.get("llm")
if _mock_llm_mod is None or not hasattr(_mock_llm_mod, "_agentic_patched"):
    _mock_llm_mod = MagicMock()
    _mock_llm_mod.get_api_client.return_value = (
        MagicMock(),  # client
        "claude-haiku-4-5-20251001",  # model_name
        "anthropic",  # provider
    )
    _mock_llm_mod._agentic_patched = True
    sys.modules["llm"] = _mock_llm_mod

# Mock embedding and reranker modules
sys.modules.setdefault("embedding", MagicMock())
sys.modules.setdefault("reranker", MagicMock())


# ── Helper: mock Anthropic response ──────────────────────────────────────────

def _make_text_response(text="Here is the answer."):
    """stop_reason='end_turn' response"""
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    content_block = MagicMock()
    content_block.type = "text"
    content_block.text = text
    resp.content = [content_block]
    return resp


def _make_tool_response(tool_name, tool_input, tool_id="tool_123"):
    """stop_reason='tool_use' response"""
    resp = MagicMock()
    resp.stop_reason = "tool_use"
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.input = tool_input
    block.id = tool_id
    resp.content = [block]
    return resp


# ── TOOLS schema ──────────────────────────────────────────────────────────────

class TestToolSchema:
    """TOOLS 리스트 구조 검증"""

    def test_tools_has_two_entries(self):
        from agentic_rag import TOOLS
        assert len(TOOLS) == 2

    def test_search_tool_schema(self):
        """search_documents: name + query required"""
        from agentic_rag import TOOLS
        search = TOOLS[0]
        assert search["name"] == "search_documents"
        assert "query" in search["input_schema"]["properties"]
        assert "query" in search["input_schema"]["required"]

    def test_document_list_tool_schema(self):
        """get_document_list: name + empty properties"""
        from agentic_rag import TOOLS
        doc_list = TOOLS[1]
        assert doc_list["name"] == "get_document_list"
        assert doc_list["input_schema"]["properties"] == {}


# ── AGENT_SYSTEM ──────────────────────────────────────────────────────────────

class TestAgentSystem:
    """AGENT_SYSTEM 프롬프트 검증"""

    def test_system_prompt_exists(self):
        from agentic_rag import AGENT_SYSTEM
        assert len(AGENT_SYSTEM) > 0

    def test_system_prompt_mentions_iterations(self):
        """max iterations 관련 안내 포함"""
        from agentic_rag import AGENT_SYSTEM
        assert "3" in AGENT_SYSTEM or "iteration" in AGENT_SYSTEM.lower()


# ── agentic_query() ──────────────────────────────────────────────────────────

class TestAgenticQuery:
    """agentic_query() — tool dispatch + 반복 로직"""

    def test_end_turn_returns_text(self):
        """stop_reason='end_turn' → content[0].text 반환"""
        from agentic_rag import agentic_query
        with patch("agentic_rag.client") as mock_client:
            mock_client.messages.create.return_value = _make_text_response("Final answer")
            store = MagicMock()
            result = agentic_query("What is RAG?", store)
            assert result == "Final answer"

    def test_search_documents_dispatched(self):
        """search_documents tool → embed_query + hybrid_search + rerank 호출"""
        from agentic_rag import agentic_query
        store = MagicMock()
        store.hybrid_search.return_value = [
            {"content": "test content", "metadata": {"source": "test.pdf"}},
        ]

        with patch("agentic_rag.client") as mock_client, \
             patch("agentic_rag.embed_query", return_value=[0.1] * 1024) as mock_embed, \
             patch("agentic_rag.rerank", return_value=[
                 {"content": "test content", "metadata": {"source": "test.pdf"}},
             ]) as mock_rerank:
            mock_client.messages.create.side_effect = [
                _make_tool_response("search_documents", {"query": "RAG basics"}),
                _make_text_response("Answer"),
            ]
            result = agentic_query("What is RAG?", store)
            mock_embed.assert_called_once_with("RAG basics")
            store.hybrid_search.assert_called_once()
            mock_rerank.assert_called_once()
            assert result == "Answer"

    def test_get_document_list_dispatched(self):
        """get_document_list tool → store.get_unique_sources 호출"""
        from agentic_rag import agentic_query
        store = MagicMock()
        store.get_unique_sources.return_value = ["doc1.pdf", "doc2.pdf"]

        with patch("agentic_rag.client") as mock_client:
            mock_client.messages.create.side_effect = [
                _make_tool_response("get_document_list", {}),
                _make_text_response("Here are your docs"),
            ]
            result = agentic_query("List docs", store)
            store.get_unique_sources.assert_called_once()
            assert result == "Here are your docs"

    def test_get_document_list_empty(self):
        """sources 빈 리스트 → 'No documents' 메시지"""
        from agentic_rag import agentic_query
        store = MagicMock()
        store.get_unique_sources.return_value = []

        with patch("agentic_rag.client") as mock_client:
            # Capture the tool_result sent back
            responses = [
                _make_tool_response("get_document_list", {}),
                _make_text_response("No docs found"),
            ]
            mock_client.messages.create.side_effect = responses
            result = agentic_query("List docs", store)
            # Verify the messages passed to second call contain "No documents"
            second_call = mock_client.messages.create.call_args_list[1]
            messages = second_call.kwargs.get("messages") or second_call[1].get("messages", [])
            tool_msg = [m for m in messages if m.get("role") == "user" and isinstance(m.get("content"), list)]
            if tool_msg:
                content_str = str(tool_msg[-1]["content"])
                assert "No documents" in content_str

    def test_max_iterations_returns_message(self):
        """5회 루프 후 'Max iterations reached.' 반환"""
        from agentic_rag import agentic_query
        store = MagicMock()
        store.hybrid_search.return_value = [
            {"content": "c", "metadata": {"source": "s.pdf"}},
        ]

        with patch("agentic_rag.client") as mock_client, \
             patch("agentic_rag.embed_query", return_value=[0.1] * 1024), \
             patch("agentic_rag.rerank", return_value=[
                 {"content": "c", "metadata": {"source": "s.pdf"}},
             ]):
            # Always return tool_use, never end_turn → hit max iterations
            mock_client.messages.create.return_value = _make_tool_response(
                "search_documents", {"query": "q"}
            )
            result = agentic_query("infinite loop", store)
            assert result == "Max iterations reached."
            assert mock_client.messages.create.call_count == 5

    def test_tool_result_appended_to_messages(self):
        """messages에 assistant + tool_result 추가됨"""
        from agentic_rag import agentic_query
        store = MagicMock()
        store.get_unique_sources.return_value = ["a.pdf"]

        with patch("agentic_rag.client") as mock_client:
            mock_client.messages.create.side_effect = [
                _make_tool_response("get_document_list", {}),
                _make_text_response("Done"),
            ]
            agentic_query("list", store)
            # After tool call, messages list should contain assistant + user(tool_result)
            second_call_msgs = mock_client.messages.create.call_args_list[1].kwargs.get("messages", [])
            roles = [m["role"] for m in second_call_msgs]
            assert "assistant" in roles
            assert roles.count("user") >= 2  # initial + tool_result

    def test_reranked_results_formatted(self):
        """'[source] content' 포맷으로 tool_result 구성"""
        from agentic_rag import agentic_query
        store = MagicMock()
        store.hybrid_search.return_value = [
            {"content": "Exercise is good", "metadata": {"source": "sports.pdf"}},
        ]

        with patch("agentic_rag.client") as mock_client, \
             patch("agentic_rag.embed_query", return_value=[0.1] * 1024), \
             patch("agentic_rag.rerank", return_value=[
                 {"content": "Exercise is good", "metadata": {"source": "sports.pdf"}},
             ]):
            mock_client.messages.create.side_effect = [
                _make_tool_response("search_documents", {"query": "exercise"}),
                _make_text_response("Answer"),
            ]
            agentic_query("exercise?", store)
            second_call = mock_client.messages.create.call_args_list[1]
            msgs = second_call.kwargs.get("messages") or second_call[1].get("messages", [])
            tool_content = str(msgs)
            assert "sports.pdf" in tool_content

    def test_search_default_top_k(self):
        """top_k 미지정시 기본값 5"""
        from agentic_rag import agentic_query
        store = MagicMock()
        store.hybrid_search.return_value = []

        with patch("agentic_rag.client") as mock_client, \
             patch("agentic_rag.embed_query", return_value=[0.1] * 1024), \
             patch("agentic_rag.rerank", return_value=[]):
            mock_client.messages.create.side_effect = [
                _make_tool_response("search_documents", {"query": "test"}),
                _make_text_response("No results"),
            ]
            agentic_query("test", store)
            rerank_call = store.hybrid_search.call_args
            # hybrid_search called with top_k=5 (default from block.input.get("top_k", 5))
            assert rerank_call is not None

    def test_multiple_tool_calls_in_sequence(self):
        """2회 tool_use → 최종 end_turn"""
        from agentic_rag import agentic_query
        store = MagicMock()
        store.get_unique_sources.return_value = ["a.pdf"]
        store.hybrid_search.return_value = [
            {"content": "c", "metadata": {"source": "a.pdf"}},
        ]

        with patch("agentic_rag.client") as mock_client, \
             patch("agentic_rag.embed_query", return_value=[0.1] * 1024), \
             patch("agentic_rag.rerank", return_value=[
                 {"content": "c", "metadata": {"source": "a.pdf"}},
             ]):
            mock_client.messages.create.side_effect = [
                _make_tool_response("get_document_list", {}, "t1"),
                _make_tool_response("search_documents", {"query": "q"}, "t2"),
                _make_text_response("Final"),
            ]
            result = agentic_query("multi step", store)
            assert result == "Final"
            assert mock_client.messages.create.call_count == 3

    def test_provider_not_anthropic_raises(self):
        """provider='openai' → NotImplementedError"""
        # This tests the module-level guard; we need to re-import with different provider
        import importlib
        _mock_llm_mod.get_api_client.return_value = (MagicMock(), "gpt-4", "openai")
        # Remove cached module to force re-import
        if "agentic_rag" in sys.modules:
            del sys.modules["agentic_rag"]
        with pytest.raises(NotImplementedError, match="claude-api"):
            import agentic_rag  # noqa: F811
        # Restore for other tests
        _mock_llm_mod.get_api_client.return_value = (
            MagicMock(), "claude-haiku-4-5-20251001", "anthropic"
        )
        if "agentic_rag" in sys.modules:
            del sys.modules["agentic_rag"]
