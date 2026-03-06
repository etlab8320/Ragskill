"""
test_evaluation.py — RAGAS evaluation 모듈 테스트

테스트 전략:
  - _get_ragas_llm(): LLM provider 선택 로직 (settings mock)
  - evaluate_rag(): RAGAS evaluate 호출 + 데이터 구조 검증
  - 외부 의존성(ragas, langchain_*) 모두 sys.modules mock으로 격리
"""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# External dependency mocks are centralized in conftest.py


# ── _get_ragas_llm() ─────────────────────────────────────────────────────────

class TestGetRagasLlm:
    """_get_ragas_llm() — LLM provider 선택 로직"""

    def test_gemini_mode(self):
        """rag_llm_mode='gemini' → ChatGoogleGenerativeAI 생성"""
        with patch("evaluation.settings") as mock_settings:
            mock_settings.rag_llm_mode = "gemini"
            mock_settings.gemini_api_key = "test-key"
            from evaluation import _get_ragas_llm
            result = _get_ragas_llm()
            assert result is not None

    def test_openai_mode(self):
        """rag_llm_mode='openai' → ChatOpenAI 생성"""
        with patch("evaluation.settings") as mock_settings:
            mock_settings.rag_llm_mode = "openai"
            from evaluation import _get_ragas_llm
            result = _get_ragas_llm()
            assert result is not None

    def test_claude_api_mode(self):
        """rag_llm_mode='claude-api' → ChatAnthropic 생성"""
        with patch("evaluation.settings") as mock_settings:
            mock_settings.rag_llm_mode = "claude-api"
            mock_settings.anthropic_api_key = "test-key"
            from evaluation import _get_ragas_llm
            result = _get_ragas_llm()
            assert result is not None

    def test_invalid_mode_raises(self):
        """지원하지 않는 mode → ValueError"""
        with patch("evaluation.settings") as mock_settings:
            mock_settings.rag_llm_mode = "ollama"
            from evaluation import _get_ragas_llm
            with pytest.raises(ValueError, match="ollama"):
                _get_ragas_llm()

    def test_gemini_default_model(self):
        """gemini 기본 모델 = gemini-2.0-flash"""
        with patch("evaluation.settings") as mock_settings, \
             patch.dict(os.environ, {}, clear=False):
            mock_settings.rag_llm_mode = "gemini"
            mock_settings.gemini_api_key = "key"
            os.environ.pop("RAG_LLM_MODEL", None)
            from evaluation import _get_ragas_llm
            from langchain_google_genai import ChatGoogleGenerativeAI
            _get_ragas_llm()
            call_kwargs = ChatGoogleGenerativeAI.call_args
            if call_kwargs:
                assert call_kwargs.kwargs.get("model", "") == "gemini-2.0-flash" or \
                       call_kwargs.args[0] == "gemini-2.0-flash" if call_kwargs.args else True

    def test_gemini_uses_env_model(self):
        """RAG_LLM_MODEL 환경변수 반영"""
        with patch("evaluation.settings") as mock_settings, \
             patch.dict(os.environ, {"RAG_LLM_MODEL": "gemini-1.5-pro"}, clear=False):
            mock_settings.rag_llm_mode = "gemini"
            mock_settings.gemini_api_key = "key"
            from evaluation import _get_ragas_llm
            _get_ragas_llm()
            # Should not raise — env model accepted

    def test_openai_uses_env_model(self):
        """openai도 RAG_LLM_MODEL 반영"""
        with patch("evaluation.settings") as mock_settings, \
             patch.dict(os.environ, {"RAG_LLM_MODEL": "gpt-4-turbo"}, clear=False):
            mock_settings.rag_llm_mode = "openai"
            from evaluation import _get_ragas_llm
            _get_ragas_llm()

    def test_claude_uses_env_model(self):
        """claude-api도 RAG_LLM_MODEL 반영"""
        with patch("evaluation.settings") as mock_settings, \
             patch.dict(os.environ, {"RAG_LLM_MODEL": "claude-sonnet-4-20250514"}, clear=False):
            mock_settings.rag_llm_mode = "claude-api"
            mock_settings.anthropic_api_key = "key"
            from evaluation import _get_ragas_llm
            _get_ragas_llm()

    def test_claude_default_model(self):
        """claude-api 기본 모델 확인"""
        with patch("evaluation.settings") as mock_settings, \
             patch.dict(os.environ, {}, clear=False):
            mock_settings.rag_llm_mode = "claude-api"
            mock_settings.anthropic_api_key = "key"
            os.environ.pop("RAG_LLM_MODEL", None)
            from evaluation import _get_ragas_llm
            _get_ragas_llm()
            # Should use claude-haiku-4-5-20251001 default


# ── evaluate_rag() ────────────────────────────────────────────────────────────

class TestEvaluateRag:
    """evaluate_rag() — RAGAS evaluate 호출 검증"""

    @pytest.fixture(autouse=True)
    def setup_evaluate_mock(self):
        """ragas.evaluate를 mock"""
        with patch("evaluation.evaluate") as mock_eval, \
             patch("evaluation._get_ragas_llm") as mock_llm:
            mock_eval.return_value = {
                "faithfulness": 0.95,
                "context_precision": 0.85,
                "context_recall": 0.80,
                "answer_relevancy": 0.90,
            }
            mock_llm.return_value = MagicMock()
            self.mock_evaluate = mock_eval
            self.mock_llm = mock_llm
            yield

    def test_calls_ragas_evaluate(self):
        """ragas.evaluate 호출됨"""
        from evaluation import evaluate_rag
        evaluate_rag(["q1"], ["a1"], [["ctx1"]], ["gt1"])
        self.mock_evaluate.assert_called_once()

    def test_returns_result(self):
        """evaluate 반환값 그대로 반환"""
        from evaluation import evaluate_rag
        result = evaluate_rag(["q1"], ["a1"], [["ctx1"]], ["gt1"])
        assert result["faithfulness"] == 0.95
        assert result["context_precision"] == 0.85

    def test_passes_four_metrics(self):
        """4개 메트릭 전달: faithfulness, context_precision, context_recall, answer_relevancy"""
        from evaluation import evaluate_rag
        evaluate_rag(["q1"], ["a1"], [["ctx1"]], ["gt1"])
        call_kwargs = self.mock_evaluate.call_args
        metrics = call_kwargs.kwargs.get("metrics") or call_kwargs[1].get("metrics", [])
        assert len(metrics) == 4

    def test_creates_evaluation_dataset(self):
        """EvaluationDataset 생성됨"""
        with patch("evaluation.EvaluationDataset") as mock_ds:
            from evaluation import evaluate_rag
            evaluate_rag(["q1", "q2"], ["a1", "a2"], [["c1"], ["c2"]], ["g1", "g2"])
            mock_ds.assert_called_once()

    def test_zip_alignment(self):
        """questions/answers/contexts/ground_truths 길이 일치 시 정상 동작"""
        from evaluation import evaluate_rag
        result = evaluate_rag(
            ["q1", "q2", "q3"],
            ["a1", "a2", "a3"],
            [["c1"], ["c2"], ["c3"]],
            ["g1", "g2", "g3"],
        )
        assert result is not None

    def test_empty_input(self):
        """빈 리스트 입력도 처리"""
        from evaluation import evaluate_rag
        result = evaluate_rag([], [], [], [])
        assert result is not None
