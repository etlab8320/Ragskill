"""
test_crag.py — CRAG 파싱 로직 + 검색 평가 테스트

핵심: _parse_verdict()를 직접 테스트 (llm mock 불필요)
      JSON-first parsing + string-match fallback 전부 커버
"""

import pytest
from unittest.mock import patch, MagicMock
from crag import Verdict, CRAGResult, _parse_verdict, evaluate_retrieval


class TestParseVerdict:
    """_parse_verdict() 직접 테스트 — JSON + fallback 경로"""

    def test_json_correct(self):
        response = '{"verdict":"CORRECT","confidence":0.9,"reason":"relevant"}'
        result = _parse_verdict(response)
        assert result.verdict == Verdict.CORRECT
        assert result.confidence == 0.9
        assert result.reason == "relevant"

    def test_json_incorrect(self):
        response = '{"verdict":"INCORRECT","confidence":0.8,"reason":"not relevant"}'
        result = _parse_verdict(response)
        assert result.verdict == Verdict.INCORRECT
        assert result.confidence == 0.8

    def test_json_ambiguous(self):
        response = '{"verdict":"AMBIGUOUS","confidence":0.5,"reason":"unclear"}'
        result = _parse_verdict(response)
        assert result.verdict == Verdict.AMBIGUOUS
        assert result.confidence == 0.5

    def test_case_insensitive(self):
        """SKILL.md: verdict.upper() 처리 명시"""
        response = '{"verdict":"correct","confidence":0.9,"reason":"ok"}'
        result = _parse_verdict(response)
        assert result.verdict == Verdict.CORRECT

    def test_json_broken_fallback_to_correct(self):
        """깨진 JSON → string fallback → INCORRECT/AMBIGUOUS 없으면 CORRECT"""
        result = _parse_verdict("not valid json at all")
        assert result.verdict == Verdict.CORRECT

    def test_json_incorrect_string_fallback(self):
        """JSON 실패 → 문자열에 INCORRECT 있으면 탐지"""
        result = _parse_verdict("This is INCORRECT, the document is not relevant.")
        assert result.verdict == Verdict.INCORRECT

    def test_json_ambiguous_string_fallback(self):
        """JSON 실패 → 문자열에 AMBIGUOUS 있으면 탐지"""
        result = _parse_verdict("The result seems AMBIGUOUS and unclear.")
        assert result.verdict == Verdict.AMBIGUOUS

    def test_json_with_surrounding_text(self):
        """JSON이 다른 텍스트 사이에 있어도 추출"""
        response = 'Here is my assessment: {"verdict":"CORRECT","confidence":0.85,"reason":"relevant"} Done.'
        result = _parse_verdict(response)
        assert result.verdict == Verdict.CORRECT
        assert result.confidence == 0.85

    def test_missing_confidence_defaults_to_half(self):
        """confidence 없으면 0.5 기본값"""
        response = '{"verdict":"CORRECT","reason":"ok"}'
        result = _parse_verdict(response)
        assert result.confidence == 0.5


class TestEvaluateRetrieval:
    """evaluate_retrieval() — llm mock 사용"""

    def test_all_correct_returns_correct_status(self, sample_chunk_dicts):
        """crag.llm을 직접 패치 (from llm import llm 사용하므로)"""
        with patch("crag.llm", return_value='{"verdict":"CORRECT","confidence":0.9,"reason":"relevant"}'):
            chunks, status = evaluate_retrieval("What is VO2max?", sample_chunk_dicts)
        assert status == "correct"
        assert len(chunks) == len(sample_chunk_dicts)

    def test_all_incorrect_returns_empty(self, sample_chunk_dicts):
        with patch("crag.llm", return_value='{"verdict":"INCORRECT","confidence":0.1,"reason":"irrelevant"}'):
            chunks, status = evaluate_retrieval("unrelated query", sample_chunk_dicts)
        assert status == "incorrect"
        assert chunks == []

    def test_ambiguous_returned_when_no_correct(self, sample_chunk_dicts):
        with patch("crag.llm", return_value='{"verdict":"AMBIGUOUS","confidence":0.5,"reason":"unclear"}'):
            chunks, status = evaluate_retrieval("vague query", sample_chunk_dicts)
        assert status == "ambiguous"
        assert len(chunks) > 0

    def test_correct_prioritized_over_ambiguous(self, sample_chunk_dicts):
        """CORRECT 있으면 AMBIGUOUS보다 우선"""
        responses = iter([
            '{"verdict":"CORRECT","confidence":0.9,"reason":"relevant"}',
            '{"verdict":"AMBIGUOUS","confidence":0.5,"reason":"unclear"}',
        ])
        with patch("crag.llm", side_effect=lambda *a, **kw: next(responses)):
            chunks, status = evaluate_retrieval("query", sample_chunk_dicts)
        assert status == "correct"
        assert len(chunks) == 1  # CORRECT인 것만
