"""
test_chunking.py — semantic_chunk + late_chunk 폴백 검증
"""

import pytest
from unittest.mock import MagicMock, patch

from chunking import Chunk, semantic_chunk, late_chunk


class TestSemanticChunk:

    def test_basic_chunking(self):
        """긴 텍스트 → 여러 청크로 분리"""
        text = "\n\n".join([f"Paragraph {i}. " * 50 for i in range(10)])
        chunks = semantic_chunk(text, source="test.pdf")
        assert len(chunks) > 1

    def test_short_text_single_chunk(self):
        """짧은 텍스트 → 청크 1개"""
        text = "Short text about exercise."
        chunks = semantic_chunk(text, source="test.pdf")
        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_metadata_source_set(self):
        """source 메타데이터 올바르게 설정"""
        chunks = semantic_chunk("Some text.", source="paper.pdf")
        assert all(c.metadata["source"] == "paper.pdf" for c in chunks)

    def test_metadata_chunk_index(self):
        """chunk_index 순차적으로 설정"""
        text = "\n\n".join(["Word " * 100 for _ in range(5)])
        chunks = semantic_chunk(text, source="test.pdf", max_tokens=50)
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i

    def test_chunk_dataclass_defaults(self):
        """Chunk dataclass 기본값 확인"""
        chunk = Chunk(content="test")
        assert chunk.context == ""
        assert chunk.summary == ""
        assert chunk.embedding == []
        assert chunk.metadata == {}

    def test_empty_text(self):
        """빈 텍스트 → 빈 리스트 또는 1개 청크"""
        chunks = semantic_chunk("", source="empty.pdf")
        # 빈 단락은 필터링되므로 빈 리스트
        assert len(chunks) == 0

    def test_max_tokens_respected(self):
        """max_tokens 한도로 청킹되어 여러 청크 생성"""
        long_para = "word " * 600
        text = long_para + "\n\n" + long_para
        chunks = semantic_chunk(text, source="test.pdf", max_tokens=512)
        # 600*2 단어 → 512 한도로 나뉘어 최소 2개 이상
        assert len(chunks) >= 2


class TestLateChunk:

    def test_use_late_chunking_false_calls_semantic(self):
        """use_late_chunking=False → semantic_chunk() 호출 (Voyage 미사용)"""
        vo_mock = MagicMock()
        text = "Exercise improves health.\n\nMental training helps athletes."
        chunks = late_chunk(text, "test.pdf", vo=vo_mock, use_late_chunking=False)
        # vo.embed_chunks가 호출되지 않아야 함
        vo_mock.embed_chunks.assert_not_called()
        assert len(chunks) > 0

    def test_use_late_chunking_true_calls_embed_chunks(self):
        """use_late_chunking=True → vo.embed_chunks() 호출"""
        vo_mock = MagicMock()
        vo_mock.embed_chunks.return_value = MagicMock(
            embeddings=[[[0.1] * 1024, [0.2] * 1024]]
        )
        text = "\n\n".join(["Exercise text. " * 30 for _ in range(3)])
        chunks = late_chunk(text, "test.pdf", vo=vo_mock, use_late_chunking=True)
        vo_mock.embed_chunks.assert_called_once()
        # 임베딩이 주입되었는지 확인
        for chunk in chunks:
            assert len(chunk.embedding) == 1024

    def test_late_chunk_empty_text(self):
        """빈 텍스트 → 빈 리스트"""
        vo_mock = MagicMock()
        chunks = late_chunk("", "test.pdf", vo=vo_mock, use_late_chunking=True)
        assert chunks == []
        vo_mock.embed_chunks.assert_not_called()
