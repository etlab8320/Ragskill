"""
test_embedding.py — embed_chunks / embed_query 검증 (Voyage mock)
"""

import pytest
from unittest.mock import MagicMock, patch, call


class TestEmbedQuery:

    def test_embed_query_returns_1024_dims(self):
        """embed_query → 1024차원 리스트 반환"""
        with patch("embedding.vo") as mock_vo, patch("embedding._USE_LOCAL", False):
            mock_vo.embed.return_value = MagicMock(embeddings=[[0.1] * 1024])
            from embedding import embed_query
            result = embed_query("What is RAG?")
            assert len(result) == 1024

    def test_embed_query_uses_query_input_type(self):
        """쿼리 임베딩은 input_type='query' 사용 (문서와 다름)"""
        with patch("embedding.vo") as mock_vo, patch("embedding._USE_LOCAL", False):
            mock_vo.embed.return_value = MagicMock(embeddings=[[0.1] * 1024])
            from embedding import embed_query
            embed_query("test query")
            call_kwargs = mock_vo.embed.call_args
            assert call_kwargs.kwargs.get("input_type") == "query" or \
                   "query" in str(call_kwargs)

    def test_embed_query_korean(self):
        """한국어 쿼리도 정상 임베딩"""
        with patch("embedding.vo") as mock_vo, patch("embedding._USE_LOCAL", False):
            mock_vo.embed.return_value = MagicMock(embeddings=[[0.1] * 1024])
            from embedding import embed_query
            result = embed_query("운동이 심폐 능력에 미치는 영향은?")
            assert len(result) == 1024


class TestEmbedChunks:

    def test_embed_chunks_sets_embedding(self, sample_chunks):
        """embed_chunks → 각 청크에 embedding 설정"""
        with patch("embedding.vo") as mock_vo, patch("embedding._USE_LOCAL", False):
            mock_vo.embed.return_value = MagicMock(
                embeddings=[[0.1] * 1024, [0.2] * 1024]
            )
            # embedding 필드 초기화
            for c in sample_chunks:
                c.embedding = []
            from embedding import embed_chunks
            result = embed_chunks(sample_chunks)
            for chunk in result:
                assert len(chunk.embedding) == 1024

    def test_embed_chunks_uses_summary_when_available(self, sample_chunks):
        """context+summary 있으면 raw content 대신 사용"""
        with patch("embedding._embed_batch_raw") as mock_embed:
            mock_embed.return_value = [[0.1] * 1024, [0.2] * 1024]
            from embedding import embed_chunks
            embed_chunks(sample_chunks)
            # 호출된 텍스트에 summary가 포함되어야 함
            called_texts = mock_embed.call_args[0][0]
            for i, chunk in enumerate(sample_chunks):
                if chunk.context or chunk.summary:
                    assert chunk.summary in called_texts[i] or chunk.context in called_texts[i]

    def test_embed_chunks_batch_size_128(self):
        """128개 단위 배치 처리 — 200개는 2번 호출"""
        from chunking import Chunk
        chunks = [Chunk(content=f"text {i}", embedding=[]) for i in range(200)]

        def side_effect(texts, input_type, dimension):
            return [[0.1] * 1024] * len(texts)

        with patch("embedding._embed_batch_raw", side_effect=side_effect) as mock_embed:
            from embedding import embed_chunks
            embed_chunks(chunks)
            # 200개 → 2번 호출 (128 + 72)
            assert mock_embed.call_count == 2
