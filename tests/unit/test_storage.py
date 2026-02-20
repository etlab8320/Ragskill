"""
test_storage.py — ChunkStore 싱글턴 패턴 + DB 작업 검증

mock_pool: ConnectionPool 클래스 mock
  mock_pool.return_value = pool 인스턴스
  mock_pool.return_value.connection.return_value = conn
"""

import pytest
import json
from unittest.mock import MagicMock, patch


class TestChunkStoreSingleton:

    def setup_method(self):
        """각 테스트 전 _pool 초기화"""
        import storage
        storage.ChunkStore._pool = None

    def test_singleton_pool_reused(self, mock_pool):
        """두 번 ChunkStore() 생성해도 ConnectionPool은 한 번만 호출"""
        from storage import ChunkStore
        s1 = ChunkStore()
        s2 = ChunkStore()
        # ConnectionPool 클래스가 1번만 instantiate되어야 함
        assert mock_pool.call_count == 1

    def test_close_pool_sets_none(self, mock_pool):
        """close_pool() 후 _pool = None"""
        from storage import ChunkStore
        ChunkStore()
        assert ChunkStore._pool is not None
        ChunkStore.close_pool()
        assert ChunkStore._pool is None

    def test_close_pool_idempotent(self):
        """close_pool() _pool 없는 상태에서 호출해도 에러 없음"""
        from storage import ChunkStore
        ChunkStore._pool = None
        ChunkStore.close_pool()  # 에러 없어야 함


class TestChunkStoreOperations:

    def setup_method(self):
        import storage
        storage.ChunkStore._pool = None

    def test_store_batch_uses_executemany(self, mock_pool, sample_chunks):
        """store_batch → executemany 호출"""
        from storage import ChunkStore
        store = ChunkStore()

        pool = mock_pool.return_value          # pool 인스턴스
        conn = pool.connection.return_value    # conn 컨텍스트 매니저
        cursor = conn.cursor.return_value      # cursor 컨텍스트 매니저

        store.store_batch(sample_chunks)
        cursor.executemany.assert_called_once()

    def test_store_batch_serializes_metadata(self, mock_pool, sample_chunks):
        """metadata가 JSON 문자열로 직렬화되어 저장"""
        from storage import ChunkStore
        store = ChunkStore()

        pool = mock_pool.return_value
        conn = pool.connection.return_value
        cursor = conn.cursor.return_value

        store.store_batch(sample_chunks)

        call_args = cursor.executemany.call_args
        rows = call_args[0][1]  # executemany(sql, rows)
        for row in rows:
            assert isinstance(row[3], str)  # metadata는 JSON 문자열
            parsed = json.loads(row[3])
            assert isinstance(parsed, dict)

    def test_hybrid_search_returns_list(self, mock_pool):
        """hybrid_search → list 반환"""
        from storage import ChunkStore
        store = ChunkStore()
        results = store.hybrid_search([0.1] * 1024, "exercise", top_k=5)
        assert isinstance(results, list)

    def test_storage_error_on_pool_failure(self):
        """ConnectionPool 실패 시 StorageError 발생"""
        from exceptions import StorageError
        import storage
        storage.ChunkStore._pool = None
        with patch("storage.ConnectionPool", side_effect=Exception("connection failed")):
            from storage import ChunkStore
            with pytest.raises(StorageError):
                ChunkStore()
