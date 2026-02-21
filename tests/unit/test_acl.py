"""
test_acl.py — 권한 관리 + 스마트 재인제스션 테스트

테스트 전략:
  - AccessLevel: 열거형 값 검증
  - get_accessible_levels(): 역할 계층 + 미등록 역할 폴백
  - tag_chunk(): 청크 메타데이터 태깅
  - smart_ingest(): delete_by_source 호출 + 재인제스션 흐름
"""

import pytest
from unittest.mock import MagicMock, patch, call


# ── AccessLevel ───────────────────────────────────────────────────────────────

class TestAccessLevel:
    """AccessLevel 열거형 값 검증"""

    def test_all_levels_defined(self):
        from acl import AccessLevel
        expected = {"public", "hr", "finance", "executive", "confidential"}
        actual = {level.value for level in AccessLevel}
        assert actual == expected

    def test_string_comparison(self):
        """AccessLevel은 str 상속 — 문자열과 직접 비교 가능"""
        from acl import AccessLevel
        assert AccessLevel.PUBLIC == "public"
        assert AccessLevel.HR == "hr"
        assert AccessLevel.FINANCE == "finance"
        assert AccessLevel.EXECUTIVE == "executive"
        assert AccessLevel.CONFIDENTIAL == "confidential"

    def test_enum_from_value(self):
        """문자열로 AccessLevel 생성"""
        from acl import AccessLevel
        assert AccessLevel("hr") == AccessLevel.HR
        assert AccessLevel("public") == AccessLevel.PUBLIC


# ── get_accessible_levels() ───────────────────────────────────────────────────

class TestGetAccessibleLevels:
    """역할별 접근 레벨 반환 검증"""

    def test_employee_only_public(self):
        from acl import get_accessible_levels
        levels = get_accessible_levels("employee")
        assert "public" in levels
        assert "hr" not in levels
        assert "finance" not in levels
        assert "executive" not in levels

    def test_hr_sees_public_and_hr(self):
        from acl import get_accessible_levels
        levels = get_accessible_levels("hr")
        assert "public" in levels
        assert "hr" in levels
        assert "finance" not in levels
        assert "executive" not in levels

    def test_finance_sees_public_and_finance(self):
        from acl import get_accessible_levels
        levels = get_accessible_levels("finance")
        assert "public" in levels
        assert "finance" in levels
        assert "hr" not in levels
        assert "executive" not in levels

    def test_executive_sees_all_standard_levels(self):
        from acl import get_accessible_levels
        levels = get_accessible_levels("executive")
        assert "public" in levels
        assert "hr" in levels
        assert "finance" in levels
        assert "executive" in levels

    def test_unknown_role_fallback_to_public_only(self):
        """미등록 역할 → public만 접근"""
        from acl import get_accessible_levels
        levels = get_accessible_levels("unknown_role")
        assert levels == ["public"]

    def test_returns_list(self):
        from acl import get_accessible_levels
        result = get_accessible_levels("employee")
        assert isinstance(result, list)

    def test_no_duplicates(self):
        """같은 레벨이 중복으로 들어가지 않음"""
        from acl import get_accessible_levels
        for role in ["employee", "hr", "finance", "executive"]:
            levels = get_accessible_levels(role)
            assert len(levels) == len(set(levels)), f"Duplicates in role={role}"

    def test_executive_does_not_include_confidential(self):
        """confidential은 역할 계층에 포함 안 됨 — 명시적 지정만 허용"""
        from acl import get_accessible_levels
        levels = get_accessible_levels("executive")
        assert "confidential" not in levels


# ── tag_chunk() ───────────────────────────────────────────────────────────────

class TestTagChunk:
    """청크 메타데이터 접근 레벨 태깅"""

    def _make_chunk(self):
        from chunking import Chunk
        return Chunk(
            content="Test content",
            metadata={"source": "test.pdf"},
        )

    def test_tag_with_enum(self):
        from acl import AccessLevel, tag_chunk
        chunk = self._make_chunk()
        tag_chunk(chunk, AccessLevel.HR)
        assert chunk.metadata["access_level"] == "hr"

    def test_tag_with_string(self):
        """문자열로도 태깅 가능"""
        from acl import tag_chunk
        chunk = self._make_chunk()
        tag_chunk(chunk, "finance")
        assert chunk.metadata["access_level"] == "finance"

    def test_tag_public(self):
        from acl import AccessLevel, tag_chunk
        chunk = self._make_chunk()
        tag_chunk(chunk, AccessLevel.PUBLIC)
        assert chunk.metadata["access_level"] == "public"

    def test_tag_executive(self):
        from acl import AccessLevel, tag_chunk
        chunk = self._make_chunk()
        tag_chunk(chunk, AccessLevel.EXECUTIVE)
        assert chunk.metadata["access_level"] == "executive"

    def test_tag_confidential(self):
        from acl import AccessLevel, tag_chunk
        chunk = self._make_chunk()
        tag_chunk(chunk, AccessLevel.CONFIDENTIAL)
        assert chunk.metadata["access_level"] == "confidential"

    def test_tag_preserves_existing_metadata(self):
        """태깅 시 기존 메타데이터 유지"""
        from acl import AccessLevel, tag_chunk
        chunk = self._make_chunk()
        chunk.metadata["chunk_index"] = 3
        tag_chunk(chunk, AccessLevel.HR)
        assert chunk.metadata["chunk_index"] == 3
        assert chunk.metadata["access_level"] == "hr"

    def test_tag_overwrites_existing_level(self):
        """기존 access_level이 있으면 덮어씀"""
        from acl import AccessLevel, tag_chunk
        chunk = self._make_chunk()
        tag_chunk(chunk, AccessLevel.PUBLIC)
        tag_chunk(chunk, AccessLevel.EXECUTIVE)
        assert chunk.metadata["access_level"] == "executive"


# ── smart_ingest() ────────────────────────────────────────────────────────────

class TestSmartIngest:
    """smart_ingest() — auto-delete + re-ingest 흐름"""

    def _make_store_graph(self):
        store = MagicMock()
        graph = MagicMock()
        return store, graph

    def _make_fake_chunks(self, n=2):
        from chunking import Chunk
        return [
            Chunk(
                content=f"chunk {i}",
                metadata={"source": "test.pdf", "chunk_index": i},
                context="ctx",
                summary="sum",
                embedding=[0.1] * 1024,
            )
            for i in range(n)
        ]

    def test_deletes_existing_before_ingest(self):
        """재인제스션 시 기존 데이터 먼저 삭제"""
        from smart_ingest import smart_ingest
        from acl import AccessLevel
        store, graph = self._make_store_graph()
        fake_chunks = self._make_fake_chunks(3)

        with patch("smart_ingest.load_document", return_value="doc text"), \
             patch("smart_ingest.semantic_chunk", return_value=fake_chunks), \
             patch("smart_ingest.enrich_chunk"), \
             patch("smart_ingest.tag_chunk"), \
             patch("smart_ingest.build_graph"), \
             patch("smart_ingest.embed_chunks"):
            smart_ingest("test.pdf", store, graph, AccessLevel.PUBLIC)

        store.delete_by_source.assert_called_once_with("test.pdf")
        graph.delete_by_source.assert_called_once_with("test.pdf")

    def test_delete_called_before_store(self):
        """delete → store 순서 보장"""
        from smart_ingest import smart_ingest
        from acl import AccessLevel
        store, graph = self._make_store_graph()
        call_order = []
        store.delete_by_source.side_effect = lambda _: call_order.append("delete")
        store.store_batch.side_effect = lambda _: call_order.append("store")
        fake_chunks = self._make_fake_chunks(1)

        with patch("smart_ingest.load_document", return_value="text"), \
             patch("smart_ingest.semantic_chunk", return_value=fake_chunks), \
             patch("smart_ingest.enrich_chunk"), \
             patch("smart_ingest.tag_chunk"), \
             patch("smart_ingest.build_graph"), \
             patch("smart_ingest.embed_chunks"):
            smart_ingest("test.pdf", store, graph, AccessLevel.PUBLIC)

        assert call_order.index("delete") < call_order.index("store")

    def test_returns_chunk_count(self):
        """인제스션된 청크 수 반환"""
        from smart_ingest import smart_ingest
        from acl import AccessLevel
        store, graph = self._make_store_graph()
        fake_chunks = self._make_fake_chunks(5)

        with patch("smart_ingest.load_document", return_value="text"), \
             patch("smart_ingest.semantic_chunk", return_value=fake_chunks), \
             patch("smart_ingest.enrich_chunk"), \
             patch("smart_ingest.tag_chunk"), \
             patch("smart_ingest.build_graph"), \
             patch("smart_ingest.embed_chunks"):
            count = smart_ingest("test.pdf", store, graph, AccessLevel.PUBLIC)

        assert count == 5

    def test_tags_all_chunks_with_access_level(self):
        """모든 청크에 접근 레벨 태깅"""
        from smart_ingest import smart_ingest
        from acl import AccessLevel
        store, graph = self._make_store_graph()
        fake_chunks = self._make_fake_chunks(3)

        with patch("smart_ingest.load_document", return_value="text"), \
             patch("smart_ingest.semantic_chunk", return_value=fake_chunks), \
             patch("smart_ingest.enrich_chunk"), \
             patch("smart_ingest.tag_chunk") as mock_tag, \
             patch("smart_ingest.build_graph"), \
             patch("smart_ingest.embed_chunks"):
            smart_ingest("test.pdf", store, graph, AccessLevel.HR)

        assert mock_tag.call_count == 3
        # 모든 호출에 HR 레벨 전달됐는지 확인
        for c in mock_tag.call_args_list:
            assert c[0][1] == AccessLevel.HR

    def test_builds_graph_for_each_chunk(self):
        """청크마다 build_graph 호출"""
        from smart_ingest import smart_ingest
        from acl import AccessLevel
        store, graph = self._make_store_graph()
        fake_chunks = self._make_fake_chunks(4)

        with patch("smart_ingest.load_document", return_value="text"), \
             patch("smart_ingest.semantic_chunk", return_value=fake_chunks), \
             patch("smart_ingest.enrich_chunk"), \
             patch("smart_ingest.tag_chunk"), \
             patch("smart_ingest.build_graph") as mock_build, \
             patch("smart_ingest.embed_chunks"):
            smart_ingest("test.pdf", store, graph, AccessLevel.PUBLIC)

        assert mock_build.call_count == 4

    def test_store_batch_called_once(self):
        """store_batch는 한 번만 (배치로) 호출"""
        from smart_ingest import smart_ingest
        from acl import AccessLevel
        store, graph = self._make_store_graph()
        fake_chunks = self._make_fake_chunks(3)

        with patch("smart_ingest.load_document", return_value="text"), \
             patch("smart_ingest.semantic_chunk", return_value=fake_chunks), \
             patch("smart_ingest.enrich_chunk"), \
             patch("smart_ingest.tag_chunk"), \
             patch("smart_ingest.build_graph"), \
             patch("smart_ingest.embed_chunks"):
            smart_ingest("test.pdf", store, graph, AccessLevel.PUBLIC)

        store.store_batch.assert_called_once()

    def test_uses_filename_as_source(self):
        """전체 경로에서 파일명만 source로 사용"""
        from smart_ingest import smart_ingest
        from acl import AccessLevel
        store, graph = self._make_store_graph()
        fake_chunks = self._make_fake_chunks(1)

        with patch("smart_ingest.load_document", return_value="text"), \
             patch("smart_ingest.semantic_chunk", return_value=fake_chunks) as mock_chunk, \
             patch("smart_ingest.enrich_chunk"), \
             patch("smart_ingest.tag_chunk"), \
             patch("smart_ingest.build_graph"), \
             patch("smart_ingest.embed_chunks"):
            smart_ingest("/some/deep/path/hr_policy.pdf", store, graph, AccessLevel.HR)

        # source는 파일명만 (smart_ingest는 keyword arg로 전달)
        call_kwargs = mock_chunk.call_args[1]
        assert call_kwargs["source"] == "hr_policy.pdf"
        store.delete_by_source.assert_called_once_with("hr_policy.pdf")

    def test_default_access_level_is_public(self):
        """access_level 기본값 = PUBLIC"""
        from smart_ingest import smart_ingest
        from acl import AccessLevel
        store, graph = self._make_store_graph()
        fake_chunks = self._make_fake_chunks(1)

        with patch("smart_ingest.load_document", return_value="text"), \
             patch("smart_ingest.semantic_chunk", return_value=fake_chunks), \
             patch("smart_ingest.enrich_chunk"), \
             patch("smart_ingest.tag_chunk") as mock_tag, \
             patch("smart_ingest.build_graph"), \
             patch("smart_ingest.embed_chunks"):
            smart_ingest("test.pdf", store, graph)  # access_level 생략

        assert mock_tag.call_args[0][1] == AccessLevel.PUBLIC
