"""
test_graph_rag.py — GraphRAG 핵심 로직 테스트

테스트 전략:
  - _parse_graph_response(): LLM 응답 파싱 (mock 불필요)
  - _detect_communities(): BFS 그래프 순회 (순수 함수)
  - GraphStore: DB mock을 통한 upsert/traverse 검증
  - build_graph(): llm mock + GraphStore mock 통합
  - summarize_communities(): 커뮤니티 요약 흐름
  - graph_augment(): 쿼리 보강 end-to-end
"""

import json
import pytest
from unittest.mock import MagicMock, patch


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_graph_pool():
    """
    GraphStore는 ChunkStore._pool을 직접 참조.
    ChunkStore._pool을 MagicMock으로 교체하고 복원.
    """
    import storage
    pool = MagicMock()
    conn = MagicMock()
    cursor = MagicMock()

    # context manager 설정
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cursor
    cursor.fetchone.return_value = (42,)       # upsert_node RETURNING id
    cursor.fetchall.return_value = []          # 기본: 빈 결과
    pool.connection.return_value = conn

    storage.ChunkStore._pool = pool
    yield pool, conn, cursor
    storage.ChunkStore._pool = None


@pytest.fixture
def sample_nodes():
    return [
        {"id": 1, "name": "Alice", "type": "PERSON", "description": "Engineer"},
        {"id": 2, "name": "Acme Corp", "type": "ORG", "description": "Company"},
        {"id": 3, "name": "Bob", "type": "PERSON", "description": "Manager"},
        {"id": 4, "name": "Isolated", "type": "CONCEPT", "description": "Solo"},
    ]


@pytest.fixture
def sample_edges():
    return [
        {"source_id": 1, "target_id": 2, "relation": "works_for", "weight": 0.9},
        {"source_id": 3, "target_id": 2, "relation": "works_for", "weight": 0.8},
    ]


# ── _parse_graph_response() ───────────────────────────────────────────────────

class TestParseGraphResponse:
    """JSON 응답 파싱 — 순수 함수, mock 불필요"""

    def test_valid_json_entities(self):
        from graph_rag import _parse_graph_response
        response = json.dumps({
            "entities": [
                {"name": "Alice", "type": "PERSON", "description": "Engineer"},
                {"name": "Acme Corp", "type": "ORG", "description": "Tech company"},
            ],
            "relations": []
        })
        entities, relations = _parse_graph_response(response)
        assert len(entities) == 2
        assert entities[0].name == "Alice"
        assert entities[0].type == "PERSON"
        assert entities[1].name == "Acme Corp"

    def test_valid_json_relations(self):
        from graph_rag import _parse_graph_response
        response = json.dumps({
            "entities": [
                {"name": "Alice", "type": "PERSON", "description": ""},
                {"name": "Acme", "type": "ORG", "description": ""},
            ],
            "relations": [
                {"source": "Alice", "target": "Acme", "relation": "works_for", "weight": 0.9}
            ]
        })
        entities, relations = _parse_graph_response(response)
        assert len(relations) == 1
        assert relations[0].source == "Alice"
        assert relations[0].target == "Acme"
        assert relations[0].relation == "works_for"
        assert relations[0].weight == 0.9

    def test_json_embedded_in_text(self):
        """JSON이 다른 텍스트 사이에 있어도 추출"""
        from graph_rag import _parse_graph_response
        response = 'Here is the result: ' + json.dumps({
            "entities": [{"name": "Bob", "type": "PERSON", "description": "dev"}],
            "relations": []
        }) + " That is all."
        entities, _ = _parse_graph_response(response)
        assert len(entities) == 1
        assert entities[0].name == "Bob"

    def test_entity_type_uppercased(self):
        """type 소문자도 대문자로 정규화"""
        from graph_rag import _parse_graph_response
        response = json.dumps({
            "entities": [{"name": "Seoul", "type": "location", "description": "city"}],
            "relations": []
        })
        entities, _ = _parse_graph_response(response)
        assert entities[0].type == "LOCATION"

    def test_skips_entities_without_name(self):
        """name 없는 엔티티 필터링"""
        from graph_rag import _parse_graph_response
        response = json.dumps({
            "entities": [
                {"name": "", "type": "PERSON", "description": ""},
                {"name": "Alice", "type": "PERSON", "description": "ok"},
            ],
            "relations": []
        })
        entities, _ = _parse_graph_response(response)
        assert len(entities) == 1

    def test_skips_relations_without_source_or_target(self):
        """source/target 없는 relation 필터링"""
        from graph_rag import _parse_graph_response
        response = json.dumps({
            "entities": [],
            "relations": [
                {"source": "", "target": "Acme", "relation": "works_for"},
                {"source": "Alice", "target": "", "relation": "works_for"},
                {"source": "Alice", "target": "Acme", "relation": "works_for"},
            ]
        })
        _, relations = _parse_graph_response(response)
        assert len(relations) == 1

    def test_default_relation_weight(self):
        """weight 없으면 기본값 1.0"""
        from graph_rag import _parse_graph_response
        response = json.dumps({
            "entities": [],
            "relations": [{"source": "A", "target": "B", "relation": "related_to"}]
        })
        _, relations = _parse_graph_response(response)
        assert relations[0].weight == 1.0

    def test_invalid_json_returns_empty(self):
        """JSON 파싱 실패 → 빈 결과"""
        from graph_rag import _parse_graph_response
        entities, relations = _parse_graph_response("not json at all")
        assert entities == []
        assert relations == []

    def test_empty_response_returns_empty(self):
        from graph_rag import _parse_graph_response
        entities, relations = _parse_graph_response("")
        assert entities == []
        assert relations == []


# ── _detect_communities() ─────────────────────────────────────────────────────

class TestDetectCommunities:
    """BFS 커뮤니티 탐지 — 순수 함수, mock 불필요"""

    def test_connected_nodes_same_community(self, sample_nodes, sample_edges):
        from graph_rag import _detect_communities
        result = _detect_communities(sample_nodes, sample_edges)
        # 1(Alice) - 2(Acme) - 3(Bob) 연결 → 같은 커뮤니티
        assert result[1] == result[2] == result[3]

    def test_isolated_node_own_community(self, sample_nodes, sample_edges):
        from graph_rag import _detect_communities
        result = _detect_communities(sample_nodes, sample_edges)
        # 4(Isolated)는 엣지 없음 → 별도 커뮤니티
        assert result[4] != result[1]

    def test_empty_graph(self):
        from graph_rag import _detect_communities
        result = _detect_communities([], [])
        assert result == {}

    def test_no_edges_all_singletons(self):
        """엣지 없으면 모두 다른 커뮤니티"""
        from graph_rag import _detect_communities
        nodes = [
            {"id": 1, "name": "A"},
            {"id": 2, "name": "B"},
            {"id": 3, "name": "C"},
        ]
        result = _detect_communities(nodes, [])
        assert result[1] != result[2]
        assert result[2] != result[3]

    def test_all_connected_same_community(self):
        """전부 연결된 경우 모두 같은 커뮤니티"""
        from graph_rag import _detect_communities
        nodes = [{"id": i} for i in range(1, 5)]
        edges = [
            {"source_id": 1, "target_id": 2, "relation": "r", "weight": 1.0},
            {"source_id": 2, "target_id": 3, "relation": "r", "weight": 1.0},
            {"source_id": 3, "target_id": 4, "relation": "r", "weight": 1.0},
        ]
        result = _detect_communities(nodes, edges)
        assert result[1] == result[2] == result[3] == result[4]

    def test_community_ids_are_integers(self, sample_nodes, sample_edges):
        from graph_rag import _detect_communities
        result = _detect_communities(sample_nodes, sample_edges)
        for cid in result.values():
            assert isinstance(cid, int)

    def test_all_nodes_assigned(self, sample_nodes, sample_edges):
        """모든 노드에 커뮤니티 ID 할당"""
        from graph_rag import _detect_communities
        result = _detect_communities(sample_nodes, sample_edges)
        node_ids = {n["id"] for n in sample_nodes}
        assert set(result.keys()) == node_ids


# ── GraphStore (DB mock) ──────────────────────────────────────────────────────

class TestGraphStore:
    """GraphStore DB 작업 검증 (mock_graph_pool 사용)"""

    def test_init_requires_chunk_store_pool(self):
        """ChunkStore._pool 없으면 StorageError"""
        import storage
        storage.ChunkStore._pool = None
        from exceptions import StorageError
        from graph_rag import GraphStore
        with pytest.raises(StorageError):
            GraphStore()

    def test_upsert_node_returns_id(self, mock_graph_pool):
        """upsert_node → cursor.fetchone()[0] 반환"""
        pool, conn, cursor = mock_graph_pool
        cursor.fetchone.return_value = (42,)
        from graph_rag import GraphStore, Entity
        store = GraphStore()
        entity = Entity(name="Alice", type="PERSON", description="Engineer")
        node_id = store.upsert_node(entity, source="doc.pdf")
        assert node_id == 42

    def test_upsert_node_executes_insert(self, mock_graph_pool):
        """upsert_node → INSERT ... ON CONFLICT 실행"""
        pool, conn, cursor = mock_graph_pool
        from graph_rag import GraphStore, Entity
        store = GraphStore()
        entity = Entity(name="Alice", type="PERSON", description="")
        store.upsert_node(entity, source="doc.pdf")
        cursor.execute.assert_called_once()
        sql = cursor.execute.call_args[0][0]
        assert "INSERT INTO graph_nodes" in sql
        assert "ON CONFLICT" in sql

    def test_upsert_edge_executes_insert(self, mock_graph_pool):
        """upsert_edge → INSERT INTO graph_edges 실행"""
        pool, conn, cursor = mock_graph_pool
        from graph_rag import GraphStore
        store = GraphStore()
        store.upsert_edge(1, 2, "works_for", 0.9)
        conn.execute.assert_called_once()
        sql = conn.execute.call_args[0][0]
        assert "INSERT INTO graph_edges" in sql

    def test_set_community_updates_node(self, mock_graph_pool):
        """set_community → UPDATE graph_nodes SET community_id"""
        pool, conn, cursor = mock_graph_pool
        from graph_rag import GraphStore
        store = GraphStore()
        store.set_community(node_id=1, community_id=0)
        conn.execute.assert_called_once()
        sql = conn.execute.call_args[0][0]
        assert "UPDATE graph_nodes" in sql

    def test_save_community_summary_inserts(self, mock_graph_pool):
        """save_community_summary → INSERT INTO graph_communities"""
        pool, conn, cursor = mock_graph_pool
        from graph_rag import GraphStore
        store = GraphStore()
        store.save_community_summary(0, [1, 2, 3], "Summary text.")
        conn.execute.assert_called_once()
        sql = conn.execute.call_args[0][0]
        assert "INSERT INTO graph_communities" in sql

    def test_get_all_nodes_returns_list(self, mock_graph_pool):
        """get_all_nodes → list[dict]"""
        pool, conn, cursor = mock_graph_pool
        cursor.fetchall.return_value = [
            (1, "Alice", "PERSON", "Engineer"),
            (2, "Acme", "ORG", "Company"),
        ]
        from graph_rag import GraphStore
        store = GraphStore()
        nodes = store.get_all_nodes()
        assert len(nodes) == 2
        assert nodes[0]["name"] == "Alice"
        assert nodes[1]["type"] == "ORG"

    def test_get_all_edges_returns_list(self, mock_graph_pool):
        """get_all_edges → list[dict]"""
        pool, conn, cursor = mock_graph_pool
        cursor.fetchall.return_value = [
            (1, 2, "works_for", 0.9),
        ]
        from graph_rag import GraphStore
        store = GraphStore()
        edges = store.get_all_edges()
        assert len(edges) == 1
        assert edges[0]["source_id"] == 1
        assert edges[0]["relation"] == "works_for"

    def test_traverse_empty_when_no_entities(self, mock_graph_pool):
        """entity_names 비어있으면 DB 쿼리 없이 빈 리스트"""
        pool, conn, cursor = mock_graph_pool
        from graph_rag import GraphStore
        store = GraphStore()
        result = store.traverse([])
        assert result == []
        cursor.execute.assert_not_called()

    def test_traverse_returns_list(self, mock_graph_pool):
        """traverse → list[dict]"""
        pool, conn, cursor = mock_graph_pool
        cursor.fetchall.return_value = [
            (1, "Alice", "PERSON", "Engineer", "Community summary here"),
        ]
        from graph_rag import GraphStore
        store = GraphStore()
        result = store.traverse(["Alice"])
        assert len(result) == 1
        assert result[0]["name"] == "Alice"
        assert result[0]["community_summary"] == "Community summary here"

    def test_get_community_summaries_empty_ids(self, mock_graph_pool):
        """빈 community_ids → DB 쿼리 없이 빈 리스트"""
        pool, conn, cursor = mock_graph_pool
        from graph_rag import GraphStore
        store = GraphStore()
        result = store.get_community_summaries([])
        assert result == []
        cursor.execute.assert_not_called()


# ── build_graph() ─────────────────────────────────────────────────────────────

class TestBuildGraph:
    """build_graph() — llm mock + GraphStore mock"""

    def test_build_graph_calls_llm(self, mock_graph_pool):
        from graph_rag import build_graph, GraphStore
        llm_response = json.dumps({
            "entities": [{"name": "Alice", "type": "PERSON", "description": "dev"}],
            "relations": []
        })
        with patch("graph_rag.llm", return_value=llm_response):
            graph = GraphStore()
            build_graph("Alice is a developer.", "doc.pdf", graph)
            import graph_rag as gr
            gr.llm.assert_called_once()

    def test_build_graph_stores_entities(self, mock_graph_pool):
        """엔티티 추출 후 upsert_node 호출"""
        pool, conn, cursor = mock_graph_pool
        cursor.fetchone.return_value = (1,)
        from graph_rag import build_graph, GraphStore
        llm_response = json.dumps({
            "entities": [
                {"name": "Alice", "type": "PERSON", "description": "dev"},
                {"name": "Acme", "type": "ORG", "description": "company"},
            ],
            "relations": [{"source": "Alice", "target": "Acme", "relation": "works_for", "weight": 0.9}]
        })
        with patch("graph_rag.llm", return_value=llm_response):
            graph = GraphStore()
            build_graph("Alice works for Acme Corp.", "doc.pdf", graph)
        # upsert_node → cursor.execute 2회 (엔티티 2개)
        assert cursor.execute.call_count == 2

    def test_build_graph_skips_self_loop_edges(self, mock_graph_pool):
        """source_id == target_id 엣지는 저장 안 함"""
        pool, conn, cursor = mock_graph_pool
        cursor.fetchone.return_value = (1,)  # 두 엔티티가 같은 id 반환 (극단적 mock)
        from graph_rag import build_graph, GraphStore
        llm_response = json.dumps({
            "entities": [{"name": "A", "type": "CONCEPT", "description": ""}],
            "relations": [{"source": "A", "target": "A", "relation": "self_ref", "weight": 0.5}]
        })
        with patch("graph_rag.llm", return_value=llm_response):
            graph = GraphStore()
            build_graph("A refers to A.", "doc.pdf", graph)
        # conn.execute (upsert_edge) 호출 안 됨
        conn.execute.assert_not_called()

    def test_build_graph_empty_llm_response(self, mock_graph_pool):
        """LLM이 파싱 불가 응답 → 에러 없이 통과"""
        from graph_rag import build_graph, GraphStore
        with patch("graph_rag.llm", return_value="invalid json"):
            graph = GraphStore()
            build_graph("Some text.", "doc.pdf", graph)  # 예외 없어야 함


# ── summarize_communities() ───────────────────────────────────────────────────

class TestSummarizeCommunities:
    """summarize_communities() — 전체 흐름 검증"""

    def test_no_nodes_returns_early(self, mock_graph_pool):
        """노드 없으면 LLM 호출 없이 바로 반환"""
        pool, conn, cursor = mock_graph_pool
        cursor.fetchall.return_value = []
        from graph_rag import GraphStore, summarize_communities
        graph = GraphStore()
        with patch("graph_rag.llm") as mock_llm:
            summarize_communities(graph)
            mock_llm.assert_not_called()

    def test_singleton_community_skipped(self, mock_graph_pool):
        """노드 1개짜리 커뮤니티는 요약 안 함"""
        pool, conn, cursor = mock_graph_pool
        # get_all_nodes → 1개, get_all_edges → 없음 (2번 fetchall 호출됨)
        cursor.fetchall.side_effect = [
            [(1, "Solo", "CONCEPT", "alone")],  # get_all_nodes
            [],                                   # get_all_edges
        ]
        from graph_rag import GraphStore, summarize_communities
        graph = GraphStore()
        with patch("graph_rag.llm") as mock_llm:
            summarize_communities(graph)
            mock_llm.assert_not_called()

    def test_multi_node_community_gets_summary(self, mock_graph_pool):
        """2개 이상 노드 커뮤니티 → LLM 호출"""
        pool, conn, cursor = mock_graph_pool
        cursor.fetchall.side_effect = [
            [                                       # get_all_nodes
                (1, "Alice", "PERSON", "dev"),
                (2, "Acme", "ORG", "company"),
            ],
            [                                       # get_all_edges
                (1, 2, "works_for", 0.9),
            ],
        ]
        from graph_rag import GraphStore, summarize_communities
        graph = GraphStore()
        with patch("graph_rag.llm", return_value="Alice works for Acme.") as mock_llm:
            summarize_communities(graph)
            mock_llm.assert_called_once()


# ── graph_augment() ───────────────────────────────────────────────────────────

class TestGraphAugment:
    """graph_augment() — 쿼리 보강 end-to-end"""

    def test_returns_empty_when_no_entities(self, mock_graph_pool):
        """LLM이 빈 엔티티 반환 → 빈 문자열"""
        from graph_rag import GraphStore, graph_augment
        graph = GraphStore()
        with patch("graph_rag.llm", return_value='{"entities": []}'):
            result = graph_augment("vague query", graph)
        assert result == ""

    def test_returns_empty_when_traverse_empty(self, mock_graph_pool):
        """엔티티 있어도 traverse 결과 없으면 빈 문자열"""
        pool, conn, cursor = mock_graph_pool
        cursor.fetchall.return_value = []  # traverse → 빈 결과
        from graph_rag import GraphStore, graph_augment
        graph = GraphStore()
        with patch("graph_rag.llm", return_value='{"entities": ["Alice"]}'):
            result = graph_augment("Who is Alice?", graph)
        assert result == ""

    def test_includes_knowledge_graph_header(self, mock_graph_pool):
        """결과에 [Knowledge Graph Context] 헤더 포함"""
        pool, conn, cursor = mock_graph_pool
        cursor.fetchall.side_effect = [
            [(1, "Alice", "PERSON", "Engineer", None)],  # traverse
            [],                                            # get_community_summaries
        ]
        from graph_rag import GraphStore, graph_augment
        graph = GraphStore()
        with patch("graph_rag.llm", return_value='{"entities": ["Alice"]}'):
            result = graph_augment("Who is Alice?", graph)
        assert "[Knowledge Graph Context]" in result

    def test_includes_entity_descriptions(self, mock_graph_pool):
        """관련 엔티티 설명이 결과에 포함"""
        pool, conn, cursor = mock_graph_pool
        cursor.fetchall.side_effect = [
            [(1, "Alice", "PERSON", "Senior Engineer at Acme", None)],
            [],
        ]
        from graph_rag import GraphStore, graph_augment
        graph = GraphStore()
        with patch("graph_rag.llm", return_value='{"entities": ["Alice"]}'):
            result = graph_augment("Who is Alice?", graph)
        assert "Alice" in result
        assert "Senior Engineer at Acme" in result

    def test_includes_community_summaries(self, mock_graph_pool):
        """traverse SQL이 community_summary를 직접 반환 → 결과에 포함"""
        pool, conn, cursor = mock_graph_pool
        # traverse: community_summary가 이미 JOIN돼서 반환됨
        cursor.fetchall.return_value = [
            (1, "Alice", "PERSON", "dev", "Alice works at Acme Corp."),
        ]
        from graph_rag import GraphStore, graph_augment
        graph = GraphStore()
        with patch("graph_rag.llm", return_value='{"entities": ["Alice"]}'):
            result = graph_augment("Who is Alice?", graph)
        assert "Community insight" in result
        assert "Alice works at Acme Corp." in result

    def test_invalid_llm_json_returns_empty(self, mock_graph_pool):
        """LLM 응답이 JSON이 아니면 빈 문자열"""
        from graph_rag import GraphStore, graph_augment
        graph = GraphStore()
        with patch("graph_rag.llm", return_value="not valid json"):
            result = graph_augment("some query", graph)
        assert result == ""
