"""
루트 conftest.py — SKILL.md Python 블록을 conftest 로드 시점에 추출해 sys.path에 추가.

pytest가 conftest.py를 collection 전에 로드하므로, 이 파일이 임포트될 때
SKILL.md Named Block을 임시 디렉토리에 추출하고 sys.path에 추가한다.
이후 tests/unit/*.py의 top-level import가 정상 동작.
"""

import re
import sys
import atexit
import shutil
import tempfile
from pathlib import Path

import pytest

SKILL_MD = Path(__file__).parent / "skill" / "SKILL.md"


def _extract_named_blocks(skill_path: Path) -> dict[str, str]:
    text = skill_path.read_text(encoding="utf-8")
    raw_blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)

    blocks = {}
    for block in raw_blocks:
        first_line = block.split("\n")[0].strip()
        match = re.match(r"^#\s+([\w]+\.py)", first_line)
        if match:
            name = match.group(1)
            if name not in blocks:
                blocks[name] = block
    return blocks


def _setup_skill_modules() -> str:
    """SKILL.md 블록을 임시 디렉토리에 쓰고 경로 반환."""
    tmp_dir = tempfile.mkdtemp(prefix="ragskill_modules_")
    blocks = _extract_named_blocks(SKILL_MD)
    for filename, code in blocks.items():
        Path(tmp_dir, filename).write_text(code, encoding="utf-8")
    return tmp_dir


# conftest.py 로드(= collection 전) 시점에 즉시 실행
_modules_dir = _setup_skill_modules()
sys.path.insert(0, _modules_dir)

# 프로세스 종료 시 임시 디렉토리 정리
atexit.register(shutil.rmtree, _modules_dir, True)


# conftest.py에서 정의하는 fixtures (tests/conftest.py로 이동하지 않고
# 루트에 두면 tests/unit/까지 자동 적용됨)
@pytest.fixture
def mock_voyage():
    from unittest.mock import MagicMock, patch
    with patch("voyageai.Client") as mock:
        client = MagicMock()
        client.embed.return_value = MagicMock(embeddings=[[0.1] * 1024])
        client.rerank.return_value = MagicMock(
            results=[MagicMock(index=0, relevance_score=0.9)]
        )
        client.embed_chunks.return_value = MagicMock(embeddings=[[[0.1] * 1024]])
        mock.return_value = client
        yield client


@pytest.fixture
def mock_llm():
    from unittest.mock import patch
    with patch("llm.llm") as mock:
        mock.return_value = '{"verdict":"CORRECT","confidence":0.9,"reason":"relevant"}'
        yield mock


@pytest.fixture
def mock_pool():
    """ConnectionPool 클래스 mock을 yield.
    mock_pool          = ConnectionPool (class mock)
    mock_pool()        = pool instance (ConnectionPool(...) 반환값)
    mock_pool().connection() = conn context manager
    """
    from unittest.mock import MagicMock, patch
    with patch("storage.ConnectionPool") as mock_cls:
        pool = MagicMock()
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = [
            {
                "id": "1",
                "content": "test content",
                "context": "test context",
                "summary": "test summary",
                "metadata": {"source": "test.pdf"},
                "rrf_score": 0.9,
            }
        ]
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        conn.cursor.return_value = cursor
        pool.connection.return_value = conn
        mock_cls.return_value = pool
        yield mock_cls  # 클래스 mock 자체를 yield — pool은 mock_cls.return_value


@pytest.fixture
def sample_chunks():
    from chunking import Chunk
    return [
        Chunk(
            content="Exercise improves cardiovascular health.",
            metadata={"source": "test.pdf", "chunk_index": 0},
            context="From a sports science paper about exercise.",
            summary="Exercise benefits cardiovascular system.",
            embedding=[0.1] * 1024,
        ),
        Chunk(
            content="Mental toughness helps athletes perform under pressure.",
            metadata={"source": "mental.pdf", "chunk_index": 0},
            context="From a sports psychology paper.",
            summary="Mental toughness and performance.",
            embedding=[0.2] * 1024,
        ),
    ]


@pytest.fixture
def sample_chunk_dicts():
    return [
        {
            "id": "1",
            "content": "Exercise improves cardiovascular health and VO2max.",
            "context": "From a sports physiology paper on aerobic capacity.",
            "summary": "Exercise cardiovascular benefits.",
            "metadata": {"source": "sports.pdf"},
            "rrf_score": 0.9,
        },
        {
            "id": "2",
            "content": "Mental toughness is a key predictor of athletic success.",
            "context": "From a sports psychology study.",
            "summary": "Mental toughness and athletic performance.",
            "metadata": {"source": "mental.pdf"},
            "rrf_score": 0.7,
        },
    ]
