"""
SKILL.md Python 코드 블록 유효성 검증기.

동작:
1. SKILL.md에서 Named Python 블록 추출 (# filename.py 로 시작하는 블록)
2. py_compile로 구문 오류 탐지
3. 각 블록에 필수 import/패턴 존재 여부 검사
4. 결과 출력 + 오류 시 exit(1)
"""

import os
import re
import sys
import py_compile
import tempfile
from pathlib import Path

# SKILL.md 경로: 이 파일 기준 ../skill/SKILL.md
SKILL_MD = Path(__file__).parent.parent / "skill" / "SKILL.md"

# 각 Named Block에서 반드시 존재해야 할 패턴
REQUIRED_PATTERNS: dict[str, list[str]] = {
    "pipeline.py": [
        "from embedding import embed_query",
        "from reranker import rerank",
        "from crag import evaluate_retrieval",
        "from storage import ChunkStore",
    ],
    "embedding.py": [
        "from chunking import Chunk",
    ],
    "storage.py": [
        "from chunking import Chunk",
    ],
    "config.py": [
        'voyage_api_key: str = ""',
    ],
    "crag.py": [
        "class Verdict",
    ],
}


class SkillValidator:
    def __init__(self, skill_path: Path):
        self.skill_path = skill_path
        self.blocks: dict[str, str] = {}
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def extract_blocks(self) -> None:
        """SKILL.md에서 Named Python 블록 추출.

        Named Block: 첫 줄이 '# filename.py' 형태인 블록
        Example Block: 설명용 (독립 실행 불가) → 스킵
        """
        text = self.skill_path.read_text(encoding="utf-8")
        raw_blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)

        for block in raw_blocks:
            first_line = block.split("\n")[0].strip()
            # "# config.py" 또는 "# config.py — 설명" 형태 매칭
            match = re.match(r"^#\s+([\w]+\.py)", first_line)
            if match:
                name = match.group(1)
                # 중복 블록은 첫 번째 우선
                if name not in self.blocks:
                    self.blocks[name] = block

    def check_syntax(self) -> None:
        """py_compile로 각 Named Block 구문 오류 탐지."""
        for name, code in self.blocks.items():
            with tempfile.NamedTemporaryFile(
                suffix=".py", mode="w", delete=False, encoding="utf-8"
            ) as f:
                f.write(code)
                fname = f.name
            try:
                py_compile.compile(fname, doraise=True)
            except py_compile.PyCompileError as e:
                # 오류 메시지에서 임시파일 경로 제거
                msg = str(e).replace(fname, name)
                self.errors.append(f"[SYNTAX] {name}: {msg}")
            finally:
                os.unlink(fname)

    def check_imports(self) -> None:
        """필수 패턴이 해당 Named Block에 존재하는지 검사."""
        for filename, patterns in REQUIRED_PATTERNS.items():
            if filename not in self.blocks:
                self.warnings.append(f"[WARN] {filename}: Named block not found in SKILL.md")
                continue
            code = self.blocks[filename]
            for pattern in patterns:
                if pattern not in code:
                    self.errors.append(
                        f"[IMPORT] {filename}: missing required pattern: {pattern!r}"
                    )

    def run(self) -> bool:
        """전체 검증 실행. True = 통과, False = 오류 있음."""
        if not self.skill_path.exists():
            print(f"ERROR: SKILL.md not found at {self.skill_path}", file=sys.stderr)
            return False

        self.extract_blocks()

        if not self.blocks:
            self.errors.append("No Named Python blocks found in SKILL.md")
        else:
            print(f"Found {len(self.blocks)} named blocks: {', '.join(sorted(self.blocks))}")

        self.check_syntax()
        self.check_imports()

        if self.warnings:
            for w in self.warnings:
                print(w)

        if self.errors:
            for e in self.errors:
                print(e, file=sys.stderr)
            print(f"\nResult: {len(self.errors)} error(s), {len(self.warnings)} warning(s)")
            return False

        print(f"\nResult: 0 errors, {len(self.warnings)} warning(s) — PASS")
        return True


if __name__ == "__main__":
    validator = SkillValidator(SKILL_MD)
    ok = validator.run()
    sys.exit(0 if ok else 1)
