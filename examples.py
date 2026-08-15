"""Real usage examples, lifted from the repository's own tests.

An explanation tells you what a function is for. An example shows you what
calling it looks like, which is often what the reader actually wanted.

Tests are the best source for this: someone on the project wrote them, CI runs
them, and they use the real API rather than an imagined one. Phase 1 filters
tests out on purpose, because assertions and fixtures would pollute the call
graph and the ranking. Here they are read back in, for examples only.

Nothing is generated, so nothing needs verifying. Quoted code is quoted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from analyze import analyze_files
from ingest import SKIP_DIRS, is_test_path

# A test long enough to be a wall of setup stops being an example.
MAX_EXAMPLE_LINES = 28
MAX_EXAMPLES = 2


@dataclass
class Example:
    name: str
    file: str
    start_line: int
    source: str
    uses: str  # the symbol this example was found for

    @property
    def line_count(self) -> int:
        return self.source.count("\n") + 1


def test_files(root: Path) -> list[Path]:
    """Exactly the files Phase 1 throws away for being tests."""
    found = []
    for path in root.rglob("*.py"):
        rel_parts = path.relative_to(root).parts
        if any(p in SKIP_DIRS for p in rel_parts):
            continue
        if is_test_path(path, root):
            found.append(path)
    return sorted(found)


def _score(sym, name: str) -> tuple:
    """Rank candidate tests. Lower sorts first."""
    src = sym.source
    calls_it = bool(re.search(rf"\b{re.escape(name)}\s*\(", src))
    # A test that calls the thing beats one that merely mentions it, and a
    # short test beats a long one.
    return (not calls_it, sym.line_count, sym.qualname)


def find_examples(root: Path, name: str, limit: int = MAX_EXAMPLES,
                  cache: dict | None = None) -> list[Example]:
    """Tests that exercise `name`, shortest and most direct first."""
    short = name.split(".")[-1]
    if not short or len(short) < 3:
        return []

    key = str(root)
    if cache is not None and key in cache:
        symbols = cache[key]
    else:
        files = test_files(root)
        symbols = analyze_files(root, files).symbols if files else {}
        if cache is not None:
            cache[key] = symbols

    pattern = re.compile(rf"\b{re.escape(short)}\b")
    hits = [
        s for s in symbols.values()
        if s.kind in ("function", "method")
        and s.name.startswith("test")
        and s.line_count <= MAX_EXAMPLE_LINES
        and pattern.search(s.source)
    ]
    hits.sort(key=lambda s: _score(s, short))

    return [
        Example(name=s.name, file=s.file, start_line=s.start_line,
                source=s.source.strip(), uses=short)
        for s in hits[:limit]
    ]


def render(examples: list[Example]) -> str:
    """Markdown block. Empty when the repo has no test covering the symbol."""
    if not examples:
        return ""
    out = ["**How the project's own tests use it**", ""]
    for e in examples:
        out += [f"From `{e.file}:{e.start_line}`:", "",
                "```python", e.source, "```", ""]
    return "\n".join(out).rstrip()


if __name__ == "__main__":
    import sys

    root = Path(sys.argv[1]).resolve()
    target = sys.argv[2]
    found = find_examples(root, target)
    print(render(found) or f"No test in {root.name} exercises {target!r}.")
