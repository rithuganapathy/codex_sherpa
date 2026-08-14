"""Phase 1 — clone a GitHub repo and list its source files."""

import sys
from pathlib import Path

from git import Repo

REPOS_DIR = Path(__file__).parent / "repos"

SKIP_DIRS = {
    ".git", ".github", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".tox", "build", "dist", "site-packages",
    ".idea", ".vscode", "docs", "examples",
}

SKIP_FILE_PREFIXES = ("test_", "conftest")
SKIP_FILE_SUFFIXES = ("_test.py", "setup.py")


def clone_repo(url: str) -> Path:
    name = url.rstrip("/").removesuffix(".git").split("/")[-1]
    dest = REPOS_DIR / name
    if dest.exists():
        print(f"[ingest] reusing existing clone at {dest}")
        return dest
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[ingest] cloning {url} -> {dest}")
    Repo.clone_from(url, dest, depth=1)
    return dest


def is_test_path(path: Path, root: Path) -> bool:
    rel_parts = path.relative_to(root).parts
    if any(p in ("test", "tests") for p in rel_parts[:-1]):
        return True
    return path.name.startswith(SKIP_FILE_PREFIXES) or path.name.endswith(SKIP_FILE_SUFFIXES)


def list_python_files(root: Path, include_tests: bool = False) -> list[Path]:
    found = []
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if not include_tests and is_test_path(path, root):
            continue
        found.append(path)
    return sorted(found)


def ingest(url: str, include_tests: bool = False) -> tuple[Path, list[Path]]:
    root = clone_repo(url)
    return root, list_python_files(root, include_tests)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python ingest.py <github-repo-url>")
    repo_root, files = ingest(sys.argv[1])
    for f in files:
        print(f.relative_to(repo_root).as_posix())
    print(f"\n[ingest] {len(files)} python files under {repo_root}")
