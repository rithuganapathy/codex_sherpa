"""Phase 1 — clone a GitHub repo and list its source files."""

import sys
from pathlib import Path

from git import Repo

REPOS_DIR = Path(__file__).parent / "repos"

SKIP_DIRS = {
    ".git", ".github", ".venv", "venv", "env", "node_modules", "__pycache__",
    "coverage", ".next", ".nuxt", "vendor", "third_party",
    ".pytest_cache", ".mypy_cache", ".tox", "build", "dist", "site-packages",
    ".idea", ".vscode", "docs", "examples",
}

SKIP_FILE_PREFIXES = ("test_", "conftest")
# JavaScript keeps its tests beside the code far more often than Python does.
SKIP_FILE_INFIXES = (".test.", ".spec.", ".min.", ".d.")
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
    if any(part in path.name for part in SKIP_FILE_INFIXES):
        return True
    return path.name.startswith(SKIP_FILE_PREFIXES) or path.name.endswith(SKIP_FILE_SUFFIXES)


def list_source_files(root: Path, include_tests: bool = False,
                      subdir: str = "") -> list[Path]:
    """Every file in a language we can parse, minus the junk.

    Named for what it does now. `list_python_files` stays as an alias because
    the name is used in a few places and the behaviour is unchanged for Python.
    """
    from languages import extensions

    # A repo can hold more than one project. unslothai/unsloth carries a web
    # app in studio/ that is 13x the size of the library, and it would win the
    # ranking and take the documentation with it.
    scope = root / subdir if subdir else root
    if subdir and not scope.is_dir():
        raise FileNotFoundError(
            f"{subdir!r} is not a folder in this repository. "
            f"Top level holds: "
            + ", ".join(sorted(p.name for p in root.iterdir()
                               if p.is_dir() and not p.name.startswith("."))[:8]))

    found = []
    candidates: list[Path] = []
    for ext in extensions():
        candidates.extend(scope.rglob(f"*{ext}"))
    for path in candidates:
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if not include_tests and is_test_path(path, root):
            continue
        found.append(path)
    return sorted(found)


list_python_files = list_source_files


def ingest(url: str, include_tests: bool = False,
           subdir: str = "") -> tuple[Path, list[Path]]:
    root = clone_repo(url)
    return root, list_source_files(root, include_tests, subdir)


def suggest_subdirs(root: Path, limit: int = 6) -> list[tuple[str, int]]:
    """Top-level folders holding source, largest first.

    Used to tell someone which parts of a repo they could scope to, with the
    file counts that make the choice obvious.
    """
    counts: dict[str, int] = {}
    for path in list_source_files(root):
        parts = path.relative_to(root).parts
        key = parts[0] if len(parts) > 1 else "(root)"
        counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])[:limit]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python ingest.py <github-repo-url>")
    repo_root, files = ingest(sys.argv[1])
    for f in files:
        print(f.relative_to(repo_root).as_posix())
    print(f"\n[ingest] {len(files)} python files under {repo_root}")
