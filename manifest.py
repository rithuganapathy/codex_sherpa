"""Getting-started facts, read from the repo's own files.

Phases 1-9 only ever look at `.py` files, so the pipeline never sees the README,
`pyproject.toml`, or `requirements.txt`. That is where install and run
instructions actually live, which is why the generated docs could describe a
codebase perfectly and still not tell you how to start it.

Everything here is extraction, not generation. No model is involved, so there is
nothing to hallucinate and nothing for Phase 7 to verify. Quoted text is quoted,
not paraphrased.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# Enough to show what a project pulls in without pasting a lock file.
MAX_DEPS = 12
MAX_BLOCK_CHARS = 700

README_NAMES = ("README.md", "README.rst", "README.txt", "README",
                "readme.md", "Readme.md")


@dataclass
class Manifest:
    name: str = ""
    summary: str = ""
    python_requires: str = ""
    dependencies: list[str] = field(default_factory=list)
    dep_source: str = ""
    total_deps: int = 0
    console_scripts: dict[str, str] = field(default_factory=dict)
    install_cmds: list[str] = field(default_factory=list)
    usage_block: str = ""
    usage_lang: str = "python"
    readme: str = ""
    runnable_modules: list[str] = field(default_factory=list)
    has_tests: bool = False

    @property
    def found_anything(self) -> bool:
        return bool(self.dependencies or self.install_cmds or self.usage_block
                    or self.console_scripts or self.runnable_modules)


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf8", errors="replace")
    except OSError:
        return ""


def _from_pyproject(root: Path, m: Manifest) -> None:
    path = root / "pyproject.toml"
    if not path.exists():
        return
    try:
        data = tomllib.loads(_read(path))
    except tomllib.TOMLDecodeError:
        return

    project = data.get("project", {})
    m.name = m.name or str(project.get("name", ""))
    m.summary = m.summary or str(project.get("description", ""))
    m.python_requires = m.python_requires or str(project.get("requires-python", ""))

    deps = [d for d in project.get("dependencies", []) if isinstance(d, str)]
    if deps:
        m.total_deps = len(deps)
        m.dependencies = deps[:MAX_DEPS]
        m.dep_source = "pyproject.toml"

    scripts = project.get("scripts", {})
    if isinstance(scripts, dict):
        m.console_scripts = {str(k): str(v) for k, v in scripts.items()}


def _from_requirements(root: Path, m: Manifest) -> None:
    if m.dependencies:
        return  # pyproject is the better source
    path = root / "requirements.txt"
    if not path.exists():
        return
    deps = [ln.strip() for ln in _read(path).splitlines()
            if ln.strip() and not ln.strip().startswith(("#", "-"))]
    if deps:
        m.total_deps = len(deps)
        m.dependencies = deps[:MAX_DEPS]
        m.dep_source = "requirements.txt"


def _code_blocks(text: str) -> list[tuple[str, str]]:
    """(language, body) for every fenced block, in document order."""
    return [(lang.strip().lower() or "text", body.strip())
            for lang, body in re.findall(r"```([^\n`]*)\n(.*?)```", text, re.S)]


def _from_readme(root: Path, m: Manifest) -> None:
    for name in README_NAMES:
        path = root / name
        if path.exists():
            m.readme = name
            break
    else:
        return

    text = _read(path)
    blocks = _code_blocks(text)

    # Install lines: taken verbatim wherever they appear, including inside a
    # larger block, because that is what a reader would copy.
    seen: set[str] = set()
    for _, body in blocks:
        for line in body.splitlines():
            line = line.strip().lstrip("$ ").strip()
            if re.match(r"^(pip|pip3|python -m pip|uv|poetry|pipenv|conda)\s+"
                        r"(install|add|sync)\b", line) and line not in seen:
                seen.add(line)
                m.install_cmds.append(line)
    if not m.install_cmds:
        for line in text.splitlines():
            line = line.strip().strip("`").lstrip("$ ").strip()
            if re.match(r"^pip install\s+\S", line) and line not in seen:
                seen.add(line)
                m.install_cmds.append(line)
    m.install_cmds = m.install_cmds[:4]

    # Usage example: prefer the first Python block that is not just an install.
    for lang, body in blocks:
        if lang in ("python", "py", "pycon") and "pip install" not in body:
            m.usage_block = body[:MAX_BLOCK_CHARS]
            m.usage_lang = "python"
            return
    for lang, body in blocks:
        if "pip install" not in body and len(body) > 20:
            m.usage_block = body[:MAX_BLOCK_CHARS]
            m.usage_lang = lang if lang != "text" else "bash"
            return


def _entry_points(root: Path, m: Manifest, py_files: list[Path]) -> None:
    for path in py_files:
        rel = path.relative_to(root).as_posix()
        if path.name == "__main__.py":
            m.runnable_modules.append(rel)
        elif re.search(r'^if\s+__name__\s*==\s*["\']__main__["\']',
                       _read(path), re.M):
            m.runnable_modules.append(rel)
    m.runnable_modules = sorted(set(m.runnable_modules))[:8]
    m.has_tests = any((root / d).is_dir() for d in ("tests", "test"))


def read_manifest(root: Path, py_files: list[Path] | None = None) -> Manifest:
    m = Manifest()
    _from_pyproject(root, m)
    _from_requirements(root, m)
    _from_readme(root, m)
    _entry_points(root, m, py_files or [])
    m.name = m.name or root.name
    return m


def render(m: Manifest) -> str:
    """Markdown for the Getting started section. Empty if nothing was found."""
    if not m.found_anything:
        return ""

    out: list[str] = [
        "Taken directly from the repository's own files, not written by a "
        "model. Anything quoted below is quoted, not paraphrased."
    ]

    if m.python_requires:
        out += ["", f"**Requires Python** `{m.python_requires}`"]

    if m.install_cmds:
        out += ["", "**Install**", "", "```bash", *m.install_cmds, "```",
                f"<sub>quoted from {m.readme}</sub>"]
    elif m.dependencies or m.name:
        # No install line in the README. Say where the guess comes from rather
        # than presenting it as something the project documented.
        if m.dep_source == "requirements.txt":
            cmd, note = (f"pip install -r {m.dep_source}",
                         f"inferred from {m.dep_source}")
        elif m.name:
            cmd, note = (f"pip install {m.name}",
                         "inferred from the package name in pyproject.toml, "
                         "the README gives no install command")
        else:
            cmd, note = "pip install .", "inferred, for a local checkout"
        out += ["", "**Install**", "", "```bash", cmd, "```", f"<sub>{note}</sub>"]

    if m.dependencies:
        shown = ", ".join(f"`{d}`" for d in m.dependencies)
        more = (f" and {m.total_deps - len(m.dependencies)} more"
                if m.total_deps > len(m.dependencies) else "")
        out += ["", f"**Depends on** ({m.dep_source}): {shown}{more}"]

    if m.console_scripts:
        out += ["", "**Command-line entry points**", ""]
        out += [f"- `{k}` runs `{v}`" for k, v in sorted(m.console_scripts.items())]

    if m.runnable_modules:
        out += ["", "**Files you can run directly**", ""]
        out += [f"- `{f}`" for f in m.runnable_modules]

    if m.usage_block:
        out += ["", f"**Example from {m.readme or 'the README'}**", "",
                f"```{m.usage_lang}", m.usage_block, "```"]

    if m.has_tests:
        out += ["", "The repository ships a test suite, so `pytest` is usually "
                    "the fastest way to confirm your setup works."]
    return "\n".join(out)


if __name__ == "__main__":
    import sys

    from ingest import list_python_files

    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    man = read_manifest(root, list_python_files(root))
    print(render(man) or "(nothing found)")
