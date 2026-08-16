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

import json
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
    runtime_requires: str = ""   # "Node >=18" for JavaScript
    run_scripts: dict[str, str] = field(default_factory=dict)
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
    workspace_note: str = ""
    is_node: bool = False

    @property
    def found_anything(self) -> bool:
        return bool(self.dependencies or self.install_cmds or self.usage_block
                    or self.console_scripts or self.runnable_modules
                    or self.run_scripts)


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


def _from_package_json(root: Path, m: Manifest) -> None:
    """The JavaScript equivalent of pyproject.toml.

    Checked before requirements.txt so a mixed repo prefers the manifest that
    actually describes it.
    """
    path = root / "package.json"
    if not path.exists():
        return
    try:
        data = json.loads(_read(path))
    except json.JSONDecodeError:
        return

    m.is_node = True
    m.name = m.name or str(data.get("name", ""))
    m.summary = m.summary or str(data.get("description", ""))
    engines = data.get("engines") or {}
    if isinstance(engines, dict) and engines.get("node"):
        m.runtime_requires = f"Node {engines['node']}"

    deps = data.get("dependencies") or {}
    source = "package.json"

    # A workspace monorepo keeps nothing at the root but scripts. cordis is one:
    # the dependencies that describe the project live in packages/core. Take the
    # largest member rather than reporting a project with no dependencies.
    if not deps and data.get("workspaces"):
        best, best_path = {}, ""
        for member in sorted(root.glob("packages/*/package.json")):
            try:
                sub = json.loads(_read(member))
            except json.JSONDecodeError:
                continue
            sub_deps = sub.get("dependencies") or {}
            if len(sub_deps) > len(best):
                best = sub_deps
                best_path = member.relative_to(root).as_posix()
        if best:
            deps, source = best, best_path
            m.workspace_note = (f"A workspace monorepo. The dependencies below "
                                f"are {best_path}, its largest package.")

    if isinstance(deps, dict) and deps:
        pairs = [f"{k}@{v}" for k, v in deps.items()]
        m.total_deps = len(pairs)
        m.dependencies = pairs[:MAX_DEPS]
        m.dep_source = source

    scripts = data.get("scripts") or {}
    if isinstance(scripts, dict):
        # Only the ones a newcomer would actually run first.
        for key in ("start", "dev", "build", "test"):
            if scripts.get(key):
                m.run_scripts[key] = str(scripts[key])

    bins = data.get("bin")
    if isinstance(bins, str) and m.name:
        m.console_scripts[m.name] = bins
    elif isinstance(bins, dict):
        m.console_scripts.update({str(k): str(v) for k, v in bins.items()})


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
    _from_package_json(root, m)
    _from_requirements(root, m)
    _from_readme(root, m)
    _entry_points(root, m, py_files or [])
    m.name = m.name or root.name
    return m


def render(m: Manifest) -> str:
    """The Getting started section, as steps in the order you would do them.

    A list of loose facts (version, deps, entry points) left the reader to
    assemble the order themselves. These are the same facts, arranged as the
    path from nothing installed to something running.
    """
    if not m.found_anything:
        return ""

    step = 0

    def head(title: str, note: str = "") -> list[str]:
        nonlocal step
        step += 1
        line = f"### {step}. {title}"
        return ["", line, ""] + ([f"<sub>{note}</sub>", ""] if note else [])

    out: list[str] = [
        "Taken directly from the repository's own files, not written by a "
        "model. Anything quoted below is quoted, not paraphrased."
    ]

    # 1 — check you can run it at all
    if m.python_requires:
        out += head("Check your Python version")
        out += [f"This project needs Python `{m.python_requires}`.", "",
                "```bash", "python --version", "```"]
    elif m.runtime_requires:
        out += head("Check your runtime version")
        out += [f"This project needs `{m.runtime_requires}`.", "",
                "```bash", "node --version", "```"]

    # 2 — install
    if m.install_cmds:
        out += head("Install it", f"quoted from {m.readme}")
        out += ["```bash", *m.install_cmds, "```"]
    elif m.dependencies or m.name:
        if m.is_node:
            cmd, note = ("npm install",
                         "inferred from package.json, the README gives no "
                         "install command")
        elif m.dep_source == "requirements.txt":
            cmd, note = (f"pip install -r {m.dep_source}",
                         f"inferred from {m.dep_source}")
        elif m.name:
            cmd, note = (f"pip install {m.name}",
                         "inferred from the package name in pyproject.toml, "
                         "the README gives no install command")
        else:
            cmd, note = "pip install .", "inferred, for a local checkout"
        out += head("Install it", note)
        out += ["```bash", cmd, "```"]

    if m.workspace_note:
        out += ["", m.workspace_note]

    if m.dependencies:
        shown = ", ".join(f"`{d}`" for d in m.dependencies)
        more = (f", and {m.total_deps - len(m.dependencies)} more"
                if m.total_deps > len(m.dependencies) else "")
        out += ["", f"That pulls in {shown}{more}. Names you meet in the source "
                    f"that are not in the list above usually come from these.",
                f"<sub>from {m.dep_source}</sub>"]

    # 3 — the smallest thing that works
    if m.usage_block:
        out += head("Run the smallest thing that works",
                    f"quoted from {m.readme or 'the README'}")
        out += [f"```{m.usage_lang}", m.usage_block, "```"]

    # 4 — the ways in
    if m.run_scripts:
        out += head("Run it", "from the scripts block in package.json")
        out += ["```bash"] + [f"npm run {k}" for k in m.run_scripts] + ["```"]
        out += ["", "Those map to: " + ", ".join(
            f"`{k}` = `{v[:60]}`" for k, v in m.run_scripts.items())]

    if m.console_scripts or m.runnable_modules:
        out += head("Know the ways in")
        if m.console_scripts:
            out += [f"- `{k}` on your command line runs `{v}`"
                    for k, v in sorted(m.console_scripts.items())]
        if m.runnable_modules:
            out += [f"- `{f}` can be run directly" for f in m.runnable_modules]

    # 5 — prove the setup
    if m.has_tests:
        out += head("Check your setup is sound")
        out += ["The project ships tests, which is the quickest way to find out "
                "whether it works on your machine before you blame your own "
                "code.", "", "```bash", "pytest", "```"]
    return "\n".join(out)


if __name__ == "__main__":
    import sys

    from ingest import list_python_files

    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    man = read_manifest(root, list_python_files(root))
    print(render(man) or "(nothing found)")
