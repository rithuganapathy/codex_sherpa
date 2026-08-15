"""What is rough about the repository you are reading.

Not about this tool. The reader came to judge someone else's code, so these are
the things worth knowing before you depend on it: what is undocumented, what
looks unused, what is oversized, whether it is tested.

Every number is counted from the parsed source. Nothing here is a model's
opinion, and nothing that passes cleanly takes up space on the page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from analyze import Analysis

# Dunder methods and tiny helpers are not "undocumented code" in any useful
# sense, and counting them makes every repo look bad.
def _documentable(sym) -> bool:
    return (sym.kind in ("function", "method", "class")
            and not sym.name.startswith("__")
            and sym.line_count >= 5)


@dataclass
class Finding:
    label: str
    detail: str = ""
    good: bool = False  # a clean result, worth one line rather than a section


@dataclass
class RepoLimits:
    findings: list[Finding] = field(default_factory=list)

    def as_markdown(self) -> str:
        return "\n".join(
            f"- **{f.label}**" + (f" — {f.detail}" if f.detail else "")
            for f in self.findings)


def undocumented(analysis: Analysis) -> Finding | None:
    docable = [s for s in analysis.symbols.values() if _documentable(s)]
    if not docable:
        return None
    missing = [s for s in docable if not s.docstring]
    pct = round(100 * len(missing) / len(docable))
    if not missing:
        return Finding("Everything is documented", good=True)
    return Finding(
        f"{len(missing)} of {len(docable)} functions have no docstring",
        f"{pct}% of the code explains itself only through its name")


def unused(analysis: Analysis, exports: set[str]) -> Finding | None:
    """Private helpers that nothing in the repo calls.

    Only private ones. A public method with no in-repo callers is not dead
    code, it is the library's API being called by its users, and counting those
    reported 121 "unused" functions in flask including `open_resource`.

    Still "appears unused" rather than "dead": a call the parser could not
    resolve is invisible here, and plugins get wired up at runtime.
    """
    suspects = [
        q for q, s in analysis.symbols.items()
        if s.kind != "class"
        and s.name.startswith("_")
        and not s.name.startswith("__")
        and s.name not in exports
        and not analysis.callers(q)
        and s.line_count >= 5
    ]
    if not suspects:
        return Finding("No obviously unused functions", good=True)
    names = ", ".join(f"`{q.split('.')[-1]}`" for q in sorted(suspects)[:4])
    more = f" and {len(suspects) - 4} more" if len(suspects) > 4 else ""
    return Finding(f"{len(suspects)} functions appear unused",
                   f"{names}{more}. Nothing in the repo calls them, though a "
                   f"caller outside it would not show here")


def oversized(analysis: Analysis, threshold: int = 80) -> Finding | None:
    big = sorted((s for s in analysis.symbols.values()
                  if s.kind != "class" and s.line_count >= threshold),
                 key=lambda s: -s.line_count)
    if not big:
        return None
    top = big[0]
    extra = f", and {len(big) - 1} others over {threshold} lines" if len(big) > 1 else ""
    return Finding(f"Longest function is `{top.name}` at {top.line_count} lines",
                   f"{top.file}:{top.start_line}{extra}")


def tested(analysis: Analysis) -> Finding | None:
    root = analysis.root
    test_dirs = [d for d in ("tests", "test") if (root / d).is_dir()]
    test_files = []
    for d in test_dirs:
        test_files += list((root / d).rglob("test_*.py"))
    test_files += [p for p in root.glob("test_*.py")]
    if not test_files:
        return Finding("No test suite found",
                       "nothing here can confirm the code still behaves")
    return Finding(f"Has a test suite, {len(test_files)} test files",
                   f"against {len(analysis.symbols)} functions and classes",
                   good=True)


def cycles(analysis: Analysis) -> Finding | None:
    """Files that call into each other both ways."""
    pairs: set[tuple[str, str]] = set()
    for src, dst in analysis.edges:
        a, b = analysis.symbols[src].file, analysis.symbols[dst].file
        if a != b:
            pairs.add((a, b))
    both = {tuple(sorted(p)) for p in pairs if (p[1], p[0]) in pairs}
    if not both:
        return Finding("No files depend on each other in circles", good=True)
    shown = ", ".join(f"`{Path(a).name}` and `{Path(b).name}`"
                      for a, b in sorted(both)[:2])
    more = f", and {len(both) - 2} more" if len(both) > 2 else ""
    return Finding(f"{len(both)} pairs of files import each other both ways",
                   f"{shown}{more}. Circular dependencies make files hard to "
                   f"read on their own")


def typed(analysis: Analysis) -> Finding | None:
    """Rough annotation coverage, read off the def line."""
    funcs = [s for s in analysis.symbols.values()
             if s.kind in ("function", "method") and s.params]
    if not funcs:
        return None
    annotated = sum(1 for s in funcs
                    if ":" in s.source.split("\n")[0].split("(", 1)[-1]
                    or "->" in s.source.split("\n")[0])
    pct = round(100 * annotated / len(funcs))
    if pct >= 80:
        return Finding(f"Type hints on {pct}% of functions", good=True)
    return Finding(f"Type hints on only {pct}% of functions",
                   "editors and type checkers will help you less here")


def repo_limits(analysis: Analysis, exports: set[str] | None = None) -> RepoLimits:
    checks = [
        undocumented(analysis),
        unused(analysis, exports or set()),
        oversized(analysis),
        cycles(analysis),
        tested(analysis),
        typed(analysis),
    ]
    found = [c for c in checks if c is not None]
    # Problems first, clean results after, so the page opens with what matters.
    return RepoLimits(sorted(found, key=lambda f: f.good))


if __name__ == "__main__":
    import sys

    from agents.mapper import public_api_names
    from analyze import analyze_files
    from ingest import REPOS_DIR, list_python_files

    root = REPOS_DIR / (sys.argv[1] if len(sys.argv) > 1 else "flask")
    a = analyze_files(root, list_python_files(root))
    print(repo_limits(a, public_api_names(a)).as_markdown())
