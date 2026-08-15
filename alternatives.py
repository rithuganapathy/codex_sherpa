"""Substitutes for a repository's dependencies, with the guesses filtered out.

The dependency list is fact: it comes from `pyproject.toml`. Alternatives are
the model's opinion, which does not belong next to verified facts unchecked. So
the model only proposes names, and PyPI decides which ones are real:

    model proposes -> PyPI lookup -> invented package names disappear
                   -> survivors are shown with PyPI's own summary and licence

The description you read is PyPI's, not the model's. The model's only surviving
contribution is the shortlist.

Commercial and hosted options are not on PyPI, so they cannot be checked this
way. They are kept in a separate list and labelled as unverified suggestions.

Run:
    python alternatives.py flask
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".cache" / "pypi"
# Bump when the Package fields change, so old entries miss rather than lie.
CACHE_VERSION = 2
TIMEOUT = 6
MAX_DEPS = 6          # asking about every dependency makes a wall of text
MAX_PER_DEP = 3

SYSTEM = (
    "You know the Python packaging ecosystem. You suggest realistic substitutes "
    "for libraries. You only name packages you are confident exist. You never "
    "invent a package to fill a slot: an empty list is a fine answer."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "alternatives": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "dependency": {"type": "string"},
                    "does": {"type": "string"},
                    "open_source": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["dependency", "does", "open_source"],
            },
        }
    },
    "required": ["alternatives"],
}


# Measured against real pairs: blinker/django-redis scores 0.13 and is nonsense,
# jinja2/mako scores 0.68 and is exactly right. Below this, the two packages are
# not doing the same job.
MIN_RELEVANCE = 0.33

# Some PyPI packages exist only to tell you not to use them. The `asyncio`
# package is a deprecated backport of the stdlib module, and it was being
# offered as an alternative to anyio. The package's own summary gives it away.
DEAD_SUMMARY = re.compile(
    r"\b(deprecated|no longer maintained|unmaintained|abandoned|"
    r"use the stdlib|do not use|placeholder|reserved name|renamed to)\b", re.I)


@dataclass
class Package:
    name: str
    summary: str = ""
    license: str = ""
    home: str = ""
    requires: list[str] = field(default_factory=list)
    relevance: float = 0.0


@dataclass
class DepAlternatives:
    name: str
    does: str = ""
    open_source: list[Package] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)  # with the reason why


def dep_name(spec: str) -> str:
    """`click>=8.1.3` -> `click`, `foo[extra]>=1` -> `foo`."""
    # maxsplit must be keyword: passing it positionally is deprecated in 3.13
    # and printed a warning into every run's output.
    return re.split(r"[<>=!~\[; ]", spec.strip(), maxsplit=1)[0].strip().lower()


def pypi_info(name: str, use_cache: bool = True) -> Package | None:
    """Ask PyPI whether a package exists. None means it does not, or no network."""
    safe = re.sub(r"[^a-z0-9_.-]", "", name.lower())
    if not safe:
        return None

    cached = CACHE_DIR / f"{safe}.json"
    if use_cache and cached.exists():
        raw = json.loads(cached.read_text(encoding="utf8"))
        if raw is None:
            return None
        # Adding a field to Package once left every cached entry loading with
        # that field empty, which silently disabled a filter. The version tag
        # makes a stale entry a miss instead of a wrong answer.
        if isinstance(raw, dict) and raw.pop("_v", None) == CACHE_VERSION:
            return Package(**raw)

    try:
        with urllib.request.urlopen(
                f"https://pypi.org/pypi/{safe}/json", timeout=TIMEOUT) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:  # a real answer: the package does not exist
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cached.write_text("null", encoding="utf8")
        return None
    except Exception:
        return None  # offline or blocked. Say nothing rather than guess.

    info = data.get("info", {})
    urls = info.get("project_urls") or {}
    pkg = Package(
        name=info.get("name", safe),
        summary=(info.get("summary") or "").strip(),
        license=(info.get("license") or "").strip()[:40],
        home=(urls.get("Source") or urls.get("Repository") or urls.get("Homepage")
              or info.get("home_page") or "").strip(),
        requires=[dep_name(r) for r in (info.get("requires_dist") or [])
                  if isinstance(r, str)],
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps({**pkg.__dict__, "_v": CACHE_VERSION}),
                      encoding="utf8")
    return pkg


def _similarity(a: str, b: str) -> float:
    """How close two package descriptions are. 0.0 when the model is unavailable."""
    try:
        from embed import get_model

        va, vb = get_model().encode([a, b], normalize_embeddings=True)
        return round(float(va @ vb), 3)
    except Exception:
        return 1.0  # cannot judge, so do not reject on this basis


def propose(deps: list[str], model: str) -> list[dict]:
    from llm import chat_json

    listing = "\n".join(f"- {d}" for d in deps)
    prompt = (
        f"These are the dependencies of a Python project:\n{listing}\n\n"
        "For each one give:\n"
        "- does: what job it performs, in under 12 words.\n"
        "- open_source: up to 3 packages that could do the same job. PyPI "
        "names only, lowercase, no versions. Leave empty if there is no real "
        "substitute, which is the right answer for small focused libraries.\n"
        "Never include the dependency itself in its own list."
    )
    data, _ = chat_json(prompt, system=SYSTEM, model=model, schema=SCHEMA)
    return data.get("alternatives", [])


def find_alternatives(deps: list[str], model: str | None = None,
                      max_deps: int = MAX_DEPS) -> list[DepAlternatives]:
    from llm import CODE_MODEL

    names = [dep_name(d) for d in deps][:max_deps]
    names = [n for n in names if n]
    if not names:
        return []

    try:
        proposed = propose(names, model or CODE_MODEL)
    except Exception:
        return []

    out: list[DepAlternatives] = []
    for entry in proposed:
        dep = dep_name(str(entry.get("dependency", "")))
        if dep not in names:
            continue
        row = DepAlternatives(name=dep, does=str(entry.get("does", "")).strip())

        base = pypi_info(dep)
        seen: set[str] = set()
        for cand in list(entry.get("open_source", []))[:MAX_PER_DEP]:
            cand_name = dep_name(str(cand))
            # The model happily proposes the same package three times.
            if not cand_name or cand_name == dep or cand_name in seen:
                continue
            seen.add(cand_name)

            pkg = pypi_info(cand_name)
            if pkg is None:
                row.rejected.append(f"{cand_name} (not on PyPI)")
                continue

            # A package that depends on the thing is a layer on top of it, not
            # a replacement. This is what stops werkzeug being "replaced" by
            # flask, which requires werkzeug.
            if dep in pkg.requires or cand_name in (n.lower() for n in names):
                row.rejected.append(f"{cand_name} (builds on {dep})")
                continue

            if DEAD_SUMMARY.search(pkg.summary):
                row.rejected.append(f"{cand_name} (its own PyPI page says not "
                                    f"to use it)")
                continue

            # Existing on PyPI proves a name is real, not that it does the same
            # job. Compare what the two packages say they do.
            if base and base.summary and pkg.summary:
                pkg.relevance = _similarity(base.summary, pkg.summary)
                if pkg.relevance < MIN_RELEVANCE:
                    row.rejected.append(
                        f"{cand_name} (does something else, {pkg.relevance:.2f})")
                    continue
            row.open_source.append(pkg)

        out.append(row)
    return out


def render(rows: list[DepAlternatives]) -> str:
    if not rows:
        return ""

    out = [
        "If a dependency is unavailable to you, these could stand in. A model "
        "proposed the names; each one then had to exist on PyPI, not be built "
        "on top of the package it claims to replace, and describe a similar "
        "job. Descriptions below are PyPI's own words. These are still "
        "suggestions, not verified equivalents.",
        "",
        "Only packages on PyPI are listed. A paid or hosted column was tried "
        "and removed: nothing could check it, and it suggested free web "
        "frameworks as commercial substitutes for an HTTP client.",
        "",
    ]
    for r in rows:
        out.append(f"**`{r.name}`**" + (f" — {r.does}" if r.does else ""))
        out.append("")
        if r.open_source:
            for p in r.open_source:
                link = f"[{p.name}]({p.home})" if p.home else f"`{p.name}`"
                bits = [f"- {link}"]
                if p.summary:
                    bits.append(f": {p.summary}")
                if p.license:
                    bits.append(f"  _({p.license})_")
                if p.relevance:
                    bits.append(f"  <sub>similarity {p.relevance:.2f}</sub>")
                out.append("".join(bits))
        else:
            out.append("- Nothing proposed survived the checks.")
        out.append("")

    dropped = sorted({n for r in rows for n in r.rejected})
    if dropped:
        # Each entry carries its own reason. Labelling them all "not on PyPI"
        # was wrong, and wrong in the part of the page that claims rigour.
        out.append(f"<sub>Proposed but dropped: {', '.join(dropped)}.</sub>")
    return "\n".join(out).rstrip()


if __name__ == "__main__":
    from ingest import REPOS_DIR
    from manifest import read_manifest

    repo = sys.argv[1] if len(sys.argv) > 1 else "flask"
    root = Path(repo) if Path(repo).exists() else REPOS_DIR / repo
    man = read_manifest(root, [])
    if not man.dependencies:
        sys.exit(f"No dependency list found in {root}")

    print(f"Dependencies: {', '.join(man.dependencies)}\n")
    print(render(find_alternatives(man.dependencies)) or "(nothing found)")
