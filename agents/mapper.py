"""Phase 5 — the Mapper agent: decide what matters, and what each piece does.

Division of labour, on purpose:

  ranking    -> pure call-graph arithmetic (deterministic, free, testable)
  explaining -> the LLM, one symbol at a time, shown only real source

The LLM never chooses what is important and never invents a relationship. Every
claim it makes is anchored to a qualname + file:line that Phase 2 proved exists,
which is what makes Phase 7's Critic able to check its work.

Run:
    python -m agents.mapper https://github.com/pallets/flask
    python -m agents.mapper https://github.com/pallets/flask --top 12
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyze import Analysis, Symbol, analyze  # noqa: E402
from ingest import SKIP_DIRS  # noqa: E402
from llm import CODE_MODEL, STATS, chat_json  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "out"

# Source sent per symbol. Enough for real context, small enough that prompt
# processing stays quick and we never approach the 8192-token window.
MAX_SOURCE_CHARS = 2200

SYSTEM = (
    "You are a precise code analyst. You describe only what the provided source "
    "code actually does. You never guess at behaviour that is not visible in the "
    "code shown. If the code's purpose is genuinely unclear, say so plainly."
)

PURPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "purpose": {"type": "string"},
        "inputs": {"type": "string"},
        "returns": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["purpose", "inputs", "returns", "confidence"],
}

COMPONENT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["title", "summary"],
}


@dataclass
class SymbolNote:
    qualname: str
    file: str
    start_line: int
    end_line: int
    kind: str
    role: str
    score: float
    fan_in: int
    fan_out: int
    purpose: str = ""
    inputs: str = ""
    returns: str = ""
    confidence: str = ""
    calls: list[str] = field(default_factory=list)


@dataclass
class Component:
    module: str
    title: str
    summary: str
    symbols: list[str]


@dataclass
class RepoMap:
    repo: str
    root: str
    model: str
    total_symbols: int
    total_edges: int
    entry_points: list[str]
    components: list[Component]
    notes: list[SymbolNote]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def load(path: Path) -> RepoMap:
        raw = json.loads(path.read_text(encoding="utf8"))
        raw["components"] = [Component(**c) for c in raw["components"]]
        raw["notes"] = [SymbolNote(**n) for n in raw["notes"]]
        return RepoMap(**raw)


def public_api_names(analysis: Analysis) -> set[str]:
    """Names a package re-exports from its `__init__.py`.

    This matters more than it looks. Call-graph fan-in ranks *internal* plumbing
    highest, because a repo's real public API (`Flask`, `Blueprint`, `route`) is
    called by users, not by the repo itself — so it scores 0 on fan-in. The
    export list is the authors stating outright what newcomers are meant to use.
    """
    names: set[str] = set()
    init_files = {
        s.file for s in analysis.symbols.values() if s.file.endswith("__init__.py")
    }
    # Symbols live in modules, but __init__.py often has no symbols at all, so
    # look for the files directly rather than through the symbol table.
    # Reuse Phase 1's skip list, or `examples/tutorial/__init__.py` leaks demo
    # names like `blog` and `auth` into what we call the public API.
    for path in analysis.root.rglob("__init__.py"):
        parts = path.relative_to(analysis.root).parts
        if any(p in SKIP_DIRS or p in ("test", "tests") for p in parts):
            continue
        init_files.add(path.relative_to(analysis.root).as_posix())

    for rel in init_files:
        try:
            text = (analysis.root / rel).read_text(encoding="utf8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            # `from .app import Flask as Flask` — the redundant alias is the
            # convention for "this is deliberately re-exported".
            if line.startswith(("from .", "from ..")) and " import " in line:
                for part in line.split(" import ", 1)[1].split(","):
                    part = part.strip().strip("()")
                    if not part:
                        continue
                    names.add(part.split(" as ")[-1].strip())
            elif line.startswith("__all__"):
                for token in line.split("=", 1)[-1].replace("[", " ").replace("]", " ").split(","):
                    token = token.strip().strip("\"'")
                    if token:
                        names.add(token)
    return {n for n in names if n and n.isidentifier()}


def score_symbol(sym: Symbol, analysis: Analysis, exports: set[str] | None = None) -> float:
    """Importance from graph shape alone — no LLM, so it is cheap and repeatable.

    Fan-in is weighted highest: something many places depend on is load-bearing.
    Fan-out matters too, because orchestrators (`wsgi_app`) call a lot and are
    exactly where a newcomer should start reading.
    """
    # Capped on purpose: 1 caller vs 5 is a real difference, 9 vs 10 is not.
    # Uncapped, a much-used internal hook outranks the entire public API.
    fan_in, fan_out = analysis.fan(sym.qualname)
    score = 3.0 * min(fan_in, 5) + 1.5 * min(fan_out, 6)

    if sym.docstring:
        score += 2.0  # the authors thought it worth explaining
    if sym.kind == "class":
        score += 1.5  # classes organise everything under them

    if exports:
        if sym.name in exports:
            score += 9.0  # declared public API — what a newcomer actually touches
        # A public method on an exported class (Flask.route) is public too.
        owner = sym.owner_class.split(".")[-1] if sym.owner_class else None
        if owner in exports and not sym.name.startswith("_"):
            score += 5.0

    if sym.name.startswith("_") and not sym.name.startswith("__"):
        score -= 4.0  # private helper — deliberately not part of the story
    if sym.name.startswith("__") and sym.name != "__init__":
        score -= 1.0  # dunder plumbing
    score += min(sym.line_count / 25.0, 2.0)  # substantial, but capped
    return round(score, 2)


def classify_role(sym: Symbol, analysis: Analysis) -> str:
    """A reader-facing label. Order matters — these categories overlap.

    "Nothing in this repo calls it" is checked first because it is the most
    actionable thing a newcomer can be told: start reading here. Something with
    both 0 callers and many callees is an entry point *and* an orchestrator;
    "entry point" is the more useful of the two.
    """
    fan_in, fan_out = analysis.fan(sym.qualname)
    if sym.kind == "class":
        return "type"
    if fan_in == 0 and fan_out > 0:
        return "entry point"
    if fan_out >= 4 and fan_in <= 1:
        return "orchestrator"
    if fan_in >= 3:
        return "shared helper"
    return "supporting"


def rank_symbols(analysis: Analysis, top: int) -> list[SymbolNote]:
    exports = public_api_names(analysis)
    notes = [
        SymbolNote(
            qualname=q,
            file=s.file,
            start_line=s.start_line,
            end_line=s.end_line,
            kind=s.kind,
            role=classify_role(s, analysis),
            score=score_symbol(s, analysis, exports),
            fan_in=len(analysis.callers(q)),
            fan_out=len(analysis.callees(q)),
            calls=analysis.callees(q)[:10],
        )
        for q, s in analysis.symbols.items()
    ]
    # Sort by score, then qualname so runs are reproducible.
    notes.sort(key=lambda n: (-n.score, n.qualname))
    return notes[:top]


def _facts(note: SymbolNote, analysis: Analysis) -> str:
    """Verified call-graph context. Phase 2 proved every line of this."""
    lines = [
        f"Name: {note.qualname}",
        f"Location: {note.file}:{note.start_line}-{note.end_line}",
        f"Kind: {note.kind}",
    ]
    if note.calls:
        lines.append("Calls (verified): " + ", ".join(c.split(".")[-1] for c in note.calls))
    callers = analysis.callers(note.qualname)[:6]
    if callers:
        lines.append("Called by (verified): " + ", ".join(c.split(".")[-1] for c in callers))
    return "\n".join(lines)


def describe_symbol(note: SymbolNote, analysis: Analysis, model: str = CODE_MODEL) -> SymbolNote:
    sym = analysis.symbols[note.qualname]
    source = sym.source
    truncated = len(source) > MAX_SOURCE_CHARS
    if truncated:
        source = source[:MAX_SOURCE_CHARS] + "\n# ... (truncated)"

    prompt = (
        f"{_facts(note, analysis)}\n\n"
        f"Source:\n```python\n{source}\n```\n\n"
        "Describe this code.\n"
        "- purpose: ONE sentence, plain English, no restating the name.\n"
        "- inputs: what it takes in. 'none' if it takes nothing meaningful.\n"
        "- returns: what it gives back. 'nothing' if it returns nothing.\n"
        "- confidence: 'low' if the source was truncated or the intent is unclear."
    )
    data, reply = chat_json(prompt, system=SYSTEM, model=model, schema=PURPOSE_SCHEMA)

    note.purpose = str(data.get("purpose", "")).strip()
    note.inputs = str(data.get("inputs", "")).strip()
    note.returns = str(data.get("returns", "")).strip()
    note.confidence = str(data.get("confidence", "medium")).strip()
    if truncated and note.confidence == "high":
        note.confidence = "medium"  # it did not see everything, so cap the claim
    return note


def describe_component(module: str, notes: list[SymbolNote], model: str = CODE_MODEL) -> Component:
    """Modules are the components. Python repos already group by concern."""
    listing = "\n".join(
        f"- {n.qualname.split('.')[-1]} ({n.role}): {n.purpose}" for n in notes
    )
    prompt = (
        f"Module: {module}\n\nIts most important pieces:\n{listing}\n\n"
        "- title: a short human-friendly name for this module (3-6 words), "
        "e.g. 'Request handling' or 'Session storage'.\n"
        "- summary: 1-2 sentences on what this module is responsible for, "
        "based only on the pieces listed."
    )
    data, _ = chat_json(prompt, system=SYSTEM, model=model, schema=COMPONENT_SCHEMA)
    return Component(
        module=module,
        title=str(data.get("title", module)).strip(),
        summary=str(data.get("summary", "")).strip(),
        symbols=[n.qualname for n in notes],
    )


def build_map(analysis: Analysis, top: int = 20, model: str = CODE_MODEL) -> RepoMap:
    notes = rank_symbols(analysis, top)
    print(f"[mapper] {len(analysis.symbols)} symbols -> describing top {len(notes)}")

    # All symbol calls first, then all component calls: same model throughout, so
    # Ollama never swaps. See the model-swap note in llm.py.
    for i, note in enumerate(notes, 1):
        short = note.qualname.split(".")[-1]
        print(f"[mapper] {i}/{len(notes)} {short} ({note.role}, score {note.score})")
        try:
            describe_symbol(note, analysis, model)
        except Exception as exc:  # one bad reply must not lose the whole run
            note.purpose = f"(failed: {exc})"
            note.confidence = "low"

    # Group by file: a class's qualname and its module's qualname overlap, so
    # splitting on dots misgroups methods. The file path is unambiguous.
    by_module: dict[str, list[SymbolNote]] = {}
    for n in notes:
        by_module.setdefault(n.file, []).append(n)

    components: list[Component] = []
    for module, group in sorted(by_module.items(), key=lambda kv: -len(kv[1])):
        print(f"[mapper] component: {module} ({len(group)} symbols)")
        try:
            components.append(describe_component(module, group, model))
        except Exception as exc:
            components.append(Component(module, module, f"(failed: {exc})",
                                        [n.qualname for n in group]))

    exports = public_api_names(analysis)
    entry = sorted(
        (q for q in analysis.symbols if not analysis.callers(q) and analysis.callees(q)),
        key=lambda q: -score_symbol(analysis.symbols[q], analysis, exports),
    )[:12]

    return RepoMap(
        repo=analysis.root.name,
        root=str(analysis.root),
        model=model,
        total_symbols=len(analysis.symbols),
        total_edges=len(analysis.edges),
        entry_points=entry,
        components=components,
        notes=notes,
    )


def _print_map(m: RepoMap) -> None:
    print(f"\n{'=' * 66}\nREPO MAP: {m.repo}")
    print(f"{m.total_symbols} symbols, {m.total_edges} edges, model {m.model}")

    print(f"\n--- {len(m.components)} components ---")
    for c in m.components:
        print(f"\n  {c.title}  [{c.module}]")
        print(f"    {c.summary}")

    print(f"\n--- top {len(m.notes)} symbols ---")
    for n in m.notes:
        flag = "  <-- low confidence" if n.confidence == "low" else ""
        print(f"\n  {n.qualname}")
        print(f"    {n.file}:{n.start_line}  {n.role}, score {n.score} "
              f"(in {n.fan_in} / out {n.fan_out}){flag}")
        print(f"    {n.purpose}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 5 — build a repo map")
    ap.add_argument("url")
    ap.add_argument("--top", type=int, default=20, help="symbols to describe")
    ap.add_argument("--model", default=CODE_MODEL)
    args = ap.parse_args()

    repo_map = build_map(analyze(args.url), top=args.top, model=args.model)
    _print_map(repo_map)

    OUT_DIR.mkdir(exist_ok=True)
    dest = OUT_DIR / f"{repo_map.repo}.map.json"
    dest.write_text(repo_map.to_json(), encoding="utf8")
    print(f"\n[mapper] saved {dest}")
    print(STATS.summary())


if __name__ == "__main__":
    main()
