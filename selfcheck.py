"""Regression checks for the Phase 2 parser. Run: python selfcheck.py

No pytest dependency on purpose — this needs to stay runnable in a bare venv.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from analyze import analyze_files

SAMPLE = '''
import functools


def helper(x):
    return x + 1


def outer():
    def inner():
        return helper(1)
    return inner()


class Base:
    """Base docstring."""

    def shared(self):
        return helper(2)


class Child(Base):
    @functools.cache
    def uses_inherited(self):
        return self.shared()

    def uses_super(self):
        return super().shared()

    def local_dict(self):
        opts = {}
        return opts.setdefault("a", helper(3))
'''

EXPECTED_CALLS = {
    "sample.helper": [],
    "sample.outer": ["sample.outer.inner"],
    "sample.outer.inner": ["sample.helper"],
    "sample.Base.shared": ["sample.helper"],
    # self.shared() and super().shared() both reach the inherited method
    "sample.Child.uses_inherited": ["sample.Base.shared"],
    "sample.Child.uses_super": ["sample.Base.shared"],
    # opts.setdefault() is a dict method, not an edge — but helper(3) nested
    # inside its arguments still counts
    "sample.Child.local_dict": ["sample.helper"],
}


def check_chunking(a, check) -> None:
    """Phase 3 chunking — pure logic, no model load or network."""
    from embed import MAX_CHUNK_CHARS, _split_long, build_chunks

    check("short source stays one chunk", len(_split_long("a\nb\nc")), 1)

    long_src = "\n".join(f"    line_{i} = compute({i})" for i in range(200))
    windows = _split_long(long_src)
    check("long source is split", len(windows) > 1, True)
    check("every window fits the limit",
          all(len(t) <= MAX_CHUNK_CHARS for t, _ in windows), True)
    check("windows start at line 0", windows[0][1], 0)
    check("windows advance monotonically",
          all(b[1] > a_[1] for a_, b in zip(windows, windows[1:])), True)
    # Overlap means consecutive windows must share lines, so no logic falls in a gap.
    first_lines = windows[0][0].splitlines()
    check("consecutive windows overlap",
          any(l in windows[1][0].splitlines() for l in first_lines[-3:]), True)

    chunks = build_chunks(a)
    check("every symbol produces >=1 chunk",
          {c.qualname for c in chunks}, set(a.symbols))
    check("chunk ids are unique",
          len({c.chunk_id for c in chunks}), len(chunks))
    check("chunk text carries its qualname",
          all(c.qualname in c.text for c in chunks), True)
    # A class body would duplicate all its methods; only the signature is kept.
    cls = [c for c in chunks if c.kind == "class"]
    check("class chunks stay small", all(len(c.text) < 800 for c in cls), True)
    check("class chunk excludes method bodies",
          any("return helper" in c.text for c in cls), False)


def check_json_salvage(check) -> None:
    """Phase 4 JSON recovery — offline, no Ollama needed.

    Small models wrap JSON in prose or code fences even when told not to. Phases
    5 and 7 parse every reply, so this fallback has to hold.
    """
    import json as _json

    from llm import _extract_json

    cases = [
        ('{"a": 1}', {"a": 1}),
        ('Here you go:\n```json\n{"a": 1}\n```\nHope that helps!', {"a": 1}),
        ('Sure! {"a": 1} — let me know', {"a": 1}),
        ("```\n[1, 2, 3]\n```", [1, 2, 3]),
        ('{"nested": {"b": [1, {"c": 2}]}}', {"nested": {"b": [1, {"c": 2}]}}),
    ]
    for raw, want in cases:
        try:
            got = _json.loads(_extract_json(raw))
        except Exception as exc:  # noqa: BLE001 - report, don't crash the suite
            got = f"<error: {exc}>"
        check(f"json salvage from {raw[:32]!r}", got, want)


def check_mapper(check) -> None:
    """Phase 5 ranking — deterministic half only, no LLM calls."""
    from agents.mapper import classify_role, public_api_names, rank_symbols, score_symbol

    pkg_src = '''
def helper(x):
    """Shared helper."""
    return x


def _private_helper(x):
    return x


class Widget:
    """A widget."""

    def render(self):
        return helper(1)

    def _internal(self):
        return helper(2)


def orchestrate():
    """Top-level flow."""
    w = Widget()
    return w.render(), helper(3), _private_helper(4)
'''
    init_src = (
        "from .core import Widget as Widget\n"
        "from .core import helper as helper\n"
        '__all__ = ["Widget", "helper"]\n'
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "core.py").write_text(pkg_src, encoding="utf8")
        (root / "__init__.py").write_text(init_src, encoding="utf8")
        (root / "examples").mkdir()
        (root / "examples" / "__init__.py").write_text(
            "from .demo import demo_thing as demo_thing\n", encoding="utf8")
        a = analyze_files(root, [root / "core.py"])

        exports = public_api_names(a)
        check("exports found", exports, {"Widget", "helper"})
        check("examples/ excluded from exports", "demo_thing" in exports, False)

        by_name = {q.split(".")[-1]: a.symbols[q] for q in a.symbols}
        s_helper = score_symbol(by_name["helper"], a, exports)
        s_private = score_symbol(by_name["_private_helper"], a, exports)
        s_render = score_symbol(by_name["render"], a, exports)
        s_internal = score_symbol(by_name["_internal"], a, exports)

        check("exported helper outranks private helper", s_helper > s_private, True)
        check("public method on exported class outranks its private sibling",
              s_render > s_internal, True)
        check("orchestrator role detected",
              classify_role(by_name["orchestrate"], a), "entry point")
        check("class role detected", classify_role(by_name["Widget"], a), "type")

        ranked = rank_symbols(a, 3)
        check("ranking returns requested count", len(ranked), 3)
        check("ranking is sorted by score",
              [n.score for n in ranked] == sorted((n.score for n in ranked), reverse=True),
              True)
        check("no private symbol in top 3",
              any(n.qualname.split(".")[-1].startswith("_") for n in ranked), False)
        # Ranking must be reproducible or Phase 8 retries become non-deterministic.
        check("ranking is stable across runs",
              [n.qualname for n in rank_symbols(a, 3)], [n.qualname for n in ranked])


def check_writer(check) -> None:
    """Phase 6 grounding + draft round-trip. Offline."""
    from agents.writer import Draft, Section, ungrounded_names

    known = {"Flask", "wsgi_app", "flask.app.Flask.wsgi_app", "helper"}

    s = Section(
        key="t", kind="component", heading="Test",
        body=(
            "The `Flask` class exposes `wsgi_app`, defined in `src/flask/app.py`. "
            "It takes a `func` argument and calls `totally_made_up_thing()`. "
            "See `flask.app.Flask.wsgi_app` for detail."
        ),
        allowed=["flask.app.Flask.wsgi_app"],
    )
    bad = ungrounded_names(s, known)
    check("invented name is flagged", "totally_made_up_thing" in bad, True)
    check("real name is not flagged", "Flask" in bad, False)
    check("qualname resolves via its last segment",
          "flask.app.Flask.wsgi_app" in bad, False)
    check("file path is not flagged", "src/flask/app.py" in bad, False)
    check("short param name is ignored", "func" in bad, False)

    # Prose backticks were flooding the checker with false positives until the
    # system prompt banned them; make sure the filter still drops them.
    s2 = Section(key="t2", kind="overview", heading="T",
                 body="The `File Sending` component is in `Blueprint utilities`.")
    check("multi-word prose in backticks is ignored", ungrounded_names(s2, known), [])

    # Phase 8 reloads drafts between retries — the round-trip must be lossless.
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "d.json"
        d = Draft(repo="r", model="m", sections=[s, s2])
        p.write_text(d.to_json(), encoding="utf8")
        back = Draft.load(p)
    check("draft round-trips", [x.key for x in back.sections], ["t", "t2"])
    check("section fields survive round-trip", back.sections[0].allowed, s.allowed)
    check("markdown has one heading per section",
          d.to_markdown().count("\n## "), 2)


def check_critic(a, check) -> None:
    """Phase 7 verification — the deterministic half. No LLM.

    Every false-positive case here was a real one produced on Flask's own docs.
    """
    from agents.critic import Review, verify_claims

    def one(claim):
        out = verify_claims([claim], a)
        return "clean" if not out else f"{out[0].severity}/{out[0].kind}"

    # --- true edges pass, invented edges fail ---
    check("verified call passes", one(
        {"type": "calls", "subject": "uses_inherited", "object": "shared",
         "quote": "uses_inherited calls shared"}), "clean")
    check("fabricated call is an error", one(
        {"type": "calls", "subject": "helper", "object": "shared",
         "quote": "helper calls shared"}), "error/false-call")
    check("call to a symbol outside the repo is only a warning", one(
        {"type": "calls", "subject": "helper", "object": "requests_get",
         "quote": "helper calls requests_get"}), "warning/unverifiable")

    # --- locations ---
    sym = a.symbols["sample.Base.shared"]
    check("correct file+line passes", one(
        {"type": "location", "subject": "shared",
         "object": f"{sym.file}:{sym.start_line}",
         "quote": f"shared is at {sym.file}:{sym.start_line}"}), "clean")
    check("wrong file is an error", one(
        {"type": "location", "subject": "shared", "object": "other/place.py:3",
         "quote": "shared is defined in other/place.py:3"}), "error/wrong-location")
    check("line number absent from the quote is ignored", one(
        {"type": "location", "subject": "shared", "object": f"{sym.file}:999",
         "quote": f"shared lives in {sym.file}"}), "clean")
    check("location misbound to a neighbour downgrades to warning", one(
        {"type": "location", "subject": "helper", "object": f"{sym.file}:999",
         "quote": "the shared method (sample.py:999)"}), "warning/unverifiable")
    # The extractor files behaviour claims under "location" with a prose object.
    check("prose as a location is ignored", one(
        {"type": "location", "subject": "shared", "object": "technical documentation",
         "quote": "shared appears in the technical documentation"}), "clean")
    check("a kind is not a location", one(
        {"type": "location", "subject": "shared", "object": "class",
         "quote": "shared is defined in a class"}), "clean")
    check("a real path with spaces is still checked", one(
        {"type": "location", "subject": "shared", "object": "my dir/place.py",
         "quote": "shared is in my dir/place.py"}), "error/wrong-location")

    check("module path counts as a location", one(
        {"type": "location", "subject": "shared", "object": "sample",
         "quote": "shared lives in module sample"}), "clean")

    # --- prose must never fail a section ---
    # Not merely downgraded to a warning: a prose subject is not a claim about a
    # symbol at all, and an unactionable warning is just noise in the report.
    check("prose noun is not reported", one(
        {"type": "location", "subject": "the whole flow", "object": "sample.py",
         "quote": "the whole flow is defined in sample.py"}), "clean")
    check("a bare unknown identifier is still a warning", one(
        {"type": "location", "subject": "mystery_thing", "object": "sample.py",
         "quote": "mystery_thing is defined in sample.py"}), "warning/unknown-name")
    check("file path as subject is ignored", one(
        {"type": "behaviour", "subject": "sample.py", "object": "does things",
         "quote": "sample.py does things"}), "clean")
    # Component titles were flooding the final report's "not verifiable" list.
    check("English phrase as subject is ignored", one(
        {"type": "behaviour", "subject": "Request Processing",
         "object": "handles requests", "quote": "Request Processing handles requests"}),
        "clean")
    check("a phrase still cannot be a caller", one(
        {"type": "calls", "subject": "Request Processing", "object": "helper",
         "quote": "Request Processing calls helper"}), "warning/unknown-name")

    check("owning class counts as a location", one(
        {"type": "location", "subject": "shared", "object": "sample.Base",
         "quote": "shared lives on sample.Base"}), "clean")

    r = Review("k", "H", False, 2, verify_claims(
        [{"type": "calls", "subject": "helper", "object": "shared", "quote": "q"}], a))
    check("feedback names the problem", "false-call" in r.feedback(), True)
    check("errors are separated from warnings", len(r.errors), 1)


def check_params(a, check) -> None:
    """Parameter names must not be reported as invented (Phase 7/8 convergence).

    Docs correctly write "it takes a `directory` and `**kwargs`". Before params
    were extracted, the Critic called those hallucinations and the Phase 8 retry
    loop burned every round trying to fix prose that was already right.
    """
    from agents.critic import known_names
    from agents.writer import Section, ungrounded_names

    check("params are extracted", a.symbols["sample.helper"].params, ["x"])
    check("self is not a param", a.symbols["sample.Base.shared"].params, [])

    known = known_names(a)
    check("param name is a known name", "x" in known, True)

    s = Section(key="p", kind="component", heading="P",
                body="It takes `x` and returns it. It also uses `made_up_symbol`.")
    bad = ungrounded_names(s, known)
    check("param in backticks is not flagged", "x" in bad, False)
    check("invented name is still flagged", "made_up_symbol" in bad, True)

    s2 = Section(key="p2", kind="component", heading="P",
                 body="Pass `**kwargs` through.")
    check("**kwargs resolves to the kwargs param",
          ungrounded_names(s2, known | {"kwargs"}), [])


def check_aggregator(a, check) -> None:
    """Phase 9 assembly — no LLM, so every part of it is checkable."""
    from agents.critic import Issue, Review
    from agents.mapper import Component, RepoMap, rank_symbols
    from agents.writer import Draft, Section
    from aggregator import (build_document, mermaid_call_graph, symbol_reference,
                            verification_report)

    notes = rank_symbols(a, 9)
    for n in notes:
        n.purpose = f"does something with {n.qualname.split('.')[-1]}"

    diagram = mermaid_call_graph(a, notes)
    check("diagram is a mermaid block", diagram.startswith("```mermaid"), True)
    check("diagram declares a direction", "graph TD" in diagram, True)
    # sample.py: outer -> inner, inner -> helper, Child methods -> Base.shared
    check("diagram contains real edges", "-->" in diagram, True)
    check("node ids are mermaid-safe",
          all(re.fullmatch(r"n_[0-9a-zA-Z_]+", tok)
              for line in diagram.splitlines()
              for tok in re.findall(r"\bn_\S+?(?=[\[\s])", line)), True)
    # An unconnected top-N would render as floating boxes that teach nothing.
    ids_in_edges = set()
    for line in diagram.splitlines():
        if "-->" in line:
            ids_in_edges.update(x.strip() for x in line.split("-->"))
    declared = set(re.findall(r"^\s+(n_\w+)\[", diagram, re.M))
    check("every drawn node participates in an edge", declared - ids_in_edges, set())

    # Streamlit renders DOT natively and offline; Mermaid would need a CDN.
    from aggregator import graphviz_call_graph
    dot = graphviz_call_graph(a, notes)
    check("dot is a digraph", dot.startswith("digraph calls {"), True)
    check("dot is balanced", dot.count("{"), dot.count("}"))
    check("dot uses arrows", "->" in dot, True)
    check("both formats draw the same edges",
          dot.count(" -> "), diagram.count(" --> "))

    reviews = [
        Review("a", "Alpha", True, 4, []),
        Review("b", "Beta", False, 3,
               [Issue("false-call", "error", "x", "x does not call y", "q"),
                Issue("unverifiable", "warning", "z", "z is external", "q")]),
    ]
    rep = verification_report(reviews)
    check("report counts verified sections", "1/2 sections verified" in rep, True)
    check("report totals the claims", "**7 claims checked**" in rep, True)
    check("singular is not written as '1 claims'", "1 claims" in rep, False)
    check("report surfaces the failure", "does not call" in rep, True)
    check("report separates unverifiable from errors", "Not verifiable" in rep, True)

    ref = symbol_reference(notes[:3])
    check("reference is a table", ref.startswith("| Symbol |"), True)
    check("reference has one row per symbol", ref.count("\n"), 4)  # header+sep+3

    m = RepoMap(repo="sample", root=str(a.root), model="m",
                total_symbols=len(a.symbols), total_edges=len(a.edges),
                entry_points=[], components=[Component("sample.py", "T", "S", [])],
                notes=notes)
    draft = Draft(repo="sample", model="m", sections=[
        Section("s1", "First bit", "Body text.", "component")])
    doc = build_document(a, m, draft, reviews)

    check("document leads with the repo name", doc.startswith("# sample"), True)
    check("document states its verification rate", "1/2 sections" in doc, True)
    check("document keeps the written prose", "Body text." in doc, True)
    for heading in ("## Architecture", "## Symbol reference",
                    "## Verification report", "## Contents"):
        check(f"document has {heading}", heading in doc, True)
    check("contents links the written section", "[First bit](#first-bit)" in doc, True)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = root / "sample.py"
        path.write_text(SAMPLE, encoding="utf8")
        a = analyze_files(root, [path])

    failures: list[str] = []

    def check(label: str, got, want) -> None:
        if got != want:
            failures.append(f"{label}\n     got:  {got}\n     want: {want}")

    check("parse errors", a.parse_errors, [])
    check("symbol set", sorted(a.symbols), sorted(
        set(EXPECTED_CALLS) | {"sample.Base", "sample.Child"}))
    for qual, want in EXPECTED_CALLS.items():
        if qual in a.symbols:
            check(f"callees of {qual}", a.callees(qual), want)

    if "sample.Child" in a.symbols:
        check("Child bases", a.symbols["sample.Child"].bases, ["Base"])
    if "sample.Child.uses_inherited" in a.symbols:
        check("decorator inside symbol span",
              a.symbols["sample.Child.uses_inherited"].source.startswith("@functools.cache"),
              True)
    if "sample.Base" in a.symbols:
        check("class docstring", a.symbols["sample.Base"].docstring, "Base docstring.")
    check("nothing unresolved", [d for _, d in a.unresolved], [])

    check_chunking(a, check)
    check_json_salvage(check)
    check_mapper(check)
    check_writer(check)
    check_critic(a, check)
    check_params(a, check)
    check_aggregator(a, check)

    if failures:
        print(f"FAIL ({len(failures)})")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"OK - {len(a.symbols)} symbols, {len(a.edges)} edges, all assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
