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

    def __init__(self):
        self.state = []

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


def documented_caller():
    """Delegates the work, using helper under the hood."""
    return outer()
'''

EXPECTED_CALLS = {
    "sample.helper": [],
    "sample.outer": ["sample.outer.inner"],
    "sample.outer.inner": ["sample.helper"],
    "sample.Base.__init__": [],
    "sample.Base.shared": ["sample.helper"],
    # self.shared() and super().shared() both reach the inherited method
    "sample.Child.uses_inherited": ["sample.Base.shared"],
    "sample.Child.uses_super": ["sample.Base.shared"],
    # opts.setdefault() is a dict method, not an edge — but helper(3) nested
    # inside its arguments still counts
    "sample.Child.local_dict": ["sample.helper"],
    # Its docstring mentions `helper`, which it never calls. Used to check that
    # a claim the source's own docs make is a warning, not an error.
    "sample.documented_caller": ["sample.outer"],
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
    from agents.critic import name_severity
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

    # Example values in a code demo are not invented API.
    s_lit = Section(key="lit", kind="component", heading="L",
                    body="Stored as `'Key'` and `\"key2\"`, shown as `Key`.")
    lit_bad = ungrounded_names(s_lit, known)
    check("quoted literals are ignored", [b for b in lit_bad if "'" in b or '\"' in b], [])

    # Answers quote real usage, which is full of expressions rather than names.
    s_expr = Section(key="ex", kind="component", heading="E",
                     body="Set `app.config['UPLOAD_FOLDER']` and pass "
                          "`as_attachment=True` to it.")
    check("code expressions are not treated as names",
          ungrounded_names(s_expr, known), [])

    # Cleanup of things a 7B model emits regardless of the prompt.
    from agents.writer import clean_body
    check("em dashes are removed", clean_body("Runs — always — fine."),
          "Runs, always, fine.")
    check("role labels are stripped",
          clean_body("`url_for` (entry point) makes URLs."),
          "`url_for` makes URLs.")
    check("copied fact headings are stripped",
          clean_body("It wraps a function.\n\nCalls (verified):\n- `record`"),
          "It wraps a function.")
    check("ordinary brackets survive",
          clean_body("It returns a list (usually empty)."),
          "It returns a list (usually empty).")
    check("a bare single word is only a warning", name_severity("Key"), "warning")
    check("snake_case unknown is an error", name_severity("turbo_encabulate"), "error")
    check("dotted unknown is an error", name_severity("flask.nope"), "error")
    check("CamelCase unknown is an error", name_severity("TurboEncabulator"), "error")
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


def check_example_placeholders(a, check) -> None:
    """A placeholder in the section's own example is not an invented API."""
    from agents.critic import review_section
    from agents.writer import Section

    body = (
        "Use `helper` to do the work. For example:\n\n"
        "```python\n"
        "result = helper(my_input_value)\n"
        "```\n\n"
        "Here `my_input_value` is whatever you want to pass in. "
        "It also calls `totally_made_up_api`."
    )
    s = Section(key="ex", kind="component", heading="E", body=body)
    # review_section calls the model; only the deterministic name pass is
    # exercised here by checking the issues it produces for known names.
    from agents.critic import known_names, name_severity
    from agents.writer import ungrounded_names

    known = known_names(a)
    flagged = ungrounded_names(s, known)
    check("both unknown names are found",
          {"my_input_value", "totally_made_up_api"} <= set(flagged), True)

    import re as _re
    in_examples = set()
    for block in _re.findall(r"```[^\n]*\n(.*?)```", s.body, _re.S):
        in_examples.update(_re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", block))
    check("the placeholder is recognised as example-only",
          "my_input_value" in in_examples, True)
    check("a name outside the example is not excused",
          "totally_made_up_api" in in_examples, False)
    check("a name outside the example is still an error",
          name_severity("totally_made_up_api"), "error")


PARAM_SAMPLE = '''
class Agent:
    """Does work."""

    def __init__(self, settings=None, resolver=None):
        self.settings = settings

    def run(self, methodology=None, *, use_docker=True):
        return methodology


def build(*, checkpointer=None, interrupt_before=None):
    return checkpointer
'''


def check_parameter_claims(check) -> None:
    """Parameters claimed for the wrong function.

    ARPA's documentation said DatasetAgent "is initialized with the methodology
    ... whether to use Docker". Both are real names, so the unknown-name check
    stayed quiet; both belong to run(), not __init__. The docs described the
    wrong method's signature and the section still passed.
    """
    import tempfile
    from pathlib import Path as _P

    from agents.critic import verify_claims
    from analyze import analyze_files

    d = _P(tempfile.mkdtemp())
    (d / "m.py").write_text(PARAM_SAMPLE, encoding="utf8")
    a = analyze_files(d, [d / "m.py"])

    def issues(subject, obj, quote):
        return verify_claims(
            [{"type": "parameter", "subject": subject, "object": obj,
              "quote": quote}], a)

    # A class claim is really a claim about __init__.
    bad = issues("Agent", "use_docker", "Agent is initialized with use_docker")
    check("a run() parameter claimed for the constructor is an error",
          [i.kind for i in bad], ["wrong-parameter"])
    check("the error says where the parameter really lives",
          "run" in bad[0].detail, True)
    check("the error names the class, not __init__",
          "Agent constructor" in bad[0].detail, True)

    check("a real constructor parameter passes",
          issues("Agent", "resolver", "Agent takes a resolver"), [])
    check("a real function parameter passes",
          issues("build", "checkpointer", "build takes a checkpointer"), [])
    check("an invented parameter is an error",
          [i.kind for i in issues("build", "nope", "build takes a nope")],
          ["wrong-parameter"])
    # Same guard the call branch uses: a sentence naming neither side means
    # extraction built the pair rather than reading it.
    check("a claim from a sentence naming neither side is dropped",
          issues("build", "nope", "the module does several things"), [])
    check("prose objects are not treated as parameters",
          issues("build", "an optional checkpointer", "build takes one"), [])


def check_critic(a, check) -> None:
    """Phase 7 verification — the deterministic half. No LLM.

    Every false-positive case here was a real one produced on Flask's own docs.
    """
    from agents.critic import Review, verify_claims

    check_parameter_claims(check)

    def one(claim):
        out = verify_claims([claim], a)
        return "clean" if not out else f"{out[0].severity}/{out[0].kind}"

    # --- true edges pass, invented edges fail ---
    check("verified call passes", one(
        {"type": "calls", "subject": "uses_inherited", "object": "shared",
         "quote": "uses_inherited calls shared"}), "clean")
    # Also the reversal case: the real edge runs shared -> helper. A claim
    # stating a true relationship backwards must stay an error, since that is
    # precisely the mistake this tool exists to catch.
    check("fabricated call is an error", one(
        {"type": "calls", "subject": "helper", "object": "shared",
         "quote": "helper calls shared"}), "error/false-call")
    # "prepare_url is a method of PreparedRequest" is ownership, and the model
    # files these under 'calls' constantly.
    check("owning class is not a false call", one(
        {"type": "calls", "subject": "shared", "object": "Base",
         "quote": "shared is a method of Base"}), "clean")
    # Nested definitions count as containment too: `inner` lives inside `outer`.
    check("enclosing function is not a false call", one(
        {"type": "calls", "subject": "inner", "object": "outer",
         "quote": "inner is defined inside outer"}), "clean")

    # From a real click run: a sentence about what a module does became a call
    # claim between two of its functions, naming neither.
    check("call pair invented from a sentence naming neither is a warning", one(
        {"type": "calls", "subject": "helper", "object": "shared",
         "quote": "This module prints messages and formats filenames."}),
        "warning/unverifiable")
    check("naming just the caller is enough to judge it", one(
        {"type": "calls", "subject": "helper", "object": "shared",
         "quote": "helper does the work itself"}), "error/false-call")
    check("naming just the callee is enough to judge it", one(
        {"type": "calls", "subject": "helper", "object": "shared",
         "quote": "it hands off to shared"}), "error/false-call")

    # Flask documents send_from_directory as "using send_file" while the code
    # delegates to werkzeug. Repeating the authors is not the writer's mistake.
    check("a call the docstring itself claims is a warning", one(
        {"type": "calls", "subject": "documented_caller", "object": "helper",
         "quote": "documented_caller uses helper to do the work"}),
        "warning/unverifiable")

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
    # Real case: an answer named FlaskCliRunner in one sentence and cli.py in
    # the next, and the path was attributed to the class across the full stop.
    check("location bound across a sentence boundary is not an error", one(
        {"type": "location", "subject": "shared", "object": "other/place.py",
         "quote": "The shared method is a helper. The main function in "
                  "other/place.py is the entry point."}), "warning/unverifiable")
    check("same sentence still fails when genuinely wrong", one(
        {"type": "location", "subject": "shared", "object": "other/place.py",
         "quote": "shared is defined in other/place.py"}), "error/wrong-location")

    check("location misbound to a neighbour downgrades to warning", one(
        {"type": "location", "subject": "helper", "object": f"{sym.file}:999",
         "quote": "the shared method (sample.py:999)"}), "warning/unverifiable")
    # The extractor files behaviour claims under "location" with a prose object.
    check("prose as a location is ignored", one(
        {"type": "location", "subject": "shared", "object": "technical documentation",
         "quote": "shared appears in the technical documentation"}), "clean")
    check("a call expression is not a location", one(
        {"type": "location", "subject": "shared", "object": "Base(__name__)",
         "quote": "obj = Base(__name__)"}), "clean")
    # Answers to questions quote usage, so the package name shows up as a place.
    check("the package name counts as a location", one(
        {"type": "location", "subject": "shared", "object": a.root.name,
         "quote": f"shared lives in {a.root.name}"}), "clean")
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

    # --- answers quote examples; the example is not a claim about the repo ---
    from agents.critic import prose_only, self_defined_names

    answer = (
        "Use `helper` for this.\n\n"
        "```python\n@app.route('/x')\ndef download_file():\n"
        "    return helper(1)\n```\n\n"
        "The `download_file` view then returns it."
    )
    # Assert on text that appears ONLY inside the block, or the check cannot
    # tell the block from the prose that discusses it.
    check("code block contents are excluded from verification",
          "@app.route" in prose_only(answer), False)
    check("surrounding prose survives",
          "Use `helper` for this." in prose_only(answer), True)
    check("names defined in the example are known",
          "download_file" in self_defined_names(answer), True)
    check("assignments in the example are known",
          "app" in self_defined_names("```python\napp = Base()\n```"), True)
    check("equality is not read as an assignment",
          self_defined_names("```python\nif x == 1:\n    pass\n```"), set())

    # The quote must name at least one of the pair, or the claim is treated as
    # misattributed by extraction rather than as a false call.
    r = Review("k", "H", False, 2, verify_claims(
        [{"type": "calls", "subject": "helper", "object": "shared",
          "quote": "helper calls shared"}], a))
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

    # Instance attributes are real code that docs mention, and are not symbols.
    check("self attributes are collected", "state" in a.attributes, True)
    check("attribute is a known name", "state" in known_names(a), True)
    s_attr = Section(key="at", kind="component", heading="A",
                     body="It appends to the `state` list.")
    check("attribute in backticks is not flagged",
          ungrounded_names(s_attr, known_names(a)), [])

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

    # The ranked symbols alone made a poor graph: on click only 2 of the top 8
    # called each other, so the picture was two boxes and one arrow. The graph
    # now grows outwards along real calls.
    from aggregator import module_edges

    seeded = mermaid_call_graph(a, notes[:1])
    check("graph grows past its seed symbol",
          seeded.count("-->") >= 2, True)
    check("growth stays within max_nodes",
          len(re.findall(r"^\s+n_\w+\[", mermaid_call_graph(a, notes, 4), re.M)) <= 4,
          True)

    mods = module_edges(a)
    check("single-file sample has no cross-file edges", mods, [])

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

    # --- limitations: every number must be counted, not asserted ---
    from aggregator import limitations

    lim = limitations(a, m, reviews)
    described, total = len(m.notes), len(a.symbols)
    check("states how many symbols were described",
          f"{described} of {total}" in lim, True)
    check("states the unresolved call count",
          f"{len(a.edges)} calls were matched" in lim, True)
    check("admits grouping is unchecked", "Grouping is not checked" in lim, True)
    check("admits runtime behaviour is invisible",
          "dynamic dispatch" in lim, True)
    check("counts unverifiable claims from the review",
          "1 claim(s) in this document could not be checked" in lim, True)
    # Local variables like `ctx` show up in unresolved calls. Calling them
    # third-party inside the honesty section would be its own small lie.
    check("does not label unresolved targets as third-party",
          "third-party" in lim, False)
    check("limitations appears in the document",
          "## What this does not cover" in doc, True)


def check_manifest(check) -> None:
    """Getting-started extraction. Pure file reading, no model, no network."""
    from manifest import read_manifest, render

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "pyproject.toml").write_text(
            '[project]\n'
            'name = "demolib"\n'
            'description = "A demo"\n'
            'requires-python = ">=3.10"\n'
            'dependencies = ["requests>=2", "click"]\n'
            '[project.scripts]\n'
            'demo = "demolib.cli:main"\n', encoding="utf8")
        (root / "README.md").write_text(
            "# demolib\n\nInstall it:\n\n```bash\n$ pip install demolib\n```\n\n"
            "Use it:\n\n```python\nimport demolib\ndemolib.go()\n```\n",
            encoding="utf8")
        (root / "__main__.py").write_text("print('hi')\n", encoding="utf8")
        (root / "runme.py").write_text(
            'if __name__ == "__main__":\n    pass\n', encoding="utf8")
        (root / "quiet.py").write_text("x = 1\n", encoding="utf8")
        (root / "tests").mkdir()

        m = read_manifest(root, list(root.rglob("*.py")))

        check("project name read", m.name, "demolib")
        check("python requirement read", m.python_requires, ">=3.10")
        check("dependencies read", m.dependencies, ["requests>=2", "click"])
        check("console scripts read", m.console_scripts, {"demo": "demolib.cli:main"})
        # The "$ " prompt must be stripped or the command is not copy-pasteable.
        check("install command quoted from README", m.install_cmds,
              ["pip install demolib"])
        check("usage example prefers the python block",
              m.usage_block.startswith("import demolib"), True)
        check("runnable files detected", m.runnable_modules,
              ["__main__.py", "runme.py"])
        check("non-runnable file excluded", "quiet.py" in m.runnable_modules, False)
        check("test suite detected", m.has_tests, True)

        out = render(m)
        check("render marks the install as quoted", "quoted from README.md" in out, True)
        check("render includes the entry point", "demolib.cli:main" in out, True)

        # Ordered steps, not a pile of facts: version, install, run, ways in,
        # check. A reader should be able to follow it top to bottom.
        steps = re.findall(r"^### (\d+)\. (.+)$", out, re.M)
        check("steps are numbered from 1 with no gaps",
              [int(n) for n, _ in steps], list(range(1, len(steps) + 1)))
        check("version check comes before install",
              [t for _, t in steps][:2],
              ["Check your Python version", "Install it"])
        check("running something comes after installing",
              [t for _, t in steps].index("Run the smallest thing that works") > 1,
              True)

        # A bare repo must produce nothing rather than an invented section.
        bare = Path(tmp) / "bare"
        bare.mkdir()
        check("empty repo renders nothing", render(read_manifest(bare, [])), "")


def check_alternatives(check) -> None:
    """Dependency substitutes. Offline: no PyPI calls, no model."""
    import alternatives as alt
    from alternatives import DepAlternatives, Package, dep_name, render

    check("version specifier stripped", dep_name("click>=8.1.3"), "click")
    check("extras stripped", dep_name("foo[extra]>=1"), "foo")
    check("marker stripped", dep_name("bar; python_version>'3'"), "bar")

    rows = [DepAlternatives(
        name="jinja2", does="template engine",
        open_source=[Package(name="Mako", summary="A templating language",
                             license="MIT", home="https://example.invalid",
                             relevance=0.68)],
        rejected=["django-redis (does something else, 0.13)"])]
    out = render(rows)
    check("real candidate is shown", "Mako" in out, True)
    check("pypi summary is used", "A templating language" in out, True)
    check("similarity is shown so the reader can judge",
          "similarity 0.68" in out, True)
    # The paid column was removed: nothing could check it, and it suggested
    # free web frameworks as commercial substitutes for an HTTP client.
    check("no paid column is rendered", "Paid or hosted" in out, False)
    check("its removal is explained rather than silent",
          "A paid or hosted column was tried and removed" in out, True)
    check("dropped names are listed with their reason",
          "django-redis (does something else, 0.13)" in out, True)
    # The footer used to claim everything was dropped for not existing.
    check("footer does not misstate the reason",
          "not found on PyPI" in out, False)
    check("nothing to show renders nothing", render([]), "")

    # A package whose PyPI page says not to use it is not an alternative.
    check("deprecation is detected from the summary",
          bool(alt.DEAD_SUMMARY.search(
              "Deprecated backport of asyncio; use the stdlib package instead")),
          True)
    check("a healthy summary is not mistaken for one",
          bool(alt.DEAD_SUMMARY.search("A friendly Python library for async "
                                       "concurrency and I/O")), False)

    empty = render([DepAlternatives(name="itsdangerous", does="signing")])
    check("no survivors says so rather than inventing one",
          "Nothing proposed survived" in empty, True)

    # A cached entry written before Package gained a field must miss, not load
    # silently with that field empty. That bug disabled a filter for a while.
    # The network is stubbed out so a miss is observable as None.
    import urllib.request

    entry = ('{{"name": "demo", "summary": "s", "license": "", "home": "",'
             ' "requires": [], "relevance": 0.0, "_v": {v}}}')

    def boom(*_a, **_kw):
        raise OSError("network disabled for this test")

    with tempfile.TemporaryDirectory() as tmp:
        old_dir, alt.CACHE_DIR = alt.CACHE_DIR, Path(tmp)
        old_open, urllib.request.urlopen = urllib.request.urlopen, boom
        try:
            path = Path(tmp) / "demo.json"

            path.write_text(entry.format(v=alt.CACHE_VERSION), encoding="utf8")
            check("current cache entry is used", getattr(
                alt.pypi_info("demo"), "summary", None), "s")

            path.write_text(entry.format(v=alt.CACHE_VERSION - 1), encoding="utf8")
            check("stale cache entry is refused", alt.pypi_info("demo"), None)
        finally:
            alt.CACHE_DIR = old_dir
            urllib.request.urlopen = old_open


def check_graph_index(check) -> None:
    """Adjacency is indexed once, not rescanned per lookup.

    Ranking asks for callers and callees of every symbol. Scanning all edges
    each time made unsloth take over ten minutes before a single model call.
    """
    import time

    from analyze import Analysis

    n = 400
    edges = [(f"m.f{i}", f"m.f{(i + 1) % n}") for i in range(n)]
    edges += [(f"m.f{i}", "m.hub") for i in range(n)]
    a = Analysis(root=Path("."), symbols={}, edges=edges, unresolved=[],
                 parse_errors=[])

    check("callees are correct", a.callees("m.f0"), ["m.f1", "m.hub"])
    check("callers are correct", len(a.callers("m.hub")), n)
    check("a symbol with no edges returns empty", a.callees("m.nobody"), [])

    started = time.monotonic()
    for i in range(n):
        a.fan(f"m.f{i}")
    elapsed = time.monotonic() - started
    # Indexed this is microseconds. Rescanning 800 edges 800 times is not.
    check("fan-out over the whole graph stays fast", elapsed < 0.5, True)


def check_scoping(check) -> None:
    """Reading one folder of a repo that holds several projects."""
    from ingest import list_source_files, suggest_subdirs

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "lib").mkdir()
        (root / "webapp").mkdir()
        (root / "lib" / "core.py").write_text("def a(): pass\n", encoding="utf8")
        for i in range(4):
            (root / "webapp" / f"page{i}.tsx").write_text(
                "export const C = () => 1;\n", encoding="utf8")

        check("unscoped reads everything", len(list_source_files(root)), 5)
        check("scoped reads only that folder",
              [p.name for p in list_source_files(root, subdir="lib")], ["core.py"])
        # The suggestion exists so a reader can see which half is which.
        check("areas are ranked by size", suggest_subdirs(root)[0], ("webapp", 4))

        try:
            list_source_files(root, subdir="nope")
            check("a bad folder raises", "no error", "FileNotFoundError")
        except FileNotFoundError as exc:
            check("a bad folder names the real ones", "webapp" in str(exc), True)


def check_insights(a, check) -> None:
    """The repo's own rough edges. Counted, never estimated."""
    from insights import repo_limits, unused

    from insights import plural

    check("one is singular", plural(1, "pair"), "1 pair")
    check("two is plural", plural(2, "pair"), "2 pairs")
    check("zero is plural", plural(0, "test file"), "0 test files")

    limits = repo_limits(a, set())
    labels = " | ".join(f.label for f in limits.findings)
    check("no line reads '1 pairs' or '1 functions'",
          any(f" 1 {w}s" in labels for w in ("pair", "function", "other")), False)

    check("undocumented functions are counted", "docstring" in labels, True)
    check("test suite is reported", "test suite" in labels.lower(), True)
    # Problems belong above clean results, or the page opens with good news.
    goods = [i for i, f in enumerate(limits.findings) if f.good]
    bads = [i for i, f in enumerate(limits.findings) if not f.good]
    check("problems come before clean results",
          (not goods or not bads or max(bads) < min(goods)), True)

    # A public method with no in-repo callers is the library's API, not dead
    # code. Counting those reported 121 "unused" functions in flask.
    u = unused(a, set())
    named = u.detail if u and not u.good else ""
    check("public functions are not called unused", "helper" in named, False)
    check("nothing is flagged unused in this sample", u.good if u else True, True)


def check_phase_flow(a, check) -> None:
    """The ordered view: steps follow real calls, labels are plain English."""
    from agents.mapper import RepoMap, rank_symbols
    from aggregator import (_short_purpose, _title_from_name,
                            markdown_phase_flow, phase_flow)

    # Fallback labels, for steps whose symbol has no usable purpose. Both of
    # these shipped wrong: capitalize() lowercases the tail, so the class
    # PaperBlock read "Paperblock", and it only uppercases position zero, which
    # on a private name is the space the underscore left behind.
    check("a private helper loses its underscore, not its capital",
          _title_from_name("pkg.mod._select_central"), "Select central")
    check("a class keeps its own casing",
          _title_from_name("pkg.mod.PaperBlock"), "PaperBlock")
    check("a fallback label never starts with a space",
          _title_from_name("pkg.mod._clean_text").startswith(" "), False)
    check("dunders do not come out empty",
          _title_from_name("pkg.mod.__init__"), "Init")

    # What matters is not ending ON a preposition. Cutting at every preposition
    # would wreck "Inserts multiple records into a SQLite table", which is a
    # perfectly good label.
    trailing = ("into", "to", "for", "with", "by", "of", "a", "an", "the")
    for purpose in ("Converts a view function's return value into an instance",
                    "Inserts multiple records into a SQLite table by batching",
                    "Handles a WSGI request"):
        label = _short_purpose(purpose)
        check(f"label does not end on a preposition: {label!r}",
              label.split()[-1].lower() in trailing, False)
    check("a short purpose survives whole",
          _short_purpose("Handles a WSGI request"), "Handles a WSGI request")
    check("a longer purpose keeps its object",
          _short_purpose("Inserts multiple records into a SQLite table by batching"),
          "Inserts multiple records into a SQLite table")

    notes = rank_symbols(a, 6)
    for n in notes:
        n.purpose = f"Does the {n.qualname.split('.')[-1]} work"
    m = RepoMap(repo="sample", root=str(a.root), model="m",
                total_symbols=len(a.symbols), total_edges=len(a.edges),
                entry_points=[], components=[], notes=notes)

    phases = phase_flow(a, m)
    check("the flow has more than one step", len(phases) > 1, True)
    check("steps are numbered in order",
          [p.number for p in phases], list(range(1, len(phases) + 1)))
    # Each step must be reachable from the one before, or the order is invented.
    for prev, nxt in zip(phases, phases[1:]):
        reachable = {c for q in prev.symbols for c in a.callees(q)}
        check(f"step {nxt.number} follows a real call from step {prev.number}",
              bool(set(nxt.symbols) & reachable), True)
    check("no symbol appears in two steps",
          len({q for p in phases for q in p.symbols}),
          sum(len(p.symbols) for p in phases))
    check("markdown flow is numbered", markdown_phase_flow(a, m).startswith("1. "),
          True)

    # Only the top few symbols get a Mapper note, so deeper steps had no purpose
    # and fell straight to the identifier: one diagram read "Reduces a paper text
    # to candidate blocks", then " select central", then "Paperblock". A
    # docstring sits between the two, and comes from the source, not the model.
    bare = rank_symbols(a, 6)
    for n in bare:
        n.purpose = ""
    m_bare = RepoMap(repo="sample", root=str(a.root), model="m",
                     total_symbols=len(a.symbols), total_edges=len(a.edges),
                     entry_points=[], components=[], notes=bare)
    bare_phases = phase_flow(a, m_bare)
    titles = {p.symbols[0]: p.title for p in bare_phases}
    check("a step with no purpose takes its label from the docstring",
          titles.get("sample.documented_caller"), "Delegates the work")
    check("a step with neither still gets a clean label",
          titles.get("sample.outer"), "Outer")

    # The flow was wired into the app but not into the document, so the
    # downloaded file was missing the one view that explains the order.
    from agents.critic import Review
    from agents.writer import Draft, Section
    from aggregator import build_document

    doc = build_document(
        a, m,
        Draft(repo="sample", model="m",
              sections=[Section("s", "Bit", "Body.", "component")]),
        [Review("s", "Bit", True, 1, [])])
    check("the document carries the flow too",
          "What happens, in order" in doc, True)
    check("the flow comes before the file graph",
          doc.index("What happens, in order") < doc.index("## Symbol reference"),
          True)


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
    check_example_placeholders(a, check)
    check_params(a, check)
    check_aggregator(a, check)
    check_manifest(check)
    check_alternatives(check)
    check_graph_index(check)
    check_scoping(check)
    check_insights(a, check)
    check_phase_flow(a, check)

    if failures:
        print(f"FAIL ({len(failures)})")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"OK - {len(a.symbols)} symbols, {len(a.edges)} edges, all assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
