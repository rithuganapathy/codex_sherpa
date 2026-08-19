"""Phase 9 — assemble the final document: prose + diagram + verification report.

No LLM. Everything here is deterministic assembly of artefacts the earlier
phases already produced, which makes it fast, repeatable, and fully testable.

The verification report is the point of the whole project: a generated document
that states plainly which of its own claims were machine-checked, and which were
not. Most AI-written docs cannot tell you that.

Run:
    python aggregator.py flask
    python aggregator.py flask --url https://github.com/psf/requests
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import manifest
from agents.critic import Issue, Review
from agents.mapper import OUT_DIR, RepoMap, SymbolNote
from agents.writer import Draft
from analyze import Analysis, analyze

MAX_DIAGRAM_NODES = 16


def _node_id(qualname: str) -> str:
    """Mermaid ids must be alphanumeric-ish and stable."""
    return "n_" + re.sub(r"[^0-9a-zA-Z]", "_", qualname)


def _label(qualname: str) -> str:
    parts = qualname.split(".")
    return ".".join(parts[-2:]) if len(parts) > 1 else qualname


MAX_MODULE_EDGES = 18


def module_edges(analysis: Analysis, max_edges: int = MAX_MODULE_EDGES
                 ) -> list[tuple[str, str, int]]:
    """File-to-file call relationships, heaviest first, as (from, to, count).

    This is the view that actually answers "how is this laid out". A repo can
    have 600 functions and still only a dozen files that talk to each other.
    """
    from collections import Counter

    counts: Counter[tuple[str, str]] = Counter()
    for src, dst in analysis.edges:
        a, b = analysis.symbols[src].file, analysis.symbols[dst].file
        if a != b:
            counts[(a, b)] += 1
    return [(a, b, n) for (a, b), n in counts.most_common(max_edges)]


def _diagram_data(analysis: Analysis, notes: list[SymbolNote], max_nodes: int
                  ) -> tuple[dict[str, list[str]], list[tuple[str, str]]]:
    """Pick the nodes and edges both diagram formats draw.

    Seeded with the ranked symbols, then grown outwards along real calls. The
    ranked symbols alone are usually a poor graph: on click, only 2 of the top 8
    called each other, so the picture was two boxes and one arrow. Their
    immediate neighbours are what turn it into something worth looking at.
    """
    # Seeds are capped too. Without this the ranked list alone can already
    # exceed the budget, and nothing below would bring it back down.
    chosen: list[str] = []
    for q in (n.qualname for n in notes):
        if q not in chosen and len(chosen) < max_nodes:
            chosen.append(q)

    # Grow one hop at a time so the nearest neighbours win the remaining slots.
    frontier = list(chosen)
    while len(chosen) < max_nodes and frontier:
        nxt: list[str] = []
        for q in frontier:
            for neighbour in analysis.callees(q) + analysis.callers(q):
                if neighbour not in chosen and len(chosen) < max_nodes:
                    chosen.append(neighbour)
                    nxt.append(neighbour)
        frontier = nxt

    picked = set(chosen)
    edges = [(a, b) for a, b in analysis.edges if a in picked and b in picked]

    # An isolated box teaches a reader nothing.
    connected = {q for e in edges for q in e}
    by_file: dict[str, list[str]] = {}
    for q in chosen:
        if q in connected:
            by_file.setdefault(analysis.symbols[q].file, []).append(q)
    return by_file, sorted(edges)


def graphviz_call_graph(analysis: Analysis, notes: list[SymbolNote],
                        max_nodes: int = MAX_DIAGRAM_NODES) -> str:
    """Same graph as DOT — Streamlit renders this natively, with no CDN."""
    by_file, edges = _diagram_data(analysis, notes, max_nodes)
    if not edges:
        return ""

    lines = ["digraph calls {", '    rankdir=TB;', '    bgcolor="transparent";',
             # Without an explicit size a 4-node graph gets scaled to whatever
             # width it is given, and the boxes end up the size of paragraphs.
             '    graph [size="7,4.5", ratio=compress, nodesep=0.30, '
             'ranksep=0.45, fontname="Helvetica"];',
             '    node [shape=box, style="rounded,filled", fillcolor="#FDE7F1", '
             'fontname="Helvetica", fontsize=10, color="#E7508F", '
             'fontcolor="#4A2C40", penwidth=1.4];',
             '    edge [color="#A87BD4", arrowsize=0.8, penwidth=1.3];']
    for i, (file, quals) in enumerate(sorted(by_file.items())):
        lines.append(f"    subgraph cluster_{i} {{")
        lines.append(f'        label="{file}"; fontsize=9; color="#F0C8DE"; '
                     f'fontcolor="#8A5C77"; style="rounded";')
        for q in quals:
            lines.append(f'        {_node_id(q)} [label="{_label(q)}"];')
        lines.append("    }")
    for a, b in edges:
        lines.append(f"    {_node_id(a)} -> {_node_id(b)};")
    lines.append("}")
    return "\n".join(lines)


MAX_PHASES = 6


@dataclass
class Phase:
    number: int
    title: str            # plain English, no filenames
    symbols: list[str]    # qualnames, in the order they are reached
    detail: str = ""


# Cutting a purpose at N words strands the sentence on a preposition:
# "Converts a view function's return value into". Drop these from the end.
_TRAILING = {"into", "to", "for", "with", "by", "of", "a", "an", "the", "and",
             "or", "in", "on", "from", "as", "that", "which", "it", "its"}


def _short_purpose(text: str, words: int = 7) -> str:
    """First clause of a Mapper purpose, as a label rather than a sentence."""
    clause = re.split(r"[,.;]| by | which | that ", text.strip(), maxsplit=1)[0]
    parts = clause.split()[:words]
    while parts and parts[-1].lower().strip(".,'\"") in _TRAILING:
        parts.pop()
    out = " ".join(parts).rstrip(" .,")
    return out[0].upper() + out[1:] if out else ""


def phase_flow(analysis: Analysis, repo_map: RepoMap,
               max_phases: int = MAX_PHASES) -> list[Phase]:
    """The order things happen in, starting from the way in.

    A file-dependency graph shows where code lives, which is no help to someone
    who has never heard of `ctx.py`. This follows the real call chain outwards
    from the entry point and names each step in plain English, using the
    purposes the Mapper already wrote. Order and arrows still come from the
    source; only the labels come from the model, and they were written earlier.
    """
    by_qual = {n.qualname: n for n in repo_map.notes}
    if not by_qual:
        return []

    def reach(qual: str, hops: int) -> tuple[int, int]:
        """(steps deep, symbols reached) within the phase budget."""
        seen_r, layer_r, depth = {qual}, [qual], 0
        for _ in range(hops):
            nxt = [c for q in layer_r for c in analysis.callees(q)
                   if c not in seen_r]
            if not nxt:
                break
            seen_r.update(nxt)
            layer_r = nxt
            depth += 1
        return depth, len(seen_r)

    def score(note) -> tuple[int, int, float]:
        # Depth first. The main flow of a program is its longest chain, not its
        # highest-ranked function: on flask, ranking picked url_for, which is a
        # two-step detour, over the wsgi_app request pipeline.
        depth, size = reach(note.qualname, max_phases)
        bonus = {"entry point": 1.0, "orchestrator": 2.0}.get(note.role, 0.0)
        return depth, size, note.score / 100 + bonus

    start = max((n for n in repo_map.notes if analysis.callees(n.qualname)),
                key=score, default=None)
    if start is None:
        return []

    phases: list[Phase] = []
    seen = {start.qualname}
    layer = [start.qualname]

    while layer and len(phases) < max_phases:
        lead = by_qual.get(layer[0])
        title = _short_purpose(lead.purpose) if lead and lead.purpose else ""
        if not title:
            title = layer[0].split(".")[-1].replace("_", " ").capitalize()
        phases.append(Phase(
            number=len(phases) + 1,
            title=title,
            symbols=list(layer),
            detail=lead.purpose if lead else "",
        ))

        nxt: list[str] = []
        for q in layer:
            for callee in analysis.callees(q):
                if callee not in seen:
                    seen.add(callee)
                    nxt.append(callee)
        # Described symbols first: they are the ones with a usable label.
        nxt.sort(key=lambda q: (q not in by_qual, q))
        layer = nxt[:4]
    return phases


def graphviz_phase_flow(analysis: Analysis, repo_map: RepoMap) -> str:
    phases = phase_flow(analysis, repo_map)
    if len(phases) < 2:
        return ""

    lines = ["digraph flow {", "    rankdir=TB;", '    bgcolor="transparent";',
             '    graph [size="6.5,7", ratio=compress, nodesep=0.2, '
             'ranksep=0.45, fontname="Helvetica"];',
             '    node [shape=box, style="rounded,filled", fillcolor="#FDE7F1", '
             'fontname="Helvetica", fontsize=11, color="#E7508F", '
             'fontcolor="#4A2C40", penwidth=1.5, margin="0.18,0.10"];',
             '    edge [color="#A87BD4", arrowsize=0.9, penwidth=1.6];']
    for p in phases:
        names = ", ".join(q.split(".")[-1] for q in p.symbols[:3])
        label = f"{p.number}. {p.title}\\n{names}"
        lines.append(f'    step{p.number} [label="{label}"];')
    for p in phases[:-1]:
        lines.append(f"    step{p.number} -> step{p.number + 1};")
    lines.append("}")
    return "\n".join(lines)


def markdown_phase_flow(analysis: Analysis, repo_map: RepoMap) -> str:
    phases = phase_flow(analysis, repo_map)
    if len(phases) < 2:
        return ""
    out = []
    for p in phases:
        names = ", ".join(f"`{q.split('.')[-1]}`" for q in p.symbols[:3])
        out.append(f"{p.number}. **{p.title}** — {names}")
    return "\n".join(out)


def graphviz_module_graph(analysis: Analysis,
                          max_edges: int = MAX_MODULE_EDGES) -> str:
    """The file-level view, with arrow weight showing how much traffic flows."""
    edges = module_edges(analysis, max_edges)
    if not edges:
        return ""

    heaviest = max(n for _, _, n in edges)
    lines = ["digraph modules {", "    rankdir=LR;", '    bgcolor="transparent";',
             '    graph [size="7.5,5", ratio=compress, nodesep=0.25, '
             'ranksep=0.7, fontname="Helvetica"];',
             '    node [shape=box, style="rounded,filled", fillcolor="#F3E8FB", '
             'fontname="Helvetica", fontsize=10, color="#A87BD4", '
             'fontcolor="#4A2C40", penwidth=1.3];']
    for a, b, n in edges:
        # Thicker arrow means more calls cross that boundary.
        width = 1.0 + 2.5 * (n / heaviest)
        lines.append(f'    "{Path(a).name}" -> "{Path(b).name}" '
                     f'[label=" {n}", penwidth={width:.1f}, '
                     f'color="#E7508F", fontsize=8, fontcolor="#8A5C77"];')
    lines.append("}")
    return "\n".join(lines)


def mermaid_module_graph(analysis: Analysis,
                         max_edges: int = MAX_MODULE_EDGES) -> str:
    edges = module_edges(analysis, max_edges)
    if not edges:
        return ""
    lines = ["```mermaid", "graph LR"]
    for a, b, n in edges:
        lines.append(f"    {_node_id(a)}[\"{Path(a).name}\"] "
                     f"-->|{n}| {_node_id(b)}[\"{Path(b).name}\"]")
    lines.append("```")
    return "\n".join(lines)


def mermaid_call_graph(analysis: Analysis, notes: list[SymbolNote],
                       max_nodes: int = MAX_DIAGRAM_NODES) -> str:
    """Diagram the call graph among the highest-ranked symbols, for Markdown."""
    by_file, edges = _diagram_data(analysis, notes, max_nodes)
    if not edges:
        return ""

    lines = ["```mermaid", "graph TD"]
    for i, (file, quals) in enumerate(sorted(by_file.items())):
        lines.append(f'    subgraph sg{i}["{file}"]')
        for q in quals:
            lines.append(f'        {_node_id(q)}["{_label(q)}"]')
        lines.append("    end")
    for a, b in edges:
        lines.append(f"    {_node_id(a)} --> {_node_id(b)}")
    lines.append("```")
    return "\n".join(lines)


def verification_report(reviews: list[Review]) -> str:
    passed = [r for r in reviews if r.passed]
    claims = sum(r.claims_checked for r in reviews)
    errors = [i for r in reviews for i in r.errors]
    warnings = [i for r in reviews for i in r.warnings]

    def plural(n: int, word: str) -> str:
        return f"{n} {word}" if n == 1 else f"{n} {word}s"

    lines = [
        "Every factual claim in this document was pulled out of the prose and "
        "checked against the call graph built from the source. What follows is "
        "the honest tally, including the claims that could not be settled "
        "either way.",
        "",
        f"- **{len(passed)}/{len(reviews)} sections verified**",
        f"- **{plural(claims, 'claim')} checked** across "
        f"{plural(len(reviews), 'section')}",
        f"- **{plural(len(errors), 'unresolved error')}**, "
        f"{plural(len(warnings), 'unverifiable claim')}",
        "",
        "| Section | Result | Claims | Errors |",
        "| --- | --- | --- | --- |",
    ]
    for r in reviews:
        mark = "verified" if r.passed else "**needs review**"
        lines.append(f"| {r.heading} | {mark} | {r.claims_checked} | {len(r.errors)} |")

    if errors:
        lines += ["", "### Claims that failed verification", "",
                  "These statements appear in the document but contradict the "
                  "source. Treat them as untrusted.", ""]
        for i in errors:
            lines.append(f"- *{i.kind}* — {i.detail}")

    if warnings:
        # The same unverifiable claim often appears in several sections; listing
        # it once is the useful form.
        seen: set[str] = set()
        unique = [i for i in warnings
                  if not (i.detail in seen or seen.add(i.detail))]
        warnings = unique
        shown = warnings[:8]
        lines += ["", "### Not verifiable", "",
                  "Claims about code outside this repository, or phrasing the "
                  "checker could not tie to a symbol. These are not errors, "
                  "just unproven.", ""]
        for i in shown:
            lines.append(f"- {i.detail}")
        if len(warnings) > len(shown):
            lines.append(f"- …and {len(warnings) - len(shown)} more")

    return "\n".join(lines)


def limitations(analysis: Analysis, repo_map: RepoMap,
                reviews: list[Review]) -> str:
    """What this document does not cover, from measured numbers.

    A tool that refuses to state unproven claims should also say where it ran
    out of certainty. Every figure here is counted, not estimated.
    """
    from collections import Counter

    from ingest import list_python_files

    all_py = list(analysis.root.rglob("*.py"))
    kept = list_python_files(analysis.root)
    skipped = len(all_py) - len(kept)

    described = len(repo_map.notes)
    total = len(analysis.symbols)
    unresolved = len(analysis.unresolved)
    edges = len(analysis.edges)
    ambiguous = sum(1 for _, d in analysis.unresolved if "ambiguous" in d)
    external = Counter(d.split(".")[0] for _, d in analysis.unresolved
                       if "." in d and "ambiguous" not in d
                       and d.split(".")[0] not in ("self", "cls"))
    low_conf = [n for n in repo_map.notes if n.confidence == "low"]
    warnings = [i for r in reviews for i in r.warnings]

    out = [
        "This tool is confident about a narrow set of things and vague about "
        "the rest. Here is the boundary, in numbers.",
        "",
        "**Coverage**",
        "",
        f"- {described} of {total} functions and classes are described here. "
        f"They were chosen by call-graph ranking, not by reading everything. "
        f"The other {total - described} are real code this document never "
        f"mentions.",
    ]
    if skipped > 0:
        out.append(f"- {skipped} of {len(all_py)} Python files were skipped "
                   f"before analysis even started: tests, docs, examples and "
                   f"build directories.")
    out.append("- Non-Python files are not read at all, apart from the "
               "packaging files used for Getting started.")

    out += ["", "**What the call graph could not resolve**", ""]
    out.append(f"- {edges} calls were matched to a definition in this "
               f"repository. {unresolved} were not.")
    if external:
        # Not all of these are libraries. The leading segment of `ctx.push()`
        # is a local variable, and calling that "third-party" in the honesty
        # section would be its own small lie.
        names = ", ".join(f"`{n}`" for n, _ in external.most_common(5))
        out.append(f"- Those go either to code outside this repository or to "
                   f"objects whose type cannot be worked out from the syntax "
                   f"alone ({names}). Absence of an arrow is not evidence that "
                   f"no call happens.")
    if ambiguous:
        out.append(f"- {ambiguous} were dropped as ambiguous, where several "
                   f"definitions share a name and the correct one cannot be "
                   f"told apart without type inference.")
    out.append("- Anything decided at runtime is invisible here: dynamic "
               "dispatch, `getattr` lookups, callbacks passed as arguments, "
               "and registries filled in at import time.")

    if analysis.parse_errors:
        out += ["", f"- {len(analysis.parse_errors)} file(s) had syntax the "
                    f"parser could only partly read."]
    if low_conf:
        names = ", ".join(f"`{n.qualname.split('.')[-1]}`" for n in low_conf[:5])
        out += ["", f"- The model reported low confidence describing "
                    f"{len(low_conf)} symbol(s): {names}."]

    out += ["", "**What the verification does not check**", "",
            "- Only two kinds of claim are machine-checked: that one function "
            "calls another, and that a symbol lives where the text says. "
            "Everything else in the prose is unverified.",
            "- Grouping is not checked. A function can be described under the "
            "wrong component and still pass.",
            "- Nothing here tests behaviour. The document can be accurate "
            "about structure and still misdescribe what the code achieves.",
            "- Whether the explanation is *useful* is not something a call "
            "graph can measure."]
    if warnings:
        out.append(f"- {len(warnings)} claim(s) in this document could not be "
                   f"checked either way. They are listed in the verification "
                   f"report as unverifiable.")
    return "\n".join(out)


def symbol_reference(notes: list[SymbolNote]) -> str:
    lines = ["| Symbol | Kind | Role | Location | Purpose |",
             "| --- | --- | --- | --- | --- |"]
    for n in notes:
        purpose = (n.purpose or "").replace("|", "\\|")
        if len(purpose) > 110:
            purpose = purpose[:107] + "…"
        lines.append(
            f"| `{n.qualname.split('.')[-1]}` | {n.kind} | {n.role} | "
            f"`{n.file}:{n.start_line}` | {purpose} |")
    return "\n".join(lines)


def _slug(heading: str) -> str:
    return re.sub(r"[^a-z0-9\s-]", "", heading.lower()).strip().replace(" ", "-")


def build_document(analysis: Analysis, repo_map: RepoMap, draft: Draft,
                   reviews: list[Review], with_alternatives: bool = False) -> str:
    passed = sum(1 for r in reviews if r.passed)
    claims = sum(r.claims_checked for r in reviews)

    parts = [
        f"# {repo_map.repo}",
        "",
        f"*Generated by Legible. {passed}/{len(reviews)} sections "
        f"machine-verified against {len(analysis.symbols)} symbols and "
        f"{len(analysis.edges)} verified calls; {claims} claims checked.*",
        "",
        "## Contents",
        "",
    ]
    started = manifest.render(
        manifest.read_manifest(analysis.root, list(analysis.root.rglob("*.py"))))

    headings = ([s.heading for s in draft.sections]
                + (["Getting started"] if started else [])
                + ["Architecture", "Symbol reference", "Verification report",
                   "What this does not cover"])
    parts += [f"{i}. [{h}](#{_slug(h)})" for i, h in enumerate(headings, 1)]
    parts.append("")

    for s in draft.sections:
        parts += [f"## {s.heading}", "", s.body.strip(), ""]

    if started:
        parts += ["## Getting started", "", started, ""]

    # Off by default on purpose: everything else in this phase is deterministic
    # assembly, and this needs a model call plus network.
    if with_alternatives:
        import alternatives

        man = manifest.read_manifest(analysis.root, [])
        rows = alternatives.find_alternatives(man.dependencies)
        rendered = alternatives.render(rows)
        if rendered:
            parts += ["### If you cannot use these dependencies", "",
                      rendered, ""]

    parts += ["## Architecture", ""]

    # Flow first, same as the app. A file graph says where code lives, which is
    # no use to someone who has never heard of these files. This says what
    # happens. It was in the app and missing from the document for a while.
    flow = markdown_phase_flow(analysis, repo_map)
    if flow:
        parts += ["**What happens, in order.** The main path through this "
                  "project, starting from the way in. The order comes from the "
                  "real call graph.", "", flow, ""]

    modules = mermaid_module_graph(analysis)
    if modules:
        n_edges = len(module_edges(analysis))
        parts += [
            f"**Which files depend on which.** {n_edges} relationships, with "
            f"the number on each arrow counting how many calls cross that "
            f"boundary. Read the heaviest arrows first: they are where the "
            f"work flows.",
            "", modules, ""]

    diagram = mermaid_call_graph(analysis, repo_map.notes)
    if diagram:
        parts += [
            "**Down at the function level.** Starts from the most important "
            "functions and follows their real calls outwards. Every arrow was "
            "read off the source, not inferred by a model.",
            "", diagram, ""]
    elif not modules:
        parts += ["No calls were resolved between files, so there is no flow "
                  "to draw.", ""]

    parts += ["## Symbol reference", "", symbol_reference(repo_map.notes), ""]
    parts += ["## Verification report", "", verification_report(reviews), ""]
    parts += ["## What this does not cover", "",
              limitations(analysis, repo_map, reviews), ""]
    return "\n".join(parts)


def load_artifacts(repo: str) -> tuple[RepoMap, Draft, list[Review]]:
    paths = {name: OUT_DIR / f"{repo}.{name}.json"
             for name in ("map", "draft", "review")}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        sys.exit("Missing artefacts:\n  " + "\n  ".join(missing)
                 + "\nRun the pipeline first: python graph.py <url>")
    return (RepoMap.load(paths["map"]), Draft.load(paths["draft"]),
            Review.load_all(paths["review"]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 9 — assemble the final doc")
    ap.add_argument("repo", help="repo name, e.g. flask")
    ap.add_argument("--url", help="repo URL (defaults to pallets/<repo>)")
    ap.add_argument("--out", type=Path, help="output path")
    ap.add_argument("--alternatives", action="store_true",
                    help="suggest substitutes for each dependency "
                         "(needs a model call and network)")
    args = ap.parse_args()

    repo_map, draft, reviews = load_artifacts(args.repo)
    analysis = analyze(args.url or f"https://github.com/pallets/{args.repo}")

    doc = build_document(analysis, repo_map, draft, reviews,
                         with_alternatives=args.alternatives)
    dest = args.out or (OUT_DIR / f"{args.repo}.FINAL.md")
    dest.write_text(doc, encoding="utf8")

    passed = sum(1 for r in reviews if r.passed)
    print(f"[aggregator] {len(doc.splitlines())} lines, "
          f"{passed}/{len(reviews)} sections verified")
    print(f"[aggregator] diagram: "
          f"{'yes' if '```mermaid' in doc else 'none (no connected symbols)'}")
    print(f"[aggregator] wrote {dest}")


if __name__ == "__main__":
    main()
