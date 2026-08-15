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


def _diagram_data(analysis: Analysis, notes: list[SymbolNote], max_nodes: int
                  ) -> tuple[dict[str, list[str]], list[tuple[str, str]]]:
    """Pick the nodes and edges both diagram formats draw.

    Only symbols that actually connect are kept. An isolated box teaches a
    reader nothing, and a diagram is worth showing only if it shows flow.
    """
    ranked = [n.qualname for n in notes][:max_nodes]
    chosen = set(ranked)
    edges = [(a, b) for a, b in analysis.edges if a in chosen and b in chosen]

    connected = {q for e in edges for q in e}
    by_file: dict[str, list[str]] = {}
    for q in ranked:
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
        f"*Generated by Codex Sherpa. {passed}/{len(reviews)} sections "
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

    diagram = mermaid_call_graph(analysis, repo_map.notes)
    parts += ["## Architecture", ""]
    if diagram:
        parts += [
            "Calls between the highest-ranked symbols, grouped by file. "
            "Every arrow was extracted from the source, not inferred by a model.",
            "", diagram, ""]
    else:
        parts += ["The highest-ranked symbols have no direct calls between "
                  "them, so there is no flow to draw.", ""]

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
