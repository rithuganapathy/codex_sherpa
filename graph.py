"""Phase 8 — LangGraph wiring: map -> write -> review -> revise -> review ...

The loop is **round-based, not per-section**, and that is the whole trick. Writer
runs on the prose model, Mapper and Critic on the code model, and only one 5GB
model fits in RAM (see llm.py). Revising section-by-section would swap models
twice per section; revising every failing section in one batch swaps twice per
*round* instead.

    analyze -> map -> write -> review --pass--> END
                       ^                |
                       |              --fail-->  revise
                       +-------------------------+

Run:
    python graph.py https://github.com/pallets/flask
    python graph.py https://github.com/pallets/flask --top 8 --max-rounds 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agents.critic import Review, review_section
from agents.mapper import OUT_DIR, RepoMap, build_map
from agents.writer import Draft, Section, regenerate, write_draft
from analyze import Analysis, analyze
from llm import CODE_MODEL, PROSE_MODEL, STATS


class State(TypedDict, total=False):
    url: str
    repo: str
    top: int
    max_rounds: int
    round: int
    reuse: bool

    analysis: Analysis
    repo_map: RepoMap
    draft: Draft
    reviews: list[Review]
    log: list[str]


def _note(state: State, msg: str) -> list[str]:
    print(f"[graph] {msg}")
    return state.get("log", []) + [msg]


class NoPythonError(RuntimeError):
    """The repo cloned fine and contains no Python for us to read."""


def node_analyze(state: State) -> dict[str, Any]:
    a = analyze(state["url"])
    if not a.symbols:
        # Fail here, in a sentence, rather than spending minutes producing an
        # empty map. Phase 1 only collects .py files, so a TypeScript or Go
        # project parses to nothing at all.
        kinds = {}
        for p in a.root.rglob("*.*"):
            if p.is_file() and ".git" not in p.parts:
                kinds[p.suffix] = kinds.get(p.suffix, 0) + 1
        top = ", ".join(f"{n} {ext}" for ext, n in
                        sorted(kinds.items(), key=lambda kv: -kv[1])[:3])
        raise NoPythonError(
            f"{a.root.name} has no Python files this tool can read. "
            f"It is mostly {top}. Only Python is supported today.")
    return {
        "analysis": a,
        "repo": a.root.name,
        "round": 0,
        "log": _note(state, f"analyzed {a.root.name}: "
                            f"{len(a.symbols)} symbols, {len(a.edges)} edges"),
    }


def node_map(state: State) -> dict[str, Any]:
    cached = OUT_DIR / f"{state['repo']}.map.json"
    if state.get("reuse") and cached.exists():
        return {"repo_map": RepoMap.load(cached),
                "log": _note(state, f"reusing map {cached.name}")}
    m = build_map(state["analysis"], top=state.get("top", 10), model=CODE_MODEL)
    OUT_DIR.mkdir(exist_ok=True)
    cached.write_text(m.to_json(), encoding="utf8")
    return {"repo_map": m,
            "log": _note(state, f"mapped {len(m.notes)} symbols into "
                                f"{len(m.components)} components")}


def node_write(state: State) -> dict[str, Any]:
    draft = write_draft(state["repo_map"], model=PROSE_MODEL)
    return {"draft": draft,
            "log": _note(state, f"wrote {len(draft.sections)} sections")}


def node_review(state: State) -> dict[str, Any]:
    a, draft = state["analysis"], state["draft"]
    reviews = [review_section(s, a, CODE_MODEL) for s in draft.sections]
    failed = [r for r in reviews if not r.passed]
    for r in reviews:
        print(f"  {'PASS' if r.passed else 'FAIL'}  {r.heading} "
              f"({r.claims_checked} claims, {len(r.errors)} errors)")
    return {
        "reviews": reviews,
        "log": _note(state, f"round {state.get('round', 0)}: "
                            f"{len(reviews) - len(failed)}/{len(reviews)} passed"),
    }


def node_revise(state: State) -> dict[str, Any]:
    """Rewrite every failing section in one batch — one model swap, not N."""
    draft, reviews, m = state["draft"], state["reviews"], state["repo_map"]
    by_key = {r.section_key: r for r in reviews}

    fixed: list[Section] = []
    revised = 0
    for s in draft.sections:
        r = by_key.get(s.key)
        if r is None or r.passed:
            fixed.append(s)
            continue
        revised += 1
        print(f"[graph] revising: {s.heading}")
        for issue in r.errors:
            print(f"         - {issue.kind}: {issue.detail[:90]}")
        try:
            fixed.append(regenerate(s, m, r.feedback(), model=PROSE_MODEL))
        except Exception as exc:  # keep the old text rather than losing the section
            print(f"[graph] revision failed ({exc}); keeping previous text")
            fixed.append(s)

    return {
        "draft": Draft(repo=draft.repo, model=draft.model, sections=fixed),
        "round": state.get("round", 0) + 1,
        "log": _note(state, f"revised {revised} sections"),
    }


def route_after_review(state: State) -> str:
    failed = [r for r in state["reviews"] if not r.passed]
    if not failed:
        print("[graph] all sections passed -> done")
        return END
    if state.get("round", 0) >= state.get("max_rounds", 2):
        print(f"[graph] {len(failed)} still failing but round limit reached -> done")
        return END
    return "revise"


def build_graph() -> Any:
    g = StateGraph(State)
    g.add_node("analyze", node_analyze)
    g.add_node("map", node_map)
    g.add_node("write", node_write)
    g.add_node("review", node_review)
    g.add_node("revise", node_revise)

    g.set_entry_point("analyze")
    g.add_edge("analyze", "map")
    g.add_edge("map", "write")
    g.add_edge("write", "review")
    g.add_conditional_edges("review", route_after_review,
                            {"revise": "revise", END: END})
    g.add_edge("revise", "review")  # re-check what was just rewritten
    return g.compile()


def save_outputs(state: State) -> Path:
    """Persist a finished run. Called by the CLI *and* the Streamlit app.

    The UI used to drive the graph itself and never wrote anything, so a run
    survived only as long as the browser session: a refresh lost it, it never
    appeared under "already done earlier", and the deep link could not find it.
    """
    repo = state["repo"]
    draft, reviews = state["draft"], state["reviews"]
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"{repo}.draft.json").write_text(draft.to_json(), encoding="utf8")
    (OUT_DIR / f"{repo}.md").write_text(draft.to_markdown(), encoding="utf8")
    (OUT_DIR / f"{repo}.review.json").write_text(Review.dump(reviews), encoding="utf8")
    return OUT_DIR / f"{repo}.md"


def run(url: str, top: int = 10, max_rounds: int = 2, reuse: bool = False) -> State:
    started = time.monotonic()
    final: State = build_graph().invoke(
        {"url": url, "top": top, "max_rounds": max_rounds, "reuse": reuse},
        {"recursion_limit": 50},
    )

    repo = final["repo"]
    reviews = final["reviews"]
    save_outputs(final)

    passed = sum(1 for r in reviews if r.passed)
    print(f"\n{'=' * 62}")
    print(f"DONE  {repo}  {passed}/{len(reviews)} sections verified "
          f"after {final.get('round', 0)} revision round(s)")
    print(f"      {sum(r.claims_checked for r in reviews)} claims checked")
    print(f"      {time.monotonic() - started:.0f}s wall clock")
    print(f"      -> {OUT_DIR / f'{repo}.md'}")
    print(STATS.summary())
    return final


def main() -> None:
    # This run takes minutes. Python block-buffers stdout when it is piped or
    # redirected, so without this the log appears only at the very end and a
    # working pipeline is indistinguishable from a hung one.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    ap = argparse.ArgumentParser(description="Phase 8 — run the full pipeline")
    ap.add_argument("url")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--max-rounds", type=int, default=2)
    ap.add_argument("--reuse", action="store_true",
                    help="reuse a saved map instead of re-running the Mapper")
    args = ap.parse_args()
    run(args.url, args.top, args.max_rounds, args.reuse)


if __name__ == "__main__":
    main()
