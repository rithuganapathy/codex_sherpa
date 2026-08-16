"""Phase 10 — Streamlit UI: paste a repo URL, get verified documentation back.

Progress comes from `graph.stream()`, which yields after each LangGraph node, so
a run that takes minutes shows what it is doing instead of freezing. Results are
kept in `st.session_state` because Streamlit re-runs this whole script on every
widget interaction — without that, clicking a tab would restart the pipeline.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from agents.critic import Review
from agents.mapper import OUT_DIR, RepoMap
from agents.writer import Draft
import manifest
from aggregator import (build_document, graphviz_call_graph,
                        graphviz_module_graph, graphviz_phase_flow,
                        module_edges)
from analyze import analyze, analyze_files
from graph import NoPythonError, build_graph, save_outputs
from ingest import list_python_files
from llm import CODE_MODEL, PROSE_MODEL, LLMError, available_models

st.set_page_config(
    page_title="Codex Sherpa",
    page_icon="🧭",
    layout="wide",
    # The sidebar holds every control this app has. Streamlit's "auto" default
    # can start it collapsed, which leaves a first-time visitor on a page with
    # nothing to click.
    initial_sidebar_state="expanded",
)

PALETTE = {
    "pink": "#E7508F",
    "pink_soft": "#FBD5E7",
    "lavender": "#A87BD4",
    "lavender_soft": "#EADCF8",
    "peach": "#FF9E7D",
    "mint": "#5FC9A8",
    "ink": "#4A2C40",
    # Forget-me-not: sky-blue petals, gold eye, and pink buds — the buds are
    # genuinely pink before they open, which is what ties them to the palette.
    "petal": "#8FC1EA",
    "petal_pale": "#BCDCF6",
    "eye": "#FFD873",
    "bud": "#F0A6C8",
    "leaf": "#9CCBA6",
}

# Petal centres: five circles at 72° steps starting at 12 o'clock, radius 7.
_PETALS = [(0, -7), (6.66, -2.16), (4.11, 5.66), (-4.11, 5.66), (-6.66, -2.16)]


def flower_cluster(uid: str, size: int = 150) -> str:
    """A sprig of forget-me-nots as inline SVG.

    `uid` keeps the <defs> ids unique — the same cluster is drawn six times on
    the page, and duplicate ids are invalid HTML.
    """
    petals = "".join(
        f'<circle cx="{x}" cy="{y}" r="5.4"/>' for x, y in _PETALS
    )
    blooms = "".join(
        f'<g transform="translate({x},{y}) scale({s}) rotate({r})">'
        f'<use href="#p{uid}" fill="{PALETTE[fill]}" '
        f'stroke="#6FA8DC" stroke-width="0.7"/>'
        f'<circle r="2.5" fill="{PALETTE["eye"]}"/>'
        f'<circle r="1" fill="#FFF6DA"/></g>'
        for x, y, s, r, fill in (
            (40, 44, 1.15, 0, "petal"),
            (72, 30, 0.85, 25, "petal_pale"),
            (60, 66, 1.0, -15, "petal"),
            (26, 74, 0.7, 40, "petal_pale"),
        )
    )
    buds = "".join(
        f'<circle cx="{x}" cy="{y}" r="{r}" fill="{PALETTE["bud"]}" '
        f'opacity="0.9"/>'
        for x, y, r in ((90, 52, 3.4), (84, 70, 2.6), (46, 22, 2.8))
    )
    return f"""
<svg width="{size}" height="{size}" viewBox="0 0 120 120" fill="none"
     xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false">
  <defs><g id="p{uid}">{petals}</g></defs>
  <g stroke="{PALETTE['leaf']}" stroke-width="1.8" stroke-linecap="round"
     fill="none" opacity="0.85">
    <path d="M14 108 C 30 92, 34 66, 40 46"/>
    <path d="M14 108 C 40 96, 62 84, 72 34"/>
    <path d="M14 108 C 38 100, 54 86, 60 68"/>
    <path d="M20 100 C 34 96, 40 88, 44 78"/>
  </g>
  <g fill="{PALETTE['leaf']}" opacity="0.5">
    <ellipse cx="30" cy="92" rx="10" ry="4.5" transform="rotate(-28 30 92)"/>
    <ellipse cx="52" cy="98" rx="8" ry="3.8" transform="rotate(14 52 98)"/>
  </g>
  {buds}{blooms}
</svg>"""

STYLE = f"""
<style>
/* ---- chrome we do not want ----
   Hide the pieces, not the toolbar itself. The button that reopens a collapsed
   sidebar lives inside stToolbar, so hiding the whole thing left no way back
   once the sidebar was closed. */
[data-testid="stToolbarActions"], [data-testid="stAppDeployButton"],
[data-testid="stMainMenu"], [data-testid="stMainMenuButton"],
[data-testid="stStatusWidget"], [data-testid="stDecoration"],
#MainMenu, footer {{
    display: none !important;
}}
/* Keep the way back to the sidebar, and keep it clickable above the flowers. */
[data-testid="stExpandSidebarButton"] {{
    display: flex !important;
    visibility: visible !important;
    z-index: 100;
}}

/* Streamlit reserves a large top margin for that toolbar. With its contents
   gone the page should start where the content starts. */
[data-testid="stAppViewBlockContainer"], .block-container {{
    padding-top: 1.2rem !important;
}}
[data-testid="stHeader"] {{ height: 0; background: transparent; }}

/* "Press Enter to apply" under every input. */
[data-testid="InputInstructions"] {{ display: none !important; }}

/* ---- headline ---- */
.sherpa-hero {{
    background: linear-gradient(120deg, #FFE3F1 0%, #F3E3FB 50%, #FFE8DC 100%);
    border-radius: 22px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.4rem;
    border: 1px solid {PALETTE['pink_soft']};
    box-shadow: 0 8px 22px rgba(231, 80, 143, 0.12);
}}
.sherpa-hero h1 {{
    margin: 0;
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, {PALETTE['pink']}, {PALETTE['lavender']} 60%, {PALETTE['peach']});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.sherpa-hero p {{
    margin: 0.4rem 0 0;
    color: #8A5C77;
    font-size: 1.02rem;
}}

/* ---- buttons ---- */
.stButton > button, .stDownloadButton > button {{
    border-radius: 999px;
    border: none;
    font-weight: 650;
    padding: 0.55rem 1.3rem;
    background: linear-gradient(90deg, {PALETTE['pink']}, {PALETTE['lavender']});
    color: #fff;
    box-shadow: 0 4px 14px rgba(231, 80, 143, 0.30);
    transition: transform .15s ease, box-shadow .15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 7px 20px rgba(231, 80, 143, 0.42);
    color: #fff;
}}

/* ---- metric cards ---- */
[data-testid="stMetric"] {{
    background: #fff;
    border: 1px solid {PALETTE['pink_soft']};
    border-radius: 18px;
    padding: 1rem 1.1rem;
    box-shadow: 0 4px 14px rgba(168, 123, 212, 0.10);
}}
[data-testid="stMetricValue"] {{
    color: {PALETTE['pink']};
    font-weight: 800;
}}
[data-testid="stMetricLabel"] p {{
    color: #9A6E88;
    font-weight: 600;
}}

/* ---- tabs ----
   Streamlit 1.61 renders role="tablist" / data-testid="stTab". Older guides use
   data-baseweb="tab-list", which silently matches nothing here. */
[data-testid="stTabs"] [role="tablist"] {{
    gap: 6px;
    background: {PALETTE['lavender_soft']};
    padding: 6px;
    border-radius: 999px;
    border-bottom: none;
}}
[data-testid="stTab"] {{
    border-radius: 999px;
    padding: 0.35rem 1.1rem;
    font-weight: 600;
    color: #7A5270;
}}
[data-testid="stTab"][aria-selected="true"] {{
    background: #fff;
    color: {PALETTE['pink']} !important;
    box-shadow: 0 2px 8px rgba(231, 80, 143, 0.18);
}}
[data-testid="stTab"][aria-selected="true"] p {{
    color: {PALETTE['pink']} !important;
}}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{
    background: transparent !important;
}}

/* ---- sidebar ---- */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #FDEAF4 0%, #F1E4FB 100%);
    border-right: 1px solid {PALETTE['pink_soft']};
}}
[data-testid="stSidebar"] h2 {{
    color: {PALETTE['pink']};
    font-weight: 800;
    margin-bottom: 0.35rem;
}}

/* The repo box is the first thing anyone touches, so it reads as a search
   field rather than a form input. */
[data-testid="stSidebar"] [data-testid="stTextInput"] input {{
    border-radius: 999px;
    border: 1px solid {PALETTE['pink_soft']};
    background: #fff;
    padding: 0.55rem 1rem;
    font-size: 0.92rem;
    box-shadow: 0 2px 10px rgba(231, 80, 143, 0.10);
}}
[data-testid="stSidebar"] [data-testid="stTextInput"] input:focus {{
    border-color: {PALETTE['pink']};
    box-shadow: 0 0 0 3px rgba(231, 80, 143, 0.14);
}}

/* ---- containers ---- */
[data-testid="stExpander"] {{
    border: 1px solid {PALETTE['pink_soft']};
    border-radius: 16px;
    background: #fff;
}}
[data-testid="stGraphVizChart"] {{
    background: #fff;
    border: 1px solid {PALETTE['pink_soft']};
    border-radius: 18px;
    padding: 0.8rem;
}}
h2, h3 {{ color: {PALETTE['ink']}; }}
hr {{ border-color: {PALETTE['pink_soft']}; }}

/* ---- forget-me-nots ----
   pointer-events:none is essential: these sit over the page corners, and
   without it they would swallow clicks on anything beneath them. */
.sherpa-corner {{
    position: fixed;
    z-index: 0;
    pointer-events: none;
    opacity: 0.55;
    filter: drop-shadow(0 3px 8px rgba(168, 123, 212, 0.18));
}}
/* Only the right-hand pair are fixed to the viewport. The left corners sit
   underneath the sidebar, which is opaque — so those two are rendered inside
   the sidebar instead (see .sherpa-sb-flower). */
.sherpa-corner--tr {{ top: -14px;    right: -16px;  transform: scaleX(-1); }}
.sherpa-corner--br {{ bottom: -16px; right: -16px;  transform: scale(-1, -1); }}

/* overflow-x matters: the sprigs are inset with negative offsets, and without
   this they widen the sidebar's scroll box and shove its controls off-screen. */
[data-testid="stSidebar"] {{ position: relative; overflow-x: hidden; }}
.sherpa-sb-flower {{
    position: absolute;
    pointer-events: none;
    opacity: 0.45;
    z-index: 0;
}}
.sherpa-sb-flower--top {{ top: -14px; left: -8px; }}
.sherpa-sb-flower--bottom {{ bottom: -10px; left: -6px; transform: scaleY(-1); }}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
    position: relative;
    z-index: 1;
}}

@media (max-width: 900px) {{
    /* On a narrow screen the corners crowd the controls. */
    .sherpa-corner {{ display: none; }}
}}

.sherpa-hero {{ position: relative; overflow: hidden; }}
.sherpa-hero .sherpa-sprig {{
    position: absolute;
    pointer-events: none;
    opacity: 0.7;
}}
.sherpa-hero .sherpa-sprig--l {{ top: -10px; left: -8px; }}
.sherpa-hero .sherpa-sprig--r {{
    bottom: -12px; right: -6px; transform: scale(-1, -1);
}}
.sherpa-hero > * {{ position: relative; z-index: 1; }}
</style>
"""

CORNERS = "".join(
    f"<div class='sherpa-corner sherpa-corner--{pos}'>"
    f"{flower_cluster(pos, 165)}</div>"
    for pos in ("tr", "br")
)

SIDEBAR_FLOWERS = "".join(
    f"<div class='sherpa-sb-flower sherpa-sb-flower--{pos}'>"
    f"{flower_cluster('sb' + pos, 140)}</div>"
    for pos in ("top", "bottom")
)

NODE_LABELS = {
    "analyze": "Reading the source, all of it",
    "map": "Working out what actually matters",
    "write": "Writing it up",
    "review": "Fact-checking every sentence",
    "revise": "Sending the wrong bits back for a rewrite",
}


def repo_name(url: str) -> str:
    return url.rstrip("/").removesuffix(".git").split("/")[-1]


def cached_repos() -> list[str]:
    if not OUT_DIR.exists():
        return []
    return sorted(p.name.removesuffix(".draft.json")
                  for p in OUT_DIR.glob("*.draft.json"))


def run_pipeline(url: str, top: int, max_rounds: int, reuse: bool,
                 subdir: str = "") -> dict:
    """Stream the graph so each node reports as it finishes."""
    started = time.monotonic()
    graph = build_graph()
    state: dict = {}

    with st.status("Starting…", expanded=True) as status:
        for chunk in graph.stream(
            {"url": url, "top": top, "max_rounds": max_rounds,
             "reuse": reuse, "subdir": subdir},
            {"recursion_limit": 50},
        ):
            for node, update in chunk.items():
                state.update(update)
                status.update(label=NODE_LABELS.get(node, node))
                st.write(f"**{NODE_LABELS.get(node, node)}**")

                if node == "analyze":
                    a = update["analysis"]
                    from ingest import suggest_subdirs

                    st.session_state["areas"] = suggest_subdirs(a.root)
                    st.caption(f"{len(a.symbols)} symbols, {len(a.edges)} "
                               f"verified calls")
                elif node == "map":
                    m = update["repo_map"]
                    st.caption(f"{len(m.notes)} symbols described, "
                               f"{len(m.components)} components")
                elif node == "write":
                    st.caption(f"{len(update['draft'].sections)} sections drafted")
                elif node == "review":
                    revs = update["reviews"]
                    ok = sum(1 for r in revs if r.passed)
                    st.caption(f"{ok}/{len(revs)} sections verified, "
                               f"{sum(r.claims_checked for r in revs)} claims checked")
                    for r in revs:
                        if not r.passed:
                            st.caption(f"  · {r.heading}: "
                                       + "; ".join(i.detail[:80] for i in r.errors))

        if state.get("draft") and state.get("reviews"):
            saved = save_outputs(state)
            st.caption(f"Saved to {saved.name}, so a refresh will not lose it.")

        status.update(label=f"Done in {time.monotonic() - started:.0f}s",
                      state="complete", expanded=False)
    return state


def load_cached(repo: str) -> dict | None:
    paths = {n: OUT_DIR / f"{repo}.{n}.json" for n in ("map", "draft", "review")}
    if not all(p.exists() for p in paths.values()):
        return None
    return {
        "repo": repo,
        "repo_map": RepoMap.load(paths["map"]),
        "draft": Draft.load(paths["draft"]),
        "reviews": Review.load_all(paths["review"]),
    }


def render_results(state: dict) -> None:
    repo_map, draft, reviews = state["repo_map"], state["draft"], state["reviews"]
    analysis = state.get("analysis")
    if analysis is None:
        # Parse the clone the map already points at. Deriving a URL from the
        # repo name would guess the wrong owner (psf/requests is not
        # pallets/requests) and this needs no network at all.
        root = Path(repo_map.root)
        if root.exists():
            with st.spinner("Re-reading the source for the diagram…"):
                analysis = analyze_files(root, list_python_files(root))
        elif state.get("url"):
            with st.spinner("Fetching the repository again…"):
                analysis = analyze(state["url"])
        state["analysis"] = analysis

    # Metrics about the reader's repository, not about our checking of it.
    # "41 claims fact-checked" describes this tool's machinery, which is not
    # what someone came here to learn. Verification still runs and still
    # rewrites bad sections; it just no longer has a scoreboard.
    errors = sum(len(r.errors) for r in reviews)
    files = len({s.file for s in analysis.symbols.values()}) if analysis else 0
    deps = len(manifest.read_manifest(analysis.root, []).dependencies) if analysis else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Functions and classes", len(analysis.symbols) if analysis else "—")
    c2.metric("Source files", files or "—")
    c3.metric("Calls mapped", len(analysis.edges) if analysis else "—")
    c4.metric("Dependencies", deps if deps else "none")

    doc = build_document(analysis, repo_map, draft, reviews) if analysis else None

    tabs = st.tabs(["Documentation", "Ask", "Getting started", "Architecture",
                    "Symbols", "Limits"])

    with tabs[0]:
        # Silence would be worse than a scoreboard: if a claim is known to be
        # wrong, say so here rather than hiding it with the report.
        if errors:
            st.warning(f"{errors} statement(s) below could not be squared with "
                       f"the source. They are marked in the downloaded file.")
        for s in draft.sections:
            st.subheader(s.heading)
            st.markdown(s.body)
        if doc:
            st.download_button("Take it with you (Markdown)", doc,
                               file_name=f"{repo_map.repo}.md",
                               mime="text/markdown")

    # A fragment reruns only itself. Without it, clicking Ask reruns the whole
    # script, and st.tabs always reopens on the first tab, so the answer landed
    # on a page the reader had been thrown off. It looked like the first click
    # did nothing and the second one worked.
    @st.fragment
    def _ask_panel(analysis):
        st.caption("Ask anything about this repo. The answer is read from the "
                   "real source, then checked before you see it.")
        if analysis is None:
            st.info("The source is not available locally, so questions cannot "
                    "be answered.")
        else:
            # A form so the box empties itself once the question is asked, and
            # so Enter submits. Left as a plain input it kept the previous
            # question sitting there looking like it had been typed again.
            with st.form("ask_form", clear_on_submit=True, border=False):
                q = st.text_input("Your question", key="ask_q",
                                  placeholder="how are session cookies signed?")
                asked = st.form_submit_button("Ask")
            if asked and q.strip():
                with st.spinner("Searching the code, answering, then checking "
                                "the answer…"):
                    try:
                        # Imported here, not at module scope: this pulls in
                        # sentence-transformers and torch, which costs 19s. The
                        # whole app would pay that on every page view even for
                        # someone who never asks a question.
                        from agents.answerer import ask

                        st.session_state["answer"] = ask(q.strip(), analysis)
                    except Exception as exc:
                        st.exception(exc)

            ans = st.session_state.get("answer")
            if ans:
                # Three outcomes, not two. A background answer has no review at
                # all, because nothing in a call graph can confirm what SQLite
                # is. Assuming a review always existed crashed this tab.
                if not ans.from_source:
                    st.info("Answered from general knowledge. There is nothing "
                            "in this repository to check it against, so it "
                            "carries no verification.")
                elif ans.review is None:
                    st.warning("Read from the source, but not checked.")
                elif ans.verified:
                    st.success(f"Verified: {ans.review.claims_checked} claims "
                               f"checked against the call graph, none "
                               f"contradicted the source.")
                else:
                    st.error(f"{len(ans.errors)} claim(s) in this answer "
                             f"contradict the source. Read it with that in mind.")
                st.markdown(ans.text)

                for i in ans.errors:
                    st.error(f"**{i.kind}** — {i.detail}")

                if ans.examples:
                    from examples import render as render_examples

                    st.markdown(render_examples(ans.examples))

                # Sources are only meaningful when the answer came from them.
                if ans.from_source and ans.sources:
                    with st.expander("What the answer was allowed to read"):
                        st.caption("If these look unrelated to your question, "
                                   "distrust the answer no matter what the "
                                   "badge says.")
                        for s in ans.sources:
                            st.markdown(f"`{s['score']:.3f}`  `{s['qualname']}` "
                                        f"— {s['file']}:{s['start_line']}")
                if ans.review and ans.review.warnings:
                    with st.expander(f"{len(ans.review.warnings)} thing(s) that "
                                     f"could not be checked"):
                        for i in ans.review.warnings:
                            st.caption(i.detail)

    with tabs[1]:
        _ask_panel(analysis)

    with tabs[2]:
        started = ""
        if analysis is not None:
            started = manifest.render(manifest.read_manifest(
                analysis.root, list(analysis.root.rglob("*.py"))))
        if started:
            # No caption here: render() opens with its own provenance line, and
            # saying it twice reads like the page does not trust itself.
            st.markdown(started, unsafe_allow_html=True)

            st.divider()

            # Same reason as the Ask panel: without a fragment this button
            # reruns the page and drops the reader back on the first tab.
            @st.fragment
            def _alternatives_panel(root):
                st.caption("Cannot use one of these dependencies? Ask for "
                           "substitutes. Takes about a minute: a model proposes "
                           "names, then each is checked against PyPI.")
                if st.button("Find alternatives", key="alt_btn"):
                    with st.spinner("Proposing, then checking each name on "
                                    "PyPI…"):
                        try:
                            import alternatives
                            import manifest as _man

                            deps = _man.read_manifest(root, []).dependencies
                            st.session_state["alts"] = alternatives.render(
                                alternatives.find_alternatives(deps))
                        except Exception as exc:
                            st.exception(exc)
                if st.session_state.get("alts"):
                    st.markdown(st.session_state["alts"], unsafe_allow_html=True)

            _alternatives_panel(analysis.root)
        else:
            st.info("This repo has no README install steps, no dependency list "
                    "and no obvious entry point, so there is nothing honest to "
                    "put here.")

    with tabs[3]:
        flow = graphviz_phase_flow(analysis, repo_map) if analysis else ""
        mods = graphviz_module_graph(analysis) if analysis else ""
        dot = graphviz_call_graph(analysis, repo_map.notes) if analysis else ""

        # Flow first. A file graph tells you where code lives, which means
        # nothing if you have never heard of ctx.py. This says what happens.
        if flow:
            st.markdown("**What happens, in order**")
            st.caption("The main path through this project, from the way in. "
                       "The order comes from the real call graph.")
            st.graphviz_chart(flow, use_container_width=False)

        if mods:
            st.divider()
            n = len(module_edges(analysis))
            st.markdown("**Which files depend on which**")
            st.caption(f"{n} relationships. The number on each arrow counts how "
                       f"many calls cross that boundary, and thicker means more.")
            st.graphviz_chart(mods, use_container_width=False)

        if dot:
            st.divider()
            st.markdown("**Down at the function level**")
            st.caption("Starts from the most important functions and follows "
                       "their real calls outwards. Every arrow was read off the "
                       "source. None were guessed.")
            # Not use_container_width: Streamlit stretches the SVG to the full
            # column, which blows a small graph up to unreadable proportions.
            st.graphviz_chart(dot, use_container_width=False)

        if not (flow or mods or dot):
            st.info("No calls were resolved between these functions, so there "
                    "is no flow worth drawing. Nudge the coverage slider up and "
                    "the picture usually appears.")

    with tabs[4]:
        st.dataframe(
            [{"Symbol": n.qualname.split(".")[-1], "Kind": n.kind, "Role": n.role,
              "Location": f"{n.file}:{n.start_line}", "Score": n.score,
              "Purpose": n.purpose} for n in repo_map.notes],
            use_container_width=True, hide_index=True)

    with tabs[5]:
        # This repo's rough edges, not ours. Ours stay in the downloaded file,
        # where someone judging the document benefits from them.
        if analysis is not None:
            from agents.mapper import public_api_names
            from insights import repo_limits

            st.caption("What to know about this codebase before you rely on it. "
                       "Every number is counted from the source.")
            st.markdown(repo_limits(analysis, public_api_names(analysis))
                        .as_markdown())
        else:
            st.info("The source is not available locally, so this cannot be "
                    "measured.")


st.markdown(STYLE, unsafe_allow_html=True)
st.markdown(CORNERS, unsafe_allow_html=True)
st.markdown(
    "<div class='sherpa-hero'>"
    f"<div class='sherpa-sprig sherpa-sprig--l'>{flower_cluster('hl', 92)}</div>"
    f"<div class='sherpa-sprig sherpa-sprig--r'>{flower_cluster('hr', 78)}</div>"
    "<h1>Codex Sherpa</h1>"
    "<p>Point it at a codebase nobody has time to explain to you. "
    "It reads the whole thing, writes it up, then marks its own homework "
    "in red pen.</p>"
    "</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(SIDEBAR_FLOWERS, unsafe_allow_html=True)
    st.header("let him cook")
    # Empty, not pre-filled. A default URL sitting in the box reads as though
    # the choice has been made for you.
    url = st.text_input("repo", "", placeholder="paste a github repo…",
                        label_visibility="collapsed")
    # Some repos hold more than one project. unslothai/unsloth carries a web app
    # in studio/ that is 13x the size of the library, and without this the
    # ranking documents the wrong half.
    subdir = st.text_input("folder", "", placeholder="whole repo",
                           label_visibility="collapsed",
                           help="Leave empty to read everything. Set it when a "
                                "repo holds several projects.")
    _areas = st.session_state.get("areas") or []
    if _areas:
        st.caption("Biggest areas: "
                   + ", ".join(f"`{d}` ({n})" for d, n in _areas[:4]))

    top = st.slider("How many functions to cover", 5, 30, 8,
                    help="More coverage, more waiting. Eight is a good first look.")
    max_rounds = st.slider("Rewrite attempts", 0, 3, 2,
                           help="How many times a section gets sent back before "
                                "we stop arguing with it.")
    reuse = st.checkbox("Reuse earlier analysis", value=True,
                        help="Skips the slow reading step for a repo you have "
                             "already run. It also ignores the slider above, so "
                             "untick it if you just changed that.")

    go = st.button("Explain this codebase", type="primary",
                   use_container_width=True)

    # Only surfaced when something is actually wrong. Which models this tool
    # runs on is our plumbing, not the reader's problem.
    try:
        installed = available_models()
        missing = [m for m in (CODE_MODEL, PROSE_MODEL) if m not in installed]
        if missing:
            st.warning("Model not installed: " + ", ".join(f"`{m}`" for m in missing))
    except LLMError as exc:
        st.error(str(exc).split("\n")[0])

    # No standing list of past runs. It only matters when the repo in the box
    # happens to be one already done, so it appears then and stays out of the
    # way otherwise.
    _name = repo_name(url) if url.strip() else ""
    if _name and _name in cached_repos():
        if st.button(f"Open the saved {_name} instead", use_container_width=True):
            loaded = load_cached(_name)
            if loaded:
                st.session_state["result"] = loaded
            else:
                st.error(f"Incomplete artefacts for {_name}")

    st.divider()
    st.caption("Give it a few minutes. Both models run on your own machine, so "
               "nothing leaves it, and nothing arrives faster than your GPU can "
               "think.")

if go:
    if not url.strip():
        st.error("Enter a repository URL.")
    else:
        try:
            result = run_pipeline(url.strip(), top, max_rounds, reuse,
                                  subdir.strip())
            result["url"] = url.strip()
            st.session_state["result"] = result
        except FileNotFoundError as exc:
            st.error(str(exc))
        except NoPythonError as exc:
            st.error(str(exc))
            st.caption("Adding another language means a tree-sitter grammar "
                       "and a few Python-specific rules in the parser. The rest "
                       "of the pipeline does not care what language it reads.")
        except LLMError as exc:
            st.error(f"Ollama problem: {exc}")
        except Exception as exc:  # surface it instead of a blank page
            st.exception(exc)

# Deep link: ?repo=flask opens a previously generated doc directly, so a result
# can be shared or bookmarked without re-running the pipeline.
_linked = st.query_params.get("repo")
if _linked and "result" not in st.session_state:
    _loaded = load_cached(_linked)
    if _loaded:
        st.session_state["result"] = _loaded
    else:
        st.warning(f"No saved run for “{_linked}”. Generate it first.")

if "result" in st.session_state:
    render_results(st.session_state["result"])
else:
    st.info("Paste a repo URL in the sidebar and press **Explain this codebase**. "
            "If you have run that repo before, a second button appears offering "
            "the saved copy, which is instant.")
    st.markdown(
        "#### How it works\n"
        "The repo gets cloned and parsed into a call graph: who calls whom, "
        "read straight off the syntax tree rather than guessed at.\n\n"
        "Then three passes over it. The first works out which functions carry "
        "their weight and describes them. The second turns that into something "
        "worth reading. The third is the interesting one. It pulls every "
        "factual claim out of the writing and checks it against the call graph. "
        "*Does `wsgi_app` really call `full_dispatch_request`?* If not, the "
        "sentence goes back for a rewrite.\n\n"
        "Which means the thing you read at the end has already had an argument "
        "with itself, and lost a few rounds. That happens quietly, and the full "
        "record of what was checked comes with the downloaded file."
    )
