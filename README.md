# Codex Sherpa

A codebase guide that explains any repo in plain English — and checks its own work before you read it.

## Setup

```bash
venv\Scripts\activate
pip install -r requirements.txt
python ingest.py https://github.com/pallets/flask   # phase 1 — file list
python analyze.py https://github.com/pallets/flask  # phase 2 — call graph
python embed.py https://github.com/pallets/flask    # phase 3 — build search index
python embed.py https://github.com/pallets/flask -q "how requests are routed"
python selfcheck.py                                 # regression checks
```

## Environment notes

Things here that differ from the plan document — check these before debugging a phase:

- **Model tag.** Ollama has `qwen2.5-coder:7b-instruct-q4_K_M` pulled, not the plan's
  `qwen2.5-coder:7b`. Use the full tag in `llm.py`.
- **tree-sitter 0.26.** The API changed from the version most tutorials use.
  Build a parser with `Parser(Language(tree_sitter_python.language()))` — no
  `set_language()`, no `.so` grammar build. `Node.sexp()` is gone; use `str(node)`.
- **torch is CPU-only** (2.13.0+cpu, no CUDA) — this affects Phase 3 embeddings only.
  Flask still indexes in ~18s, so it is not a real constraint at this scale.
- **Ollama runs on the GPU** (`ollama ps` reports 100% GPU) with its own runtime,
  independent of torch. Warm generation is ~12 tok/s.

## Phases

| File | Phase | Status |
| --- | --- | --- |
| `ingest.py` | 1 — clone repo, list source files | done |
| `analyze.py` | 2 — tree-sitter parsing, call graph | done |
| `embed.py` | 3 — chunk + embed into Chroma | done |
| `llm.py` | 4 — Ollama connection helper | done |
| `agents/mapper.py` | 5 — rank + describe what matters | done |
| `agents/writer.py` | 6 — map -> prose sections | done |
| `agents/critic.py` | 7 — verify claims vs the call graph | done |
| `graph.py` | 8 — LangGraph wiring + retry loop | done |
| `aggregator.py` | 9 — final Markdown + diagram | done |
| `app.py` | 10 — Streamlit UI | done |

## How the call graph resolves (Phase 2)

Python is dynamically typed, so a fully correct call graph is impossible without
type inference. `analyze.py` resolves in tiers, most to least reliable:

1. `self.foo()` / `super().foo()` — searched against the class, then its in-repo
   ancestors (breadth-first over `bases`). Reliable.
2. `foo()` — a bare name matching exactly one definition in the repo. Reliable.
3. `x.foo()` on an arbitrary object — matched by method name only. **Heuristic.**
   A repo with two different classes both defining `send()` will mis-attribute.
   Names ambiguous across several definitions are dropped, not guessed.

Known limits, all deliberate:

- **Unresolved is not failure.** ~50% of calls in a typical repo go to third-party
  libraries (`click.echo`, `os.path.dirname`). Only in-repo files are parsed, so
  those correctly have no edge. Judge quality by whether resolved edges are *right*,
  not by the percentage.
- Container methods (`.get`, `.setdefault`, `.append`, …) are in `BUILTINS` and never
  produce edges — otherwise `options.get()` would link to any repo method named `get`.
- Direct recursion is skipped. Decorators are included in a symbol's source span.
- Same name defined twice in a file (conditional defs, `@overload`) keeps the first.

## How search is built (Phase 3)

Model is `all-MiniLM-L6-v2` (384 dims) — small enough to be fast on CPU-only torch.
Flask indexes in ~18s. Vectors are normalized and the collection uses cosine space.

Chunking decisions, and why:

- **One chunk per symbol**, not per N characters. A fixed window would cut functions in
  half and retrieve meaningless fragments. Phase 2 already found the natural boundaries.
- **Long functions are split into overlapping line windows** (~1000 chars, 8-line
  overlap). MiniLM silently truncates past ~256 word-pieces, so without this the tail of
  a long function would never be searchable. Overlap keeps logic from falling in a gap.
- **Each chunk is prefixed with plain-English context** — qualname, file, docstring, and
  its call-graph neighbours. The model was trained on prose, not Python; this is what
  lets a query with no shared keywords still hit.
- **Classes embed as signature + docstring only.** Embedding the whole body would
  duplicate every method it contains and swamp the index with near-identical vectors.
- **`search()` returns distinct symbols.** A long function's chunks all score alike, so
  it over-fetches and keeps each symbol's best chunk — otherwise one function fills
  every slot.

Verified: "remember information about a visitor between page loads" → `sessions`,
"serve a file from disk to the browser" → `send_file`. Neither shares a keyword with
its target, which is the whole point over grep.

Scores land in the 0.26–0.58 range, not 0.9 — normal when matching English prose
against source code. **Rank matters, absolute score does not.** Do not add a score
threshold without testing it; it will silently drop correct answers.

## The model-swap constraint (Phase 4) — read before Phase 8

This machine has 15.3 GB RAM and both models are ~5 GB, so only one stays resident.
Measured:

| | |
| --- | --- |
| Same model, back-to-back | ~2.9s |
| After switching models | ~15s |

A per-symbol loop of Mapper(code) → Writer(prose) → Critic(code) pays ~42s of pure
model loading *per symbol*. **Batch by model instead**: every Mapper call, then every
Writer call, then every Critic call.

`llm.STATS.summary()` counts swaps and warns when it detects thrashing. Escape hatches:

- `SHERPA_SINGLE_MODEL=1` — route prose through the code model, zero swaps
- `SHERPA_KEEP_ALIVE` — default `30m`, up from Ollama's 5m (a pipeline that pauses to
  parse or embed between calls would otherwise return to a cold model)
- `SHERPA_NUM_CTX` — default 8192, up from Ollama's 4096. Code prompts exceed 4096 and
  the overflow is dropped **silently from the front**, which would delete your
  instructions rather than the code.

Use `chat_json(..., schema=...)` for anything machine-read. Structured output
constrains decoding so the JSON is valid by construction; there is a salvage
fallback for fenced/prose-wrapped replies, covered by `selfcheck.py`.

## What the Mapper does (Phase 5)

```bash
python -m agents.mapper https://github.com/pallets/flask --top 10
```

Writes `out/<repo>.map.json` for Phase 6 to consume, so the Writer never re-runs
this. Flask/top-10: 14 LLM calls, ~75s, **0 model swaps**.

The split that makes this work:

| Job | Done by | Why |
| --- | --- | --- |
| Which symbols matter | call-graph arithmetic | deterministic, free, testable |
| What each one does | the LLM, one at a time | needs real language |

The LLM never picks what is important and never invents a relationship. Every
prompt carries verified facts from Phase 2 (`Calls (verified): …`) plus real source,
and every claim is anchored to `file:line` — which is precisely what lets Phase 7's
Critic check it.

**Ranking is not just fan-in.** Two corrections were needed, both found by looking at
real output:

- A repo's public API has *zero* in-repo callers — `Flask`, `Blueprint`, `route` are
  called by users, not by Flask. Ranking on fan-in alone surfaced internal plumbing
  (`_get_current_object`, `_method_route`) and buried the API. Names re-exported from
  `__init__.py` now get a large boost; that list is the authors stating their public
  surface outright. Export scanning reuses Phase 1's `SKIP_DIRS`, or `examples/`
  leaks demo names like `blog` and `auth` into it.
- Fan-in is **capped** at 5. 1 caller vs 5 is a real difference; 9 vs 10 is not.
  Uncapped, one much-used internal hook outscored the entire public API.

Role labels overlap, so order matters: `entry point` (nothing in-repo calls it) is
checked before `orchestrator`, because "start reading here" is the more useful thing
to tell a newcomer.

Truncated source forces `confidence` down from high — the model cannot vouch for code
it was not shown.

## What the Writer does (Phase 6)

```bash
python -m agents.writer flask --from-map     # reuse the Phase 5 map
```

Writes `out/<repo>.md` and `out/<repo>.draft.json`. Flask: 6 calls, ~48s, 0 swaps.

- Reads the saved map; never re-runs analysis or the Mapper.
- Sections are **independently regenerable** (`Section.key`, `regenerate()`), because
  Phase 8 must rewrite one failing section without discarding the good ones. The
  draft JSON round-trip is covered by `selfcheck.py` for the same reason.
- Prose uses plain `chat()`, not `chat_json()` — JSON adds failure modes and buys
  nothing when the payload is markdown. Structured output stays for machine-read data.
- Each prompt carries an allow-list of symbols and the verified call facts.

### Voice

The reader is assumed to be smart but new to the project, and short on patience.
`SYSTEM` asks for plain words, 2-3 sentence paragraphs, a one-sentence
jargon-free opener per section, and purpose before mechanics. Each section prompt
also caps its length (150 words for the overview, 120 for a component).

Two enforcement details, because a 7B model treats style rules as suggestions:
`clean_body()` strips em dashes it used anyway, and removes fact-sheet headings
("Calls (verified):") that it sometimes copies out of the prompt.

`regenerate()` repeats the voice rules. Without that, a correction round produces
"X calls A, B and C" and quietly undoes the writing.

Measured on `requests`, dense voice vs plain voice: 683 -> 533 words, 17.9 -> 15.1
words per sentence, and 5/5 sections verified with **zero** rewrite rounds instead
of two. Simpler prose makes fewer unverifiable claims, so it also costs less.

### Two kinds of hallucination, and only one is cheap to catch

`ungrounded_names()` is a deterministic check: any backticked identifier that exists
nowhere in the repo. It costs nothing, so Phase 7's LLM budget goes to claims that
actually need judgement. Flask currently scores **0 ungrounded names**.

It does **not** catch the harder case. An early draft said to read
`full_dispatch_request` "to see how request contexts are pushed and popped" — every
name real, the relationship false (`request_context` and `push` belong to `wsgi_app`).
That is what the Phase 7 Critic is for, and it is the demo worth keeping.

One prompt fix was needed first: the model backticked component *titles*
(`` `File Sending` ``), which flooded the checker with false positives. Backticks are
now restricted to real identifiers and paths.

## How the Critic works (Phase 7)

```bash
python -m agents.critic flask          # review the saved draft
python -m agents.critic flask --demo   # feed it a deliberately wrong explanation
```

**The LLM never judges correctness.** Asking a model "is this right?" just adds a
second opinion from the same kind of system that made the mistake. Instead:

| Step | Who | Why |
| --- | --- | --- |
| Extract claims into triples | the LLM | finding assertions is a language job |
| Verify each claim | the call graph | Phase 2 edges are ground truth |

A `calls` claim is a set lookup against `analysis.edges`. A `location` claim is a
line-range check. Neither involves an opinion.

The demo catches every planted error, naming what the code *actually* does:

```
[error] unknown-name  `turbo_encabulate` exists nowhere in the repository
[error] false-call    'wsgi_app' does not call 'send_from_directory'.
                      Verified calls: full_dispatch_request, handle_exception,
                      push, request_context
```

### Severity exists to keep Phase 8 from looping forever

A critic that cries wolf is worse than none, because the retry loop would never
converge. The first run on Flask's real docs produced **4 errors, 3 of them false** —
all from location claims, all caused by extraction rather than bad docs. Each fix
below is a guard against a real observed failure:

- A line number **absent from the quoted sentence** was invented by the extractor —
  ignore the line, check the file.
- If the **subject does not appear in its own quote**, extraction bound the location to
  a neighbouring symbol. Report as warning, never error.
- A **module path** (`flask.sansio.blueprints`) or bare module name is a valid location;
  normalise to a file path — but never touch something already ending in `.py`, or
  `sample.py` becomes `sample/py.py` (caught by `selfcheck.py`, not by hand).
- **Prose nouns** ("the whole flow is defined in…") and **file paths as subjects** are
  normal writing, not missing symbols. Invented *identifiers* are still hard errors —
  `ungrounded_names()` catches those from the backticks before any LLM runs.
- **Calls into third-party code** are `unverifiable` warnings. Phase 2 only parses
  in-repo files, so absence of an edge is not evidence of absence.

After those guards: 5/6 sections pass, 51 claims checked, and the remaining failure is
genuine — the docs say `send_from_directory` invokes `send_file`, but the code only
calls `_prepare_send_file_kwargs` and delegates the rest to werkzeug. Flask's own
docstring makes the same loose claim, which is exactly why a machine check is worth
having.

## The full loop (Phase 8)

```bash
python graph.py https://github.com/pallets/flask --top 8 --max-rounds 2 --reuse
```

```
analyze -> map -> write -> review --pass--> END
                    ^                |
                    |             --fail--> revise
                    +------------------------+
```

Flask, top-8: **6/6 sections verified after 2 revision rounds**, 38 claims checked,
349s. The loop demonstrably works — a draft claiming `full_dispatch_request` catches
exceptions via `handle_exception` was rejected, and the rewrite correctly says
`handle_user_exception`.

**Revision is batched by round, not by section.** Writer runs on the prose model,
Mapper and Critic on the code model, and only one fits in RAM. Revising section by
section would swap models twice per section; revising every failing section together
swaps twice per *round*. A 3-round run costs 5 swaps — the floor for this
architecture. `STATS.summary()` now only warns when swaps exceed a quarter of all
calls, so a correctly batched run is not flagged.

### Why convergence needed two fixes

The first full run stalled at 3/6 and never improved. Neither cause was the loop:

1. **The Critic was wrong.** It flagged `` `directory` `` and `` `**kwargs` `` as
   invented names — they are *parameter* names, correct documentation writing. The
   Writer cannot fix prose that was already right, so every round was wasted. Phase 2
   now extracts `Symbol.params`, and `known_names()` includes them. Same class of bug:
   `record_once is at flask...blueprints.Blueprint` names the owning *class*, which is
   a valid location.
2. **The feedback was too polite.** "Fix the issues" produced rewrites that reworded
   the rejected claim instead of deleting it. `regenerate()` now says to delete the
   flagged sentence outright, that a `does not call` error means the call does not
   exist, and that writing less beats repeating a rejected claim.

The lesson worth keeping: **a retry loop is only as good as its verifier.** A critic
with false positives does not just add noise — it makes convergence impossible,
because there is no rewrite that satisfies it.

Also: `graph.py` sets `line_buffering` on stdout. Python block-buffers when piped, so
without it a 6-minute run prints nothing until the end and looks hung.

## Ask the repo a question

```bash
python -m agents.answerer flask "how are session cookies signed?"
```

Also the **Ask** tab in the app. The flow reuses everything already built:

```
question -> Phase 3 semantic search -> real source of the top symbols
         -> model answers from that ONLY
         -> Phase 7 verifies the answer's claims against the call graph
         -> a real example, lifted from the repo's own tests
```

Phase 3 built a search index that nothing consumed until now. This is what it
was for.

**Examples come from the test files.** Phase 1 filters tests out so assertions
and fixtures cannot pollute the call graph; `examples.py` reads them back in for
examples only. They are real, they are quoted, and CI runs them, so there is
nothing to hallucinate. Asking Flask how to send a file returns `test_send_file`
from `tests/test_helpers.py`.

**The weak link is retrieval, not verification.** If the search returns the wrong
functions, the model answers from the wrong code and the check will not save you:
it confirms the claimed calls are real, not that the answer is relevant. That is
why the sources are shown with their scores. If the hits look unrelated to the
question, distrust the answer regardless of the badge.

Answers needed four extra Critic guards that documentation never triggered,
because answers quote usage:

- Fenced code blocks are excluded from verification. A usage example is
  illustration, not a claim about the repo.
- Names the answer defines in its own example (`download_file`) count as known
  for that text.
- Code expressions are not names: `app.config['KEY']`, `as_attachment=True`.
- A call expression is not a location: `app = Flask(__name__)`.

`app.py` imports the answerer **inside** the button handler. At module scope it
pulls in sentence-transformers and torch, which measured 19s that every page view
would pay even for someone who never asks anything.

## The final document (Phase 9)

```bash
python aggregator.py flask        # -> out/flask.FINAL.md
```

**No LLM.** Phase 9 is deterministic assembly of artefacts the earlier phases already
produced, which is why all of it is covered by `selfcheck.py`. It adds three things to
the prose:

- **A Mermaid call-graph diagram**, grouped into per-file subgraphs. Only symbols that
  actually connect are drawn — an isolated box teaches a reader nothing. On Flask it
  recovers the request pipeline from source:
  `wsgi_app -> full_dispatch_request -> finalize_request -> make_response`.
  GitHub renders Mermaid natively, so it needs no build step.
- **A symbol reference table** with `file:line` for everything described.
- **A verification report** — the point of the project. It states how many claims were
  checked, which sections passed, which claims contradict the source, and which could
  not be verified at all. Most AI-written documentation cannot tell you any of that.

Current Flask output: 6/6 sections verified, 37 claims checked, 0 unresolved errors.

Two fixes this phase:

- `graph.py` and `critic.py` were writing `review.json` with **different key names**,
  so the aggregator would break depending on which ran last. `Review.dump()` /
  `Review.load_all()` are now the single reader and writer of that file.
- The "not verifiable" list was 90% component titles ("Request Processing"). An English
  phrase was never an identifier claim, so the Critic now drops multi-word subjects
  outright rather than warning about them — 9 noise entries down to 1.

## The UI (Phase 10)

```bash
streamlit run app.py            # then open http://localhost:8501
```

Sidebar takes a repo URL, symbol count, and revision-round limit. Results arrive in
four tabs — Documentation, Architecture, Verification, Symbols — plus a Markdown
download. `?repo=flask` deep-links straight to a previous run.

Three decisions that mattered:

- **Progress comes from `graph.stream()`.** LangGraph yields after every node, so a
  multi-minute run reports what it is doing (`Verifying claims against the call
  graph`) instead of freezing. Calling `.invoke()` would have been one silent block.
- **Results live in `st.session_state`.** Streamlit re-runs the entire script on every
  widget interaction — without this, switching tabs would restart the pipeline.
- **The diagram is Graphviz, not Mermaid.** `st.graphviz_chart` is built in and renders
  offline; Mermaid would need a CDN, which the app should not depend on. Both formats
  are generated from one `_diagram_data()` helper, and `selfcheck.py` asserts they
  contain the same edges. The Markdown export keeps Mermaid, since GitHub renders it.

One real bug found by actually opening the page: with Streamlit's default
`initial_sidebar_state="auto"` the sidebar did not render at all, leaving a
first-time visitor on a page with nothing to click. Now pinned to `"expanded"`.

Verified in-browser against a cached Flask run: 6/6 sections, 37 claims, 0 errors, and
the call-graph SVG drawing 4 nodes and 3 edges — matching the analysis exactly.

### Theme

Blush/lavender palette in `.streamlit/config.toml`, with component styling and
hand-drawn forget-me-not SVGs in `app.py`. The call-graph diagram uses the same
palette so it does not clash.

Two things that will bite anyone editing this:

- **Tab selectors.** Streamlit 1.61 renders `role="tablist"` and
  `data-testid="stTab"`. Most guides online use `data-baseweb="tab-list"`, which
  matches nothing here and fails silently.
- **`pointer-events: none` on the corner flowers.** They are `position: fixed` over
  the page corners; without it they swallow clicks on whatever sits beneath. Verified
  by hit-testing the buttons, inputs, tabs and checkbox after styling.

Each flower cluster takes a `uid` so its `<defs>` ids stay unique — the same sprig is
drawn six times, and duplicate ids are invalid HTML.
