"""Ask a repository a question, and check the answer before showing it.

The pipeline writes documentation up front. This answers whatever the reader
actually wanted to know, using the same rule as everything else here: the model
may only speak from retrieved source, and the Critic checks what it said.

    question -> Phase 3 semantic search -> real source of the top symbols
             -> model answers from that only
             -> Phase 7 verifies the answer's claims against the call graph

Verified question answering is the point. Plenty of tools will answer a question
about a codebase. This one tells you when its own answer did not survive a check.

Run:
    python -m agents.answerer flask "how are session cookies signed?"
    python -m agents.answerer requests "what happens when a request is redirected?"
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.critic import Issue, Review, review_section  # noqa: E402
from agents.writer import Section  # noqa: E402
from analyze import Analysis, analyze  # noqa: E402
from embed import build_index, has_index, search  # noqa: E402
from examples import Example, find_examples, render as render_examples  # noqa: E402
from llm import CODE_MODEL, STATS, chat  # noqa: E402

# Parsing a repo's test files takes a second or two. Questions come in batches,
# so keep the parsed result for the life of the process.
_TEST_CACHE: dict = {}

# Enough source for the model to answer from, small enough to keep several
# symbols inside the 8192-token window alongside the question.
MAX_SOURCE_CHARS = 1400
DEFAULT_K = 5

SYSTEM = (
    "You answer questions about a codebase using only the source code you are "
    "shown. You are talking to someone who does not know this project.\n"
    "\n"
    "RULES:\n"
    "- Answer ONLY from the code provided. If it does not contain the answer, "
    "say so plainly and name what you would need to look at instead. A short "
    "honest answer beats a confident wrong one.\n"
    "- Lead with the direct answer in one or two sentences. Details after.\n"
    "- Name real functions in backticks. Never invent a name.\n"
    "- Only say A calls B if you can see that call in the code shown.\n"
    "- Plain words, short sentences. No em dashes. Under 150 words."
)


@dataclass
class Answer:
    question: str
    text: str
    sources: list[dict] = field(default_factory=list)
    review: Review | None = None
    examples: list[Example] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return self.review is None or self.review.passed

    @property
    def errors(self) -> list[Issue]:
        return list(self.review.errors) if self.review else []

    @property
    def best_score(self) -> float:
        return max((s["score"] for s in self.sources), default=0.0)


def ensure_index(analysis: Analysis) -> None:
    """Build the search index on first use. Later questions reuse it."""
    if not has_index(analysis.root.name):
        print(f"[ask] first question for {analysis.root.name}, building the "
              f"search index (about 20s)")
        build_index(analysis)


def _context(hits: list[dict], analysis: Analysis) -> str:
    """Real source for each retrieved symbol, plus its verified calls."""
    blocks = []
    for h in hits:
        sym = analysis.symbols.get(h["qualname"])
        if sym is None:
            continue
        src = sym.source
        if len(src) > MAX_SOURCE_CHARS:
            src = src[:MAX_SOURCE_CHARS] + "\n# ... (truncated)"
        calls = [c.split(".")[-1] for c in analysis.callees(h["qualname"])[:8]]
        head = f"### `{h['qualname']}`  ({sym.file}:{sym.start_line})"
        if calls:
            head += "\ncalls (verified): " + ", ".join(calls)
        if sym.docstring:
            head += f"\ndocstring: {sym.docstring[:200]}"
        blocks.append(f"{head}\n```python\n{src}\n```")
    return "\n\n".join(blocks)


def ask(question: str, analysis: Analysis, k: int = DEFAULT_K,
        model: str = CODE_MODEL, verify: bool = True) -> Answer:
    ensure_index(analysis)
    hits = search(analysis.root.name, question, k)

    if not hits:
        return Answer(question, "Nothing in this repository looks related to "
                                "that question.", [], None)

    prompt = (
        f"Question: {question}\n\n"
        f"Here is the most relevant code from the repository:\n\n"
        f"{_context(hits, analysis)}\n\n"
        f"Answer the question using only the code above."
    )
    text = chat(prompt, system=SYSTEM, model=model, temperature=0.1).text

    # An explanation says what something is for. An example shows what calling
    # it looks like, which is usually the thing that makes it click. Prefer a
    # symbol the answer actually talked about over the top search hit.
    mentioned = {m.strip().split(".")[-1].rstrip("()")
                 for m in re.findall(r"`([^`\n]+)`", text)}
    candidates = [h["qualname"] for h in hits
                  if h["qualname"].split(".")[-1] in mentioned]
    candidates += [h["qualname"] for h in hits if h["qualname"] not in candidates]

    found: list[Example] = []
    for qual in candidates[:4]:
        found = find_examples(analysis.root, qual, cache=_TEST_CACHE)
        if found:
            break

    review = None
    if verify:
        # Same verification path the documentation goes through: claims are
        # extracted, then checked against the call graph rather than judged.
        review = review_section(
            Section(key="answer", heading=question[:60], body=text,
                    kind="component",
                    allowed=[h["qualname"] for h in hits]),
            analysis, model)

    return Answer(question, text, hits, review, found)


def _print(a: Answer) -> None:
    print("\n" + "=" * 68)
    print(f"Q: {a.question}")
    print("=" * 68)
    print(textwrap.fill(a.text, 76, replace_whitespace=False))

    if a.examples:
        e = a.examples[0]
        print(f"\nA test in this repo that uses `{e.uses}`  "
              f"({e.file}:{e.start_line}):\n")
        for line in e.source.splitlines()[:20]:
            print(f"  {line}")

    print("\nSources (what the answer was allowed to read):")
    for s in a.sources:
        print(f"  {s['score']:.3f}  {s['qualname']}  ({s['file']}:{s['start_line']})")

    if a.review is None:
        print("\n[unverified]")
        return
    if a.verified:
        print(f"\nVERIFIED  {a.review.claims_checked} claims checked against "
              f"the call graph, none contradicted the source.")
    else:
        print(f"\nNOT VERIFIED  {len(a.errors)} claim(s) contradict the source:")
        for i in a.errors:
            print(f"  - {i.detail}")
    for i in a.review.warnings:
        print(f"  (unverifiable) {i.detail[:110]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask a repository a question")
    ap.add_argument("repo", help="repo name, or a full GitHub URL")
    ap.add_argument("question", nargs="+")
    ap.add_argument("-k", type=int, default=DEFAULT_K, help="symbols to retrieve")
    ap.add_argument("--url", help="repo URL if the name alone is ambiguous")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    url = args.url or (args.repo if args.repo.startswith("http")
                       else f"https://github.com/pallets/{args.repo}")
    analysis = analyze(url)
    answer = ask(" ".join(args.question), analysis, args.k,
                 verify=not args.no_verify)
    _print(answer)
    print(f"\n{STATS.summary()}")


if __name__ == "__main__":
    main()
