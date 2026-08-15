"""Phase 6 — the Writer agent: turn the Mapper's map into prose.

Reads `out/<repo>.map.json` (Phase 5) and never re-runs analysis. Produces a
Draft of independently regenerable sections, because Phase 8's retry loop needs
to rewrite one failing section without discarding the good ones.

Grounding rule: the Writer may only use facts already in the map. Each section
prompt carries an explicit allow-list of symbol names, and Phase 7's Critic
checks the output for names that are not on it.

Run:
    python -m agents.writer https://github.com/pallets/flask
    python -m agents.writer flask --from-map        # skip straight to writing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.mapper import OUT_DIR, Component, RepoMap, SymbolNote, build_map  # noqa: E402
from analyze import analyze  # noqa: E402
from llm import PROSE_MODEL, STATS, chat  # noqa: E402

SYSTEM = (
    "You explain code to someone smart who has never seen this project and is "
    "still fairly new to programming. Think of a patient friend at a whiteboard, "
    "not a textbook. Your reader is busy and will stop reading if it gets heavy.\n"
    "\n"
    "VOICE:\n"
    "- Simple words first. 'Sends the request' beats 'dispatches the request "
    "payload'. If a plain word works, use the plain word.\n"
    "- Short sentences. One idea each. Paragraphs of 2 to 3 sentences, never "
    "more than 4.\n"
    "- Open every section with ONE sentence saying, in everyday language, what "
    "this part is for. Someone should be able to read only that sentence and "
    "still get the gist.\n"
    "- Say what something is FOR before how it works. Purpose, then mechanics.\n"
    "- When you must use a technical term, gloss it in a few words the first "
    "time: 'a WSGI request (the raw web request from the server)'.\n"
    "- A short everyday comparison is welcome when it genuinely helps. Skip it "
    "if it does not.\n"
    "- Keep it warm and light, but never at the cost of being clear.\n"
    "\n"
    "LENGTH:\n"
    "- Shorter is better. If a sentence adds no new information, delete it.\n"
    "- Never restate a fact you already gave. No summary paragraph at the end.\n"
    "- Do not list every function. Cover the few that matter and say how they "
    "connect.\n"
    "\n"
    "BANNED:\n"
    "- Em dashes and en dashes. Not one. Use a comma, a colon, or start a new "
    "sentence.\n"
    "- Corporate filler: robust, powerful, seamless, leverage, utilise, "
    "delve, comprehensive, cutting-edge, 'plays a crucial role', "
    "'is responsible for', 'in conclusion', 'it is important to note'.\n"
    "- Opening three paragraphs in a row with 'This module' or 'The X method'.\n"
    "- Praising the code. Nobody needs to be told a codebase is well structured.\n"
    "- Copying the shape of the notes you are given. Never emit lines like "
    "'Calls (verified):' or 'purpose:', and never end with a bullet list of "
    "symbol names. Those notes are your source, not your format.\n"
    "- Role labels in brackets after a name, like '`foo` (orchestrator)' or "
    "'(entry point)'. Those words are internal notes for you. A reader does not "
    "know what they mean.\n"
    "- Nested bullets. One level at most, and prefer sentences to bullets.\n"
    "- Long dotted paths in body text. Write `url_for`, not "
    "`flask.app.Flask.url_for`, once the context is clear.\n"
    "- Closing paragraphs that restate what you just said in the same order.\n"
    "\n"
    "HARD FACTUAL RULES (these outrank voice, always):\n"
    "- Use ONLY the facts you are given. Never invent a function, file, class, "
    "or behaviour that is not listed. A joke is never worth a wrong fact.\n"
    "- If the facts do not cover something, leave it out rather than guessing.\n"
    "- Backticks are ONLY for real code identifiers and file paths, e.g. "
    "`Flask.wsgi_app`. Never put a title or an English phrase in backticks, "
    "because they get checked against the real symbol list.\n"
    "- Only claim that A calls B when that call is listed as verified.\n"
    "- Do not add a top-level heading, one is added for you."
)

# Prose legitimately contains backticked things that are not repo symbols:
# parameter names, stdlib types, file paths. Only flag identifier-shaped tokens.
_IGNORE_MENTION = re.compile(r"[\s/]|\.py$|^[a-z_]{1,4}$")


def ungrounded_names(section: Section, known: set[str]) -> list[str]:
    """Backticked identifiers that exist nowhere in the repo.

    A cheap deterministic pre-filter for Phase 7: catching invented names costs
    nothing here, so the Critic's LLM budget goes to claims that need judgement.
    """
    bad = []
    for raw in sorted(section.mentioned()):
        token = raw.strip()
        # Quoted text inside backticks is a literal value from an example,
        # e.g. `'Key'` in a dict demo. Never an API name.
        if token[:1] in "'\"" or token[-1:] in "'\"":
            continue
        # Code expressions, not names: `app.config['KEY']`, `as_attachment=True`,
        # `{"a": 1}`. Answers to questions are full of these.
        if any(c in token for c in "[]{}=,"):
            continue
        # A leading dot means a filename or extension: `.env`, `.flaskenv`,
        # `.gitignore`. Real files, never repository symbols, and they were
        # being reported as invented API names because of the dot.
        if token.startswith("."):
            continue
        # `**kwargs` / `*args` are parameters written the way Python spells them.
        token = token.rstrip("()").lstrip("*").strip()
        if not token or _IGNORE_MENTION.search(token):
            continue
        if token.split(".")[-1] not in known and token not in known:
            bad.append(token)
    return bad


@dataclass
class Section:
    key: str  # stable id, so Phase 8 can regenerate exactly one
    heading: str
    body: str
    kind: str  # overview | component | reading-order
    allowed: list[str] = field(default_factory=list)  # qualnames it may cite
    attempts: int = 1

    def mentioned(self) -> set[str]:
        """Backticked identifiers in the body — what the Critic will verify."""
        return {m.strip() for m in re.findall(r"`([^`\n]+)`", self.body)}


@dataclass
class Draft:
    repo: str
    model: str
    sections: list[Section]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def load(path: Path) -> Draft:
        raw = json.loads(path.read_text(encoding="utf8"))
        raw["sections"] = [Section(**s) for s in raw["sections"]]
        return Draft(**raw)

    def to_markdown(self) -> str:
        out = [f"# {self.repo}\n"]
        for s in self.sections:
            out.append(f"## {s.heading}\n\n{s.body.strip()}\n")
        return "\n".join(out)


_LEAKED_FACTS = re.compile(
    r"\n+(?:calls \(verified\)|called by \(verified\)|purpose|takes|returns|"
    r"confidence)\s*:.*", re.I | re.S)

# The role words come from the Mapper's notes. They mean nothing to a reader,
# and the model likes to paste them in after a name.
_ROLE_LABEL = re.compile(
    r"\s*\((?:orchestrator|entry point|shared helper|supporting|type|method|"
    r"function|class)\)", re.I)


def clean_body(text: str) -> str:
    """Tidy what the model returned before anyone reads or verifies it.

    Two fixes, both for things a 7B model does no matter how the prompt is
    worded. It uses em dashes after being told not to, and it sometimes copies
    the prompt's fact-sheet headings ("Calls (verified): - `record`") straight
    into the prose.
    """
    text = _LEAKED_FACTS.sub("", text)
    text = _ROLE_LABEL.sub("", text)
    text = re.sub(r"\s+[—–]\s+", ", ", text)
    return re.sub(r"[—–]", "-", text).strip()


# Kept as the old name so existing imports and tests still resolve.
strip_dashes = clean_body


def _fact_sheet(notes: list[SymbolNote]) -> str:
    lines = []
    for n in notes:
        bits = [f"- `{n.qualname}` ({n.kind}, {n.role}) — {n.file}:{n.start_line}"]
        if n.purpose:
            bits.append(f"  purpose: {n.purpose}")
        if n.inputs and n.inputs.lower() not in ("none", "nothing"):
            bits.append(f"  takes: {n.inputs}")
        if n.returns and n.returns.lower() not in ("none", "nothing"):
            bits.append(f"  returns: {n.returns}")
        if n.calls:
            bits.append("  calls (verified): " + ", ".join(c.split(".")[-1] for c in n.calls))
        if n.confidence == "low":
            bits.append("  NOTE: low confidence — describe cautiously or omit")
        lines.append("\n".join(bits))
    return "\n".join(lines)


def write_overview(m: RepoMap, model: str = PROSE_MODEL) -> Section:
    # Each component carries its own symbols. Listing components and symbols
    # separately let the model pair them up wrongly, e.g. filing `url_for`
    # (which lives in app.py) under "Blueprint utilities".
    by_qual = {n.qualname: n for n in m.notes}
    lines = []
    for c in m.components:
        owned = [by_qual[q].qualname.split(".")[-1]
                 for q in c.symbols if q in by_qual][:4]
        lines.append(f"- {c.title} (in {c.module}): {c.summary}")
        if owned:
            lines.append("  contains ONLY these: " + ", ".join(f"`{o}`" for o in owned))
    comps = "\n".join(lines)

    prompt = (
        f"Repository: {m.repo}\n"
        f"It contains {m.total_symbols} functions/classes with "
        f"{m.total_edges} verified calls between them.\n\n"
        f"Components found:\n{comps}\n\n"
        "Never attribute a function to a component it is not listed under.\n\n"
        "Introduce this repository to someone who has never seen it.\n"
        "- Sentence 1: what this project lets you DO, in everyday language. No "
        "jargon at all in this sentence.\n"
        "- Then 2 short paragraphs: the main parts and what each is for, and "
        "where the important work happens.\n"
        "Keep the whole thing under 150 words. Every statement must come from "
        "the facts above. No em dashes."
    )
    body = strip_dashes(chat(prompt, system=SYSTEM, model=model, temperature=0.3).text)
    return Section("overview", "What this repository is", body, "overview",
                   allowed=[q for c in m.components for q in c.symbols] + m.entry_points)


def write_component(c: Component, notes: list[SymbolNote],
                    model: str = PROSE_MODEL) -> Section:
    prompt = (
        f"Component: {c.title}\nFile: {c.module}\nSummary: {c.summary}\n\n"
        f"The pieces in it:\n{_fact_sheet(notes)}\n\n"
        "Explain this part of the codebase.\n"
        "- Sentence 1: what it is for, in plain words a beginner would follow.\n"
        "- Then walk the flow: what happens first, what happens next, and why. "
        "Follow the handoffs rather than listing functions one by one.\n"
        "Under 120 words. Name the key functions in backticks. Cover only the "
        "pieces listed above. No em dashes."
    )
    body = strip_dashes(chat(prompt, system=SYSTEM, model=model, temperature=0.3).text)
    return Section(f"component:{c.module}", c.title, body, "component",
                   allowed=[n.qualname for n in notes])


def write_reading_order(m: RepoMap, notes: list[SymbolNote],
                        model: str = PROSE_MODEL) -> Section:
    ordered = [n for n in notes if n.role in ("entry point", "orchestrator")] or notes[:5]
    prompt = (
        f"Repository: {m.repo}\n\n"
        f"Good starting points:\n{_fact_sheet(ordered[:6])}\n\n"
        "Tell a newcomer where to start reading.\n"
        "- Two sentences on which file or function to open first, and why that "
        "one makes the rest easier to follow.\n"
        "- Then 3 to 4 numbered steps. Each step: what to read, and in plain "
        "words what it will make click.\n"
        "Under 130 words. Only reference the names listed above. No em dashes."
    )
    body = strip_dashes(chat(prompt, system=SYSTEM, model=model, temperature=0.3).text)
    return Section("reading-order", "Where to start reading", body, "reading-order",
                   allowed=[n.qualname for n in ordered])


def write_draft(m: RepoMap, model: str = PROSE_MODEL,
                max_components: int = 6) -> Draft:
    by_module: dict[str, list[SymbolNote]] = {}
    for n in m.notes:
        by_module.setdefault(n.file, []).append(n)

    sections: list[Section] = []
    print(f"[writer] overview ({model})")
    sections.append(write_overview(m, model))

    for c in m.components[:max_components]:
        notes = by_module.get(c.module, [])
        if not notes:
            continue
        print(f"[writer] section: {c.title}")
        sections.append(write_component(c, notes, model))

    print("[writer] reading order")
    sections.append(write_reading_order(m, m.notes, model))
    return Draft(repo=m.repo, model=model, sections=sections)


def regenerate(section: Section, m: RepoMap, feedback: str,
               model: str = PROSE_MODEL) -> Section:
    """Rewrite one section given Critic feedback. Phase 8's retry hook."""
    by_module = {n.qualname: n for n in m.notes}
    notes = [by_module[q] for q in section.allowed if q in by_module]
    prompt = (
        f"Your previous draft of the section '{section.heading}' was reviewed "
        f"against the real source code and REJECTED.\n\n"
        f"Errors found:\n{feedback}\n\n"
        f"Facts you may use:\n{_fact_sheet(notes)}\n\n"
        f"Previous draft:\n---\n{section.body}\n---\n\n"
        "Rewrite the section. Rules:\n"
        "1. DELETE every sentence the reviewer flagged. Do not reword it, do "
        "not soften it. Remove the claim entirely unless the facts above "
        "prove it.\n"
        "2. A 'does not call' error means that call does not exist. Never "
        "repeat it. Use only the 'calls (verified)' lists above.\n"
        "3. It is better to write less than to repeat a rejected claim.\n"
        "4. Keep the voice from your system instructions. A correction is not "
        "an excuse to fall back on 'X calls A, B and C'. Explain what the "
        "handoff is FOR, in the same readable prose as before. Losing the "
        "personality counts as failing this rewrite."
    )
    body = strip_dashes(chat(prompt, system=SYSTEM, model=model, temperature=0.3).text)
    return Section(section.key, section.heading, body, section.kind,
                   allowed=section.allowed, attempts=section.attempts + 1)


def load_or_build_map(target: str, from_map: bool, top: int) -> RepoMap:
    name = target.rstrip("/").removesuffix(".git").split("/")[-1]
    path = OUT_DIR / f"{name}.map.json"
    if from_map or path.exists():
        if not path.exists():
            sys.exit(f"No map at {path}. Run: python -m agents.mapper {target}")
        print(f"[writer] using existing map {path}")
        return RepoMap.load(path)
    return build_map(analyze(target), top=top)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 6 — write the docs")
    ap.add_argument("target", help="repo URL, or repo name with --from-map")
    ap.add_argument("--from-map", action="store_true",
                    help="require an existing map instead of building one")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--model", default=PROSE_MODEL)
    args = ap.parse_args()

    m = load_or_build_map(args.target, args.from_map, args.top)
    draft = write_draft(m, model=args.model)

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"{m.repo}.draft.json").write_text(draft.to_json(), encoding="utf8")
    md_path = OUT_DIR / f"{m.repo}.md"
    md_path.write_text(draft.to_markdown(), encoding="utf8")

    print(draft.to_markdown())
    print(f"\n[writer] saved {md_path}")
    print(STATS.summary())


if __name__ == "__main__":
    main()
