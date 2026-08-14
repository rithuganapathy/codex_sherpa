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
    "You are a senior engineer writing up an unfamiliar codebase for a new "
    "teammate. You are good company: dry, specific, quietly funny, and never "
    "silly. You have read the code and you are not impressed by jargon.\n"
    "\n"
    "VOICE:\n"
    "- Write like a person talking, not a spec sheet. Short sentences carry "
    "more weight than long ones.\n"
    "- Humour comes from being accurate about something slightly absurd in the "
    "code, never from a joke bolted onto the end. At most one light touch per "
    "few paragraphs. If nothing is funny, just be clear.\n"
    "- Say what something is FOR before saying what it does.\n"
    "- Confident and plain. 'This runs on every request' beats 'This method is "
    "responsible for handling the processing of incoming requests'.\n"
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
        # `**kwargs` / `*args` are parameters written the way Python spells them.
        token = raw.strip().rstrip("()").lstrip("*").strip()
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


def clean_body(text: str) -> str:
    """Tidy what the model returned before anyone reads or verifies it.

    Two fixes, both for things a 7B model does no matter how the prompt is
    worded. It uses em dashes after being told not to, and it sometimes copies
    the prompt's fact-sheet headings ("Calls (verified): - `record`") straight
    into the prose.
    """
    text = _LEAKED_FACTS.sub("", text)
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
    comps = "\n".join(f"- {c.title} ({c.module}): {c.summary}" for c in m.components)
    entries = "\n".join(f"- `{q}`" for q in m.entry_points[:8])
    prompt = (
        f"Repository: {m.repo}\n"
        f"It contains {m.total_symbols} functions/classes with "
        f"{m.total_edges} verified calls between them.\n\n"
        f"Components found:\n{comps}\n\n"
        f"Entry points (nothing inside the repo calls these):\n{entries}\n\n"
        "Write 2-3 short paragraphs introducing this repository to someone who "
        "starts on Monday and has to be useful by Wednesday. What is it, how is "
        "it laid out, and where does the real work happen. Open with something "
        "more interesting than a restatement of the name. Every statement must "
        "come from the facts above. No em dashes."
    )
    body = strip_dashes(chat(prompt, system=SYSTEM, model=model, temperature=0.3).text)
    return Section("overview", "What this repository is", body, "overview",
                   allowed=[q for c in m.components for q in c.symbols] + m.entry_points)


def write_component(c: Component, notes: list[SymbolNote],
                    model: str = PROSE_MODEL) -> Section:
    prompt = (
        f"Component: {c.title}\nFile: {c.module}\nSummary: {c.summary}\n\n"
        f"The pieces in it:\n{_fact_sheet(notes)}\n\n"
        "Write 1-2 paragraphs on what this part of the codebase is for and how "
        "its pieces hand off to each other. Follow the actual flow rather than "
        "listing functions one by one. Name the important ones in backticks. "
        "Describe nothing that is not listed above. No em dashes."
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
        "Write one paragraph on where to start reading and why that spot, then "
        "a numbered list of 3-5 steps. Give each step a reason, not just an "
        "instruction: say what the reader will understand once they have read "
        "it. Only reference the names listed above. No em dashes."
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
