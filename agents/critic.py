"""Phase 7 — the Critic agent: check the docs before a human reads them.

The design point of the whole project. A naive critic asks the LLM "is this
correct?", which just adds a second opinion from the same kind of system that
made the mistake. Instead:

    LLM        -> EXTRACT claims from the prose into structured triples
    call graph -> VERIFY each claim, deterministically

The model is used for language (finding the assertions), never for adjudication.
Phase 2's edges are ground truth, so a false claim is caught by lookup, not by
judgement.

Run:
    python -m agents.critic flask                # review the saved draft
    python -m agents.critic flask --demo         # feed it a wrong explanation
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.mapper import OUT_DIR  # noqa: E402
from agents.writer import Draft, Section, ungrounded_names  # noqa: E402
from analyze import Analysis, analyze  # noqa: E402
from llm import CODE_MODEL, STATS, chat_json  # noqa: E402

SYSTEM = (
    "You extract factual claims from technical documentation. You do not judge "
    "whether they are true — you only record what the text asserts, exactly as "
    "written. Extract every claim you find, even obvious ones."
)

CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["calls", "location", "behaviour"]},
                    "subject": {"type": "string"},
                    "object": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["type", "subject", "object", "quote"],
            },
        }
    },
    "required": ["claims"],
}


@dataclass
class Issue:
    kind: str  # unknown-name | false-call | wrong-location | unverifiable
    severity: str  # error | warning
    subject: str
    detail: str
    quote: str = ""


@dataclass
class Review:
    section_key: str
    heading: str
    passed: bool
    claims_checked: int
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @staticmethod
    def dump(reviews: list[Review]) -> str:
        """Single writer for review.json — graph.py and critic.py both use this,
        so the file has one shape no matter which produced it."""
        return json.dumps([asdict(r) for r in reviews], indent=2)

    @staticmethod
    def load_all(path: Path) -> list[Review]:
        out = []
        for raw in json.loads(path.read_text(encoding="utf8")):
            issues = [Issue(**i) for i in raw.pop("issues", [])]
            out.append(Review(**raw, issues=issues))
        return out

    def feedback(self) -> str:
        """Text handed back to the Writer by Phase 8's retry loop."""
        if not self.issues:
            return "No issues found."
        lines = []
        for i in self.issues:
            lines.append(f"- [{i.severity}] {i.kind}: {i.detail}")
            if i.quote:
                lines.append(f'  (from your text: "{i.quote.strip()[:160]}")')
        return "\n".join(lines)


def _index(analysis: Analysis) -> dict[str, list[str]]:
    """Short name -> every qualname that could mean it."""
    idx: dict[str, list[str]] = {}
    for qual in analysis.symbols:
        idx.setdefault(qual.split(".")[-1], []).append(qual)
        idx.setdefault(qual, []).append(qual)
    return idx


def _resolve(name: str, idx: dict[str, list[str]]) -> list[str]:
    token = name.strip().strip("`").rstrip("()").strip()
    if token in idx:
        return idx[token]
    tail = token.split(".")[-1]
    return idx.get(tail, [])


def extract_claims(section: Section, model: str = CODE_MODEL) -> list[dict]:
    prompt = (
        f"Documentation section:\n---\n{section.body}\n---\n\n"
        "List every factual claim it makes about the code.\n"
        "- type 'calls': the text says one function calls/invokes/dispatches to "
        "another. subject = caller, object = callee.\n"
        "- type 'location': the text says something is defined in a file or at a "
        "line. subject = the symbol, object = the file or file:line.\n"
        "- type 'behaviour': any other assertion about what code does. "
        "subject = the symbol, object = the asserted behaviour.\n"
        "Use the exact identifier names as written. quote = the sentence it came from."
    )
    data, _ = chat_json(prompt, system=SYSTEM, model=model, schema=CLAIM_SCHEMA)
    claims = data.get("claims", [])
    return [c for c in claims if isinstance(c, dict) and c.get("subject")]


def verify_claims(claims: list[dict], analysis: Analysis) -> list[Issue]:
    """Every check here is a lookup against Phase 2 facts. No judgement."""
    idx = _index(analysis)
    edges = set(analysis.edges)
    issues: list[Issue] = []

    for c in claims:
        ctype = c.get("type", "behaviour")
        subject = str(c.get("subject", "")).strip()
        obj = str(c.get("object", "")).strip()
        quote = str(c.get("quote", ""))
        subj_quals = _resolve(subject, idx)

        # Subjects that were never identifiers to begin with. A file path
        # ("`flask/app.py` handles requests") or an English phrase ("Request
        # Processing", "the wrapped function") is normal documentation, and
        # reporting them buries the real findings under noise.
        looks_like_path = "/" in subject or subject.endswith(".py")
        looks_like_prose = " " in subject.strip()
        if not subj_quals and (looks_like_path or looks_like_prose) and ctype != "calls":
            continue

        if not subj_quals:
            # Warning, not error, and deliberately so. Claim subjects are often
            # ordinary prose nouns ("the whole flow is defined in..."), which
            # would otherwise fail every section and trap Phase 8 in endless
            # retries. Invented *identifiers* are still errors — ungrounded_names
            # catches those from the backticks, before any LLM runs.
            issues.append(Issue(
                "unknown-name", "warning", subject,
                f"{subject!r} is not a symbol in this repository", quote))
            continue

        if ctype == "calls":
            obj_quals = _resolve(obj, idx)
            if not obj_quals:
                issues.append(Issue(
                    "unverifiable", "warning", subject,
                    f"{subject!r} is said to call {obj!r}, which lives outside "
                    f"this repo, so it cannot be checked either way", quote))
                continue
            if any((s, o) in edges for s in subj_quals for o in obj_quals):
                continue  # verified against the call graph

            # "prepare_url is a method of PreparedRequest" is ownership, not a
            # call. Extraction files these under 'calls' constantly.
            if any(s.startswith(o + ".") or o.startswith(s + ".")
                   for s in subj_quals for o in obj_quals):
                continue

            # Deliberately NOT done here: treating "the reverse edge exists" as
            # an extraction slip. It cannot be distinguished from docs that
            # genuinely state a relationship backwards, which is exactly the
            # error this tool exists to catch. The containment rule above
            # already covers the real-world case that motivated it.
            # If neither name appears in the sentence it came from, extraction
            # built the pair rather than reading it. A real example from click:
            # "This module helps print messages to files or the screen and
            # formats filenames" became "echo calls format_filename", which the
            # prose never said. The same guard already covers location claims.
            short_subj = subject.split(".")[-1].lower()
            short_obj = obj.split(".")[-1].lower()
            quote_l = quote.lower()
            if short_subj not in quote_l and short_obj not in quote_l:
                issues.append(Issue(
                    "unverifiable", "warning", subject,
                    f"a call from {subject!r} to {obj!r} was extracted from a "
                    f"sentence naming neither, so it cannot be attributed",
                    quote))
                continue

            real = sorted({x.split(".")[-1] for s in subj_quals
                           for x in analysis.callees(s)})
            issues.append(Issue(
                "false-call", "error", subject,
                f"{subject!r} does not call {obj!r}. Verified calls from "
                f"{subject!r}: {real or 'none'}", quote))

        elif ctype == "location":
            # An English phrase is not a place. "url_for is at 'technical
            # documentation'" means the extractor filed a behaviour claim under
            # the wrong type, not that the docs got a path wrong.
            bare = obj.split(":")[0].strip()
            if " " in bare and "/" not in bare and not bare.endswith(".py"):
                continue
            # `app = Flask(__name__)` in a usage example arrives as
            # location(app, "Flask(__name__)"). No path contains brackets.
            if any(c in bare for c in "()[]{}=,"):
                continue
            # Nor is a kind a place. "BlueprintSetupState is at 'class'" says
            # what it is, not where it lives.
            if bare.lower() in {"class", "module", "function", "method", "file",
                                "package", "repository", "codebase", "source"}:
                continue
            # "send_from_directory is in flask" is how people refer to a
            # package-level export, and it is true: __init__.py re-exports it.
            if bare in (analysis.root.name, f"{analysis.root.name}.py"):
                continue

            # Location claims are where extraction is least reliable, so the
            # quote is treated as the evidence. Three guards, each added after a
            # real false positive on Flask's own docs:
            want_file = obj.split(":")[0].strip().replace("\\", "/")
            want_line = None
            if ":" in obj:
                tail = obj.rsplit(":", 1)[-1].strip()
                want_line = int(tail) if tail.isdigit() else None

            # 0. "record_once is in flask...blueprints.Blueprint" names the
            #    owning class, not a file. If the claimed container is a real
            #    symbol that this subject lives under, the claim is true.
            container = _resolve(obj.split(":")[0], idx)
            if container and any(
                    s.startswith(c + ".") or s == c
                    for s in subj_quals for c in container):
                continue

            # 1. A module path (`flask.sansio.blueprints`) or a bare module name
            #    is a valid way to say where something lives. Normalise both to a
            #    file path — but leave anything already ending in .py alone, or
            #    `sample.py` turns into `sample/py.py`.
            if want_file and not want_file.endswith(".py") and "/" not in want_file:
                want_file = want_file.replace(".", "/") + ".py"

            # 2. A line number absent from the quoted sentence was invented by
            #    the extractor — the docs never claimed it.
            if want_line is not None and str(want_line) not in quote:
                want_line = None

            # 3. The subject and the location must appear in the SAME sentence,
            #    not merely somewhere in the same quote. A real case: an answer
            #    said "FlaskCliRunner is a CLI runner. The main function in
            #    flask/cli.py is the entry point", and cli.py was attributed to
            #    FlaskCliRunner across the full stop. It never claimed that.
            subj_short = subject.split(".")[-1].lower()
            obj_short = Path(bare).name.lower() or bare.lower()
            sentences = re.split(r"(?<=[.!?])\s+", quote)
            subj_in_quote = any(
                subj_short in s.lower()
                and (obj_short in s.lower() or bare.lower() in s.lower())
                for s in sentences
            ) or (subj_short in quote.lower() and len(sentences) <= 1)

            ok = False
            for q in subj_quals:
                sym = analysis.symbols[q]
                if want_file and not (sym.file.endswith(want_file)
                                      or want_file.endswith(sym.file)
                                      or Path(sym.file).name == Path(want_file).name):
                    continue
                if want_line is not None and not (
                        sym.start_line <= want_line <= sym.end_line):
                    continue
                ok = True
                break
            if not ok:
                actual = "; ".join(
                    f"{analysis.symbols[q].file}:{analysis.symbols[q].start_line}"
                    for q in subj_quals[:3])
                if subj_in_quote:
                    issues.append(Issue(
                        "wrong-location", "error", subject,
                        f"{subject!r} is not at {obj!r}. Actually at {actual}", quote))
                else:
                    issues.append(Issue(
                        "unverifiable", "warning", subject,
                        f"claimed location {obj!r} for {subject!r} could not be tied "
                        f"to the quoted sentence; actual location {actual}", quote))

    return issues


def name_severity(token: str) -> str:
    """How confident are we that an unknown backticked token is a real mistake?

    A fabricated API name almost always looks like one: dotted, snake_case, or
    multi-capital CamelCase. A single plain word is usually an example value
    (`Key` in a dictionary demo) or prose emphasis, and calling that an error
    fails a section nobody can fix.
    """
    looks_like_api = (
        "." in token
        or "_" in token
        or sum(c.isupper() for c in token) >= 2
    )
    return "error" if looks_like_api else "warning"


def known_names(analysis: Analysis) -> set[str]:
    """Everything the docs may legitimately name in backticks.

    Parameter names belong here. Documentation properly written says "it takes a
    `directory` and `**kwargs`" — without params in this set the Critic calls
    those invented names, and Phase 8 then loops forever trying to "fix" prose
    that was correct all along.
    """
    names = {q.split(".")[-1] for q in analysis.symbols} | set(analysis.symbols)
    for sym in analysis.symbols.values():
        names.update(sym.params)

    # Packages, modules and directories. "The `flask` package" and "`app.py`"
    # are ordinary things to write, and neither is a symbol in the table.
    names.add(analysis.root.name)
    for sym in analysis.symbols.values():
        parts = Path(sym.file).parts
        names.update(parts)                                  # src, flask, app.py
        names.update(p.removesuffix(".py") for p in parts)   # app
        names.add(Path(sym.file).stem)
    return names


def prose_only(text: str) -> str:
    """Drop fenced code blocks before verifying.

    A usage example is illustration, not a claim about the repository. It is
    full of names invented for the example (`download_file`, `app.py`), and
    checking them against the symbol table reports the example itself as a
    hallucination. Only the prose asserts anything.
    """
    return re.sub(r"```.*?```", " ", text, flags=re.S)


def self_defined_names(text: str) -> set[str]:
    """Names the text introduces in its own examples.

    An answer that shows a snippet defining `download_file` and then discusses
    `download_file` in the prose is being coherent, not inventing repository
    API. Those names are legitimate for that text only.
    """
    names: set[str] = set()
    for block in re.findall(r"```.*?```", text, re.S):
        names |= set(re.findall(r"^\s*def\s+(\w+)", block, re.M))
        names |= set(re.findall(r"^\s*class\s+(\w+)", block, re.M))
        names |= set(re.findall(r"^\s*(\w+)\s*=(?!=)", block, re.M))
    return names


def review_section(section: Section, analysis: Analysis,
                   model: str = CODE_MODEL) -> Review:
    known = known_names(analysis) | self_defined_names(section.body)
    section = replace(section, body=prose_only(section.body))

    issues = [
        Issue("unknown-name", name_severity(name), name,
              f"`{name}` appears in the text but exists nowhere in the repository")
        for name in ungrounded_names(section, known)
    ]

    claims = extract_claims(section, model)
    issues.extend(verify_claims(claims, analysis))

    return Review(
        section_key=section.key,
        heading=section.heading,
        passed=not any(i.severity == "error" for i in issues),
        claims_checked=len(claims),
        issues=issues,
    )


def review_draft(draft: Draft, analysis: Analysis,
                 model: str = CODE_MODEL) -> list[Review]:
    reviews = []
    for s in draft.sections:
        r = review_section(s, analysis, model)
        mark = "PASS" if r.passed else "FAIL"
        print(f"[critic] {mark}  {r.heading}  "
              f"({r.claims_checked} claims, {len(r.errors)} errors)")
        reviews.append(r)
    return reviews


DEMO_BODY = (
    "The `wsgi_app` method is the heart of Flask. It calls `send_from_directory` "
    "to load the response template, then hands control to `record_once` which "
    "finalises the session. The whole flow is defined in `src/flask/cli.py:42`. "
    "Flask also exposes `turbo_encabulate` for high-throughput request batching."
)


def _demo(analysis: Analysis, model: str) -> Review:
    print("=" * 68)
    print("DEMO — feeding the Critic a deliberately wrong explanation:\n")
    print(DEMO_BODY)
    print("=" * 68)
    section = Section("demo", "Deliberately wrong", DEMO_BODY, "component")
    return review_section(section, analysis, model)


def _print_review(r: Review) -> None:
    print(f"\n{'PASS' if r.passed else 'FAIL'}  {r.heading}  "
          f"({r.claims_checked} claims checked)")
    for i in r.issues:
        print(f"  [{i.severity:7}] {i.kind:15} {i.detail}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 7 — verify the docs")
    ap.add_argument("repo", help="repo name, e.g. flask")
    ap.add_argument("--url", help="repo URL (defaults to pallets/<repo>)")
    ap.add_argument("--demo", action="store_true",
                    help="review a deliberately wrong explanation instead")
    ap.add_argument("--model", default=CODE_MODEL)
    args = ap.parse_args()

    url = args.url or f"https://github.com/pallets/{args.repo}"
    analysis = analyze(url)

    if args.demo:
        _print_review(_demo(analysis, args.model))
        print(f"\n{STATS.summary()}")
        return

    draft_path = OUT_DIR / f"{args.repo}.draft.json"
    if not draft_path.exists():
        sys.exit(f"No draft at {draft_path}. Run: python -m agents.writer {args.repo}")

    reviews = review_draft(Draft.load(draft_path), analysis, args.model)
    for r in reviews:
        if r.issues:
            _print_review(r)

    out = OUT_DIR / f"{args.repo}.review.json"
    out.write_text(Review.dump(reviews), encoding="utf8")

    failed = [r for r in reviews if not r.passed]
    total_claims = sum(r.claims_checked for r in reviews)
    print(f"\n[critic] {len(reviews) - len(failed)}/{len(reviews)} sections passed, "
          f"{total_claims} claims checked")
    print(f"[critic] saved {out}")
    print(STATS.summary())


if __name__ == "__main__":
    main()
